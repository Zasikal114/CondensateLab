"""Web adapters for the supplied models. All CH quantities are dimensionless."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
from matplotlib import colormaps
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Literal
from vendor.condensate4 import CahnHilliard
from vendor import li_light, li_frap
from vendor.condensate4 import initialize_grid


class PhaseParams(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    N: int = Field(64, ge=16, le=256, strict=True)
    L: float = Field(64, ge=16, le=256)
    l: float = Field(1, ge=0.001, le=5)
    kappa: float = Field(2, ge=0.01, le=20)
    dt: float = Field(0.2, ge=0.001, le=1)
    steps: int = Field(1500, ge=10, le=10000)
    m0: float = Field(0, ge=-3, le=3)
    mean: float = Field(-0.3, ge=-3, le=3)
    w: float = Field(1, ge=0.1, le=2)
    noise: float = Field(0.05, ge=0, le=0.3)
    seed: int = Field(42, ge=0, le=2147483647)
    initial: Literal['random', 'spot'] = 'random'


def energy(ch, u):
    """Spectral gradient, consistent with the periodic spectral solver."""
    uh = np.fft.fft2(u)
    gx = np.fft.ifft2(1j * ch.KX * uh).real
    gy = np.fft.ifft2(1j * ch.KY * uh).real
    f = ch.l * ((u-ch.m0)**2-ch.w**2)**2/4
    return float(np.sum(f + ch.kappa*(gx*gx+gy*gy)/2) * ch.h**2)


def periodic_stats(u, threshold, h):
    labels, n = ndimage.label(u > threshold)
    if not n:
        return 0, 0.0
    parent = list(range(n+1))
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in list(zip(labels[0], labels[-1])) + list(zip(labels[:, 0], labels[:, -1])):
        if a and b:
            parent[root(int(a))] = root(int(b))
    counts = np.bincount(labels.ravel(), minlength=n+1)
    areas = {}
    for i in range(1, n+1):
        r = root(i)
        areas[r] = areas.get(r, 0)+counts[i]
    return len(areas), float(np.mean([np.sqrt(a/np.pi)*h for a in areas.values()]))


def simulate_phase(p: PhaseParams, out: Path, progress=lambda x: None):
    out.mkdir(parents=True, exist_ok=True)
    # The upstream guard applies even when D_eff >= 0. A sufficiently small
    # constructor dt avoids that overly conservative guard; each actual step
    # below checks the real Fourier denominator and energy.
    deff = 3*p.m0**2-p.w**2
    init_dt = min(p.dt, p.kappa / (p.l*deff)**2) if deff else p.dt
    ch = CahnHilliard(N=p.N, L=p.L, l=p.l, kappa=p.kappa,
                     dt=init_dt, potential='quartic', w=p.w, m0=p.m0)
    rng = np.random.default_rng(p.seed)
    u = p.noise*rng.standard_normal((p.N, p.N))
    if p.initial == 'spot':
        x, y = np.indices(u.shape)
        u += 0.5*p.w*np.exp(-((x-p.N/2)**2+(y-p.N/2)**2)/(2*(p.N/12)**2))
    u += p.mean-u.mean()  # independently choose the actual conserved mean
    initial_mean = float(u.mean())
    e = energy(ch, u)
    frames, rows = [], []
    schedule = set(np.unique(np.linspace(0, p.steps, min(97, p.steps+1)).astype(int)))
    t, dt, rejected = 0.0, p.dt, 0
    cmap = colormaps['viridis']
    def preview(i, frame_file=None):
        rgba = cmap(np.clip((u-(p.m0-p.w))/(2*p.w), 0, 1))
        img = Image.fromarray((rgba[:, :, :3]*255).astype('uint8')).resize((256, 256), Image.Resampling.NEAREST)
        # Publish an immutable image, then expose its name through status.
        # Overwriting an image being read by a browser fails on Windows.
        preview_file=frame_file or f'preview_{i:07}.png'
        img.save(out/preview_file, format='PNG')
        progress({'percent':round(100*i/p.steps,1),'step':i,'total_steps':p.steps,
                  'simulation_time':t,'energy':e,'mean_drift':float(u.mean()-initial_mean),'preview':True,'preview_file':preview_file})
        return img
    def record(i):
        count, radius = periodic_stats(u, p.m0, ch.h)
        rows.append([i, t, e, float(u.mean()-initial_mean), count, radius, dt])
        fname = f'frame_{len(frames):03}.png'
        preview(i, fname)
        frames.append({'file': fname, 'time': t})
        with (out/'frames.jsonl').open('a',encoding='utf8') as stream: stream.write(json.dumps(frames[-1])+'\n')
    record(0)
    start = time.perf_counter()
    last_preview = start
    for i in range(1, p.steps+1):
        if (out/'cancel.flag').exists():
            raise InterruptedError('Simulation cancelled. The latest preview is retained, but this is not a completed result.')
        for attempt in range(24):
            ch.dt = dt
            ch.implicit = 1+dt*ch.K2*(p.l*ch.D_eff+p.kappa*ch.K2)
            if ch.implicit.min() <= 0:
                dt *= .5; rejected += 1; continue
            with np.errstate(over='ignore', invalid='ignore'):
                candidate = ch.step(u)
                enew = energy(ch, candidate) if np.isfinite(candidate).all() else float('inf')
            if np.isfinite(enew) and enew <= e+max(1e-9, abs(e)*1e-10):
                break
            dt *= .5; rejected += 1
        else:
            raise ValueError('No stable step found. Reduce dt / l, increase kappa, or refine the grid.')
        if dt < 1e-10:
            raise ValueError('Time step became too small; these parameters are not numerically usable.')
        u, e, t = candidate, enew, t+dt
        if i in schedule:
            record(i); last_preview=time.perf_counter()
        elif time.perf_counter()-last_preview >= 60:
            preview(i); last_preview=time.perf_counter()
    np.savetxt(out/'field.csv', u, delimiter=',')
    np.save(out/'field.npy', u)
    with (out/'diagnostics.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['step','time','free_energy','mean_drift','components','mean_equivalent_radius','dt']);writer.writerows(rows)
    images=[Image.open(out/fr['file']).convert('RGB') for fr in frames]
    images[0].save(out/'evolution.gif', save_all=True, append_images=images[1:], duration=70, loop=0)
    for img in images: img.close()
    resolution = np.sqrt(2*p.kappa/(p.l*p.w*p.w))/ch.h
    warnings=[]
    if resolution < 2: warnings.append('Interface width is below 2 grid cells; refine N or increase kappa. Morphology is under-resolved.')
    if rejected: warnings.append(f'Adaptive stepping reduced dt to {dt:g}; use actual recorded time, not steps × requested dt.')
    if abs(p.mean-p.m0) >= p.w/np.sqrt(3): warnings.append('The mean is outside the spinodal interval; random small noise may not nucleate droplets.')
    result={'kind':'phase','parameters':p.model_dump(),'elapsed_seconds':time.perf_counter()-start,'frames':frames,
            'table':{'headers':['step','time','energy','mean drift','components','radius','dt'],'data':rows},
            'warnings':warnings,'interface_cells':float(resolution),'actual_final_time':t,
            'max_mean_drift':max(abs(r[3]) for r in rows),'files':['evolution.gif','field.csv','field.npy','diagnostics.csv','parameters.json']}
    (out/'parameters.json').write_text(json.dumps(p.model_dump(),indent=2),encoding='utf-8')
    return result


class FluidParams(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    N: int = Field(64, ge=16, le=256, strict=True)
    on_time: float=Field(250,ge=5,le=500)
    off_time: float=Field(150,ge=5,le=500)
    cycles: int=Field(3,ge=1,le=5)
    dt: float=Field(.5,ge=.1,le=.5)
    k_on: float=Field(.02,ge=.001,le=.2)
    k_off: float=Field(.02,ge=.001,le=.2)
    D_off: float=Field(.1,ge=.01,le=2)
    seed: int=Field(42,ge=0,le=2147483647)

def simulate_fluid(p,out,progress=lambda x:None):
    # Same light-cycle equations and fixed material parameters as Li Pengbo's source.
    ch=CahnHilliard(N=p.N,L=100,D=2.2,k=1,l=.05,kappa=.035,dt=p.dt,potential='quartic',w=1.5,m=.04,m0=1.)
    rng=np.random.default_rng(p.seed)
    u=initialize_grid(p.N,seed='random',rng=rng,mean=.5)
    mean=float(u.mean()); switch=li_light.ProteinStateSwitch(p.k_on,p.k_off,0)
    on_steps=round(p.on_time/p.dt); off_steps=round(p.off_time/p.dt)
    total=p.cycles*(on_steps+off_steps)
    schedule=set(np.linspace(1,total,min(96,total)).astype(int))
    frap_steps=set(np.linspace(1,total,7).astype(int))
    frames=[];rows=[];frap=[]
    for i in range(1,total+1):
        if (out/'cancel.flag').exists(): raise InterruptedError('Simulation cancelled')
        on=li_light.light_on_at(i,on_steps,off_steps)
        if on and (i==1 or not li_light.light_on_at(i-1,on_steps,off_steps)):
            noise=rng.standard_normal(u.shape);u+=.01*(noise-noise.mean())
        phi=switch.advance(on,p.dt)
        u=li_light.light_step(ch,u,phi,p.D_off,p.dt)
        if not np.isfinite(u).all(): raise ValueError('Unstable field. Reduce the time step.')
        if i in schedule:
            rgb=(colormaps['viridis'](np.clip((u+.5)/3,0,1))[:,:,:3]*255).astype('uint8')
            im=Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST)
            canvas=Image.new('RGB',(256,280),'#162d3a');canvas.paste(im,(0,24));ImageDraw.Draw(canvas).text((8,6),f'{"ON" if on else "OFF"}  t={i*p.dt:.1f}  phi={phi:.3f}',fill='white');im=canvas
            name=f'frame_{len(frames):03}.png';im.save(out/name)
            frames.append({'file':name,'time':i*p.dt,'light_on':bool(on),'activation':float(phi)})
            with (out/'frames.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(frames[-1])+'\n')
            count,radius=periodic_stats(u,1,ch.h)
            rows.append([i*p.dt,int(on),float(phi),count,float((u>1).mean()),float(u.mean()-mean)])
            progress({'percent':round(i/total*100,1),'step':i,'total_steps':total,'simulation_time':i*p.dt,'preview':True,'preview_file':name})
        if i in frap_steps:
            # The original assay bleaches a fixed central 3x3 region, including dilute regions.
            params=dict(D_mu=.5,tau=2.,c0=1.,p=8.,alpha=3.,uth=None,dt=.05,T_pre=50.,T_max=300.)
            half,baseline,reached=li_frap.frap_half_time(ch,u,params)
            frap.append([i*p.dt,float(half) if reached else None,float(1/half) if reached and half>0 else None,bool(u[p.N//2,p.N//2]>1)])
    images=[Image.open(out/f['file']).convert('RGB') for f in frames]
    images[0].save(out/'evolution.gif',save_all=True,append_images=images[1:],duration=70,loop=0)
    for im in images:im.close()
    np.savetxt(out/'light_cycles.csv',rows,delimiter=',',header='time,light_on,activation,droplets,condensed_fraction,mean_drift',comments='')
    with (out/'frap.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['time','half_recovery_time','inverse_half_time','center_in_dense_phase']);writer.writerows(frap)
    (out/'parameters.json').write_text(json.dumps(p.model_dump(),indent=2),encoding='utf8')
    valid=[r for r in frap if r[2] is not None]
    return {'kind':'fluidity','parameters':p.model_dump(),'frames':frames,'files':['evolution.gif','light_cycles.csv','frap.csv','parameters.json'],
            'source':'Li Pengbo: light_off/condensate_light.py + FRAP_by_time.py',
            'plots':[{'title':'Active fraction','x':[r[0] for r in rows],'series':[{'name':'φ','y':[r[2] for r in rows]}]},
                     {'title':'Condensed fraction','x':[r[0] for r in rows],'series':[{'name':'u > 1','y':[r[4] for r in rows]}]},
                     {'title':'FRAP: 1 / half-recovery time','x':[r[0] for r in valid],'series':[{'name':'1 / t½','y':[r[2] for r in valid]}]}],
            'warnings':['FRAP uses a frozen field and a fixed central 3×3 bleach region. The region may be in the dilute phase; see center_in_dense_phase in frap.csv. Times and parameters are in model units.'],
            'max_mean_drift':max(abs(r[5]) for r in rows)}

