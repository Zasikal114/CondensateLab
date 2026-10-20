"""Reduced, mass-conserving droplet transport model for interactive exploration.

This is a coarse-grained Langevin/gated-transfer model, not a CH solver or a
calibrated prediction of cellular import. See docs/droplet_transport.md.
"""
import json
import math
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from nuclear_chain_view import render_chain

N=256
PM=78
NE=184


class Transport:
    def __init__(self,p):
        self.p=p;self.rng=np.random.default_rng(p.seed);self.time=0.
        self.gates=np.arange(24,N-16,p.gate_spacing,dtype=float)
        self.drops=[];self.total_mass=9000.;self.pool=self.total_mass
        for i in range(p.droplet_count):
            r=float(self.rng.uniform(3.5,6));mass=math.pi*r*r
            for attempt in range(150):
                x=float(self.rng.uniform(10,N-10));y=float(self.rng.uniform(12,PM-13))
                if all(self.distance(x,y,d)>r+d['r']+1 for d in self.drops):break
            self.drops.append(dict(id=i,x=x,y=y,m=mass,r=r,phase=0,q=0.,gate=0.,vx=0.,vy=0.))
            self.pool-=mass
        if self.pool<=0:raise ValueError('The droplet count exceeds the available material budget.')
        self.next_id=p.droplet_count

    @staticmethod
    def distance(x,y,d):
        dx=(x-d['x']+N/2)%N-N/2
        return float(np.hypot(dx,y-d['y']))

    def nearest_gate(self,x):
        delta=(self.gates-x+N/2)%N-N/2;i=int(np.argmin(abs(delta)))
        return float(self.gates[i]),float(delta[i])

    def step(self,dt):
        p=self.p
        for d in self.drops:
            x0,y0=d['x'],d['y'];r=d['r'];phase=d['phase']
            if phase==0 and self.pool>0:
                dm=min(self.pool,p.growth_rate*2*math.pi*r*self.pool/(N*PM)*dt)
                d['m']+=dm;self.pool-=dm;d['r']=r=math.sqrt(d['m']/math.pi)
            noise=math.sqrt(2*p.brownian/r*dt)
            dx,dy=self.rng.normal(0,noise,2)
            if phase in (0,2):
                wall=PM if phase==0 else NE
                gate,gx=self.nearest_gate(d['x'])
                distance=max(0,wall-d['y']-r)
                capture=p.capture_strength*math.exp(-distance/32)
                drift=capture if phase==0 else p.transport_velocity+capture*.35
                d['x']=(d['x']+dx+np.clip(gx,-10,10)*capture*.16*dt)%N
                d['y']+=dy+drift*dt
                d['y']=max(r+2 if phase==0 else PM+r+3,d['y'])
                if d['y']>=wall-r*.6:
                    # A membrane arrests normal motion away from an entry site.
                    gate,gx=self.nearest_gate(d['x'])
                    d['y']=wall-r*.6
                    if abs(gx)<max(p.entry_half_width,r*.65):
                        d.update(phase=1 if phase==0 else 3,q=0.,gate=gate,x=gate)
            elif phase in (1,3):
                wall=PM if phase==1 else NE
                # Overdamped transfer against a radius-dependent constriction cost.
                width=p.entry_half_width
                mobility=p.membrane_permeability if phase==1 else p.nuclear_permeability
                resistance=1+(r/max(width,.5))**2
                dq=mobility*(1+p.capture_strength*.4)*dt/(max(r,2)*resistance)
                d['q']=min(1,d['q']+dq)
                d['x']=d['gate'];d['y']=wall+(2*d['q']-1)*(r+2)
                if d['q']>=1:d.update(phase=2 if phase==1 else 4,y=wall+r+3,q=0.)
            else:
                d['x']=(d['x']+dx*.45)%N
                # Nuclear confinement with a smooth repulsive envelope boundary.
                d['y']+=dy*.4+.04*dt
                d['y']=float(np.clip(d['y'],NE+r+3,N-r-3))
            d['vx']=(d['x']-x0+N/2)%N-N/2;d['vy']=d['y']-y0
        # Coalescence is triggered by contact, not by prescribed animation timing.
        gone=set()
        for i,a in enumerate(self.drops):
            if i in gone or a['phase'] not in (0,2,4):continue
            for j in range(i+1,len(self.drops)):
                b=self.drops[j]
                if j in gone or b['phase']!=a['phase']:continue
                if self.distance(a['x'],a['y'],b)<.72*(a['r']+b['r']):
                    m=a['m']+b['m'];dx=(b['x']-a['x']+N/2)%N-N/2
                    a['x']=(a['x']+dx*b['m']/m)%N;a['y']=(a['y']*a['m']+b['y']*b['m'])/m
                    a['m']=m;a['r']=math.sqrt(m/math.pi);gone.add(j)
        self.drops=[d for i,d in enumerate(self.drops) if i not in gone]
        self.time+=dt
        if abs(self.pool+sum(d['m'] for d in self.drops)-self.total_mass)>1e-7:
            raise ValueError('The material-conservation check failed.')

    def density(self):
        p=self.p;field=np.zeros((N,N),dtype=np.float64)
        # Soluble exterior material is included in the same material ledger.
        field[4:PM-3,:]=self.pool/((PM-7)*N)
        def deposit(x,y,rx,ry,mass,compartment=None):
            if mass<=0:return
            # Periodic x copies keep mass and appearance continuous at the boundary.
            yy,xx=np.mgrid[max(0,int(y-ry-4)):min(N,int(y+ry+5)),0:N]
            if yy.size==0:return
            dx=(xx-x+N/2)%N-N/2
            q=np.sqrt((dx/max(rx,.4))**2+((yy-y)/max(ry,.4))**2)
            kernel=.5*(1-np.tanh(np.clip((q-1)*4.5,-20,20)))
            gate_x=np.min(abs((self.gates[:,None]-np.arange(N)[None,:]+N/2)%N-N/2),axis=0)<=p.entry_half_width
            for wall in [PM,NE]:kernel[(abs(yy-wall)<=1)&~gate_x[None,:]]=0
            if compartment==0:kernel[yy>=PM-1]=0
            elif compartment==2:kernel[(yy<=PM+1)|(yy>=NE-1)]=0
            elif compartment==4:kernel[yy<=NE+1]=0
            weight=kernel.sum()
            if weight>0:field[yy[:,0],:]+=kernel*(mass/weight)
        for d in self.drops:
            r=d['r'];phase=d['phase']
            if phase in (1,3):
                q=d['q'];wall=PM if phase==1 else NE;neck=.22*math.sin(math.pi*q)
                lower=(1-q)*(1-neck);upper=q*(1-neck)
                for fraction,sign in [(lower,-1),(upper,1)]:
                    radius=max(.6,r*math.sqrt(fraction))
                    deposit(d['x'],wall+sign*(radius*.65+1.5),radius,radius,d['m']*fraction)
                deposit(d['x'],wall,min(p.entry_half_width*.75,r*.4),r*.85,d['m']*neck)
            else:
                stretch=1+min(.25,abs(d['vy'])*.6)
                deposit(d['x'],d['y'],r/math.sqrt(stretch),r*math.sqrt(stretch),d['m'],compartment=phase)
        if not np.isclose(field.sum(),self.total_mass,rtol=1e-10):
            raise ValueError('The rendered concentration field failed its material check.')
        return field


def simulate_transport(p,out,progress):
    started=time.perf_counter();s=Transport(p);dt=.3;steps=int(math.ceil(p.transport_duration/dt))
    sample=set(np.linspace(0,steps,72).round().astype(int));fields=[];rows=[];snapshots=[];frames=[];images=[]
    yy,xx=np.mgrid[:N,:N];transit=(abs(yy-PM)<=2)|(abs(yy-NE)<=2)
    masks=[(yy<PM-2), (yy>PM+2)&(yy<NE-2),transit,yy>NE+2]
    geometry=SimpleNamespace(n_phys=N,length_per_cell=None,pm_y=PM,ne_y=NE,ne_half=2,pm_half=3,pm_centers=s.gates,ne_centers=s.gates)
    def record(i):
        density=s.density();phi=-1.1+2.5*density
        fractions=[float(density[mask].sum()/s.total_mass) for mask in masks]
        row=[s.time]+fractions+[float(density.sum()/s.total_mass-1),0,float(density.sum())]
        rows.append(row);fields.append(phi.astype(np.float32));snapshots.append({'time':s.time,'soluble_mass':s.pool,'droplets':[dict(d) for d in s.drops]})
        im=render_chain(geometry,phi,s.time,rows);name=f'frame_{len(frames):03}.png';im.save(out/name)
        frames.append({'file':name,'time':s.time});images.append(im)
        with (out/'frames.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(frames[-1])+'\n')
        progress({'percent':round(90*i/steps,1),'step':i,'total_steps':steps,'simulation_time':s.time,'preview':True,'preview_file':name,'backend':'numpy'})
    record(0)
    for i in range(1,steps+1):
        if i%20==0 and (out/'cancel.flag').exists():raise InterruptedError('Simulation cancelled.')
        s.step(dt)
        if i in sample:record(i)
    # Select one informative pore, then keep that exact crop for the entire replay.
    cx=max(s.gates,key=lambda c:max(float(np.maximum(f[NE+1:NE+20,max(0,int(c)-6):min(N,int(c)+7)],0).sum()) for f in fields))
    images=[render_chain(geometry,f,r[0],rows,rows[-1][0],zoom_x=int(cx)) for f,r in zip(fields,rows)]
    progress(94)
    for im,frame in zip(images,frames):im.save(out/frame['file'])
    images[0].save(out/'nuclear_import.gif',save_all=True,append_images=images[1:],duration=100,loop=0,optimize=False,disposal=2)
    images[-1].save(out/'nuclear_final.png')
    headers=['model_time','extracellular_fraction','cytoplasm_fraction','membrane_fraction','nuclear_fraction','mass_error','source_added','total_mass']
    np.savetxt(out/'nuclear_spatial.csv',rows,delimiter=',',header=','.join(headers),comments='')
    np.savetxt(out/'nuclear_field.csv',fields[-1],delimiter=',')
    np.savez_compressed(out/'nuclear_fields.npz',fields=np.stack(fields),times=np.array(rows)[:,0],pm_y=PM,ne_y=NE,pore_centers=s.gates,density_offset=-1.1,density_scale=2.5,total_mass=s.total_mass)
    (out/'droplet_states.json').write_text(json.dumps(snapshots),encoding='utf8')
    parameters=p.model_dump();parameters.update(engine='coarse_grained_droplet_transport_v1',dt=dt,total_mass=s.total_mass)
    (out/'parameters.json').write_text(json.dumps(parameters,indent=2),encoding='utf8')
    progress(100)
    a=np.array(rows)
    return {'kind':'nuclear','model':'spatial','scene':'transport','visualization':'droplet_transport_v1','frames':frames,
        'spatial':{'times':a[:,0].tolist(),'grid':N,'pm_y':PM,'ne_y':NE,'pore_centers':s.gates.tolist(),'zoom_x':float(cx),'backend':'numpy','time_unit':'model_units','elapsed_seconds':round(time.perf_counter()-started,2)},
        'table':{'headers':headers,'data':rows},'files':['nuclear_import.gif','nuclear_final.png','nuclear_spatial.csv','nuclear_field.csv','nuclear_fields.npz','droplet_states.json','parameters.json'],
        'plots':[{'title':'Spatial material balance','x':a[:,0].tolist(),'x_label':'Model time','series':[{'name':name,'y':a[:,j+1].tolist()} for j,name in enumerate(['Extracellular','Cytoplasm','In transit','Nucleus'])]}],
        'warnings':[],'summary':''}

