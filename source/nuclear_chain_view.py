"""Full-chain field, synchronized nuclear-pore crop and same-run material curves."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from scipy import ndimage

PALETTE=(colormaps['RdBu_r'](np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
SERIES=['Extracellular','Cytoplasm','In transit','Nucleus']
COLORS=['#929aa6','#a58ae0','#329460','#703fbd']


def font(size):
    for name in ['DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default(size=size)


def field_image(field):
    values=np.clip(((field+1.2)/3.6*255).round().astype(int),0,255)
    return Image.fromarray(PALETTE[values])


def render_chain(sim,field,t,rows,end_time=None,zoom_x=None):
    physical=getattr(sim,'length_per_cell',9.4) is not None
    n=sim.n_phys;end_time=end_time or max(t,1)
    im=Image.new('RGB',(1000,820),'#fbfcfd');d=ImageDraw.Draw(im)
    title=font(19);normal=font(15);small=font(12)
    d.text((25,17),'CELL ENTRY → CYTOPLASMIC TRANSPORT → NUCLEAR IMPORT',font=title,fill='#263d4c')
    d.text((25,47),f't = {t:g} model units',font=normal,fill='#5c7280')
    mx,my,ms=25,181,520
    im.paste(field_image(field[::-1]).resize((ms,ms),Image.Resampling.BILINEAR),(mx,my))
    d=ImageDraw.Draw(im)
    def world(x,y):return mx+x*ms/n,my+(n-1-y)*ms/n
    def membrane(y,centers,rect=None,bounds=None):
        if rect is None:
            x0,y0=world(0,y);x1,_=world(n,y)
            d.line((x0,y0,x1,y0),fill='#183447',width=2)
            for c in centers:
                if 0<=c<n:
                    a,_=world(c-4,y);b,_=world(c+4,y);d.line((a,y0,b,y0),fill='#3fc37c',width=4)
        else:
            zx,zy,zs=rect;xlo,xhi,ylo,yhi=bounds
            py=zy+(yhi-1-y)*zs/(yhi-ylo)
            if zy<=py<=zy+zs:
                d.line((zx,py,zx+zs,py),fill='#183447',width=2)
                for c in centers:
                    a=zx+(max(c-4,xlo)-xlo)*zs/(xhi-xlo);b=zx+(min(c+4,xhi)-xlo)*zs/(xhi-xlo)
                    if b>a:d.line((a,py,b,py),fill='#3fc37c',width=5)
    membrane(sim.pm_y,sim.pm_centers);membrane(sim.ne_y,sim.ne_centers)
    def label(y,text,color='#e1e7ff'):
        _,py=world(0,y);d.text((mx+10,py),text,font=normal,fill=color,stroke_width=2,stroke_fill='#163d5b')
    label(n-10,'Nucleus')
    label(sim.ne_y+9,'Nuclear envelope · green = pores','#8be0b1')
    label((sim.pm_y+sim.ne_y)/2,'Cytoplasm')
    label(sim.pm_y+9,'Cell membrane','#8be0b1')
    label(max(8,sim.pm_y/2),'Extracellular fluid')
    # The ROI is a crop of exactly the same field, with no added trajectories.
    centers=[int(c) for c in sim.ne_centers if 0<=c<n]
    cx=zoom_x if zoom_x is not None else min(centers,key=lambda c:abs(c-n/2)) if centers else n//2
    half=max(12,min(n//6,48));xlo=max(0,cx-half);xhi=min(n,cx+half);ylo=max(0,sim.ne_y-half);yhi=min(n,sim.ne_y+half)
    x0,y0=world(xlo,yhi-1);x1,y1=world(xhi,ylo);d.rectangle((x0,y0,x1,y1),outline='#cf4f4b',width=2)
    zx,zy,zs=600,80,365
    im.paste(field_image(field[ylo:yhi,xlo:xhi][::-1]).resize((zs,zs),Image.Resampling.BILINEAR),(zx,zy));d=ImageDraw.Draw(im)
    membrane(sim.ne_y,sim.ne_centers,(zx,zy,zs),(xlo,xhi,ylo,yhi))
    d.rectangle((zx,zy,zx+zs,zy+zs),outline='#a8b7c0')
    d.text((zx,54),('Nuclear-pore close-up · '+(f'{(xhi-xlo)*9.4/1000:.2f} µm' if physical else f'{xhi-xlo} a.u.')),font=normal,fill='#263d4c')
    d.text((zx+10,zy+10),'Nucleus',font=small,fill='#edf2ff',stroke_width=1,stroke_fill='#17394f')
    d.text((zx+10,zy+zs-25),'Cytoplasm',font=small,fill='#edf2ff',stroke_width=1,stroke_fill='#17394f')
    # Physical scale bars use the supplied model's 9.4 nm lattice spacing.
    for x,y,pixels_per_cell,available in [(mx+ms-135,my+ms-22,ms/n,115),(zx+18,zy+zs-44,zs/(xhi-xlo),130)]:
        nm=(500 if available/pixels_per_cell*9.4>=500 else 100) if physical else (50 if available/pixels_per_cell>=50 else 20)
        length=(nm/9.4 if physical else nm)*pixels_per_cell
        d.line((x,y,x+length,y),fill='#f8fafc',width=4);d.text((x,y-19),f'{nm} '+('nm' if physical else 'a.u.'),font=small,fill='#f8fafc',stroke_width=1,stroke_fill='#234356')
    # Curves retain their full axes during playback and have a moving time cursor.
    a=np.array(rows);values=a[:,1:5]*100;lo=min(0,float(values.min()));hi=max(5,float(values.max()));pad=max(2,(hi-lo)*.05);lo-=pad;hi+=pad
    px,py,pw,ph=630,500,335,225
    for j in range(5):
        yy=py+ph-ph*j/4;d.line((px,yy,px+pw,yy),fill='#e1e8ee');d.text((px-45,yy-7),f'{lo+(hi-lo)*j/4:.0f}',font=small,fill='#607583')
        xx=px+pw*j/4;d.text((xx-12,py+ph+10),f'{end_time*j/4:g}',font=small,fill='#607583')
    for j,(name,color) in enumerate(zip(SERIES,COLORS)):
        pts=[(px+pw*x/end_time,py+ph-ph*(v-lo)/(hi-lo)) for x,v in zip(a[:,0],values[:,j])]
        if len(pts)>1:d.line(pts,fill=color,width=2)
        value=float(np.interp(t,a[:,0],values[:,j]));xx=px+pw*t/end_time;yy=py+ph-ph*(value-lo)/(hi-lo)
        d.ellipse((xx-3,yy-3,xx+3,yy+3),fill=color)
        d.text((615+(j%2)*180,465+(j//2)*17),name,font=small,fill=color)
    xx=px+pw*t/end_time;d.line((xx,py,xx,py+ph),fill='#8b98a2',width=1)
    d.text((665,763),'Model time',font=small,fill='#607583')
    d.text((600,783),'Material / initial total (%)',font=small,fill='#607583')
    # Fixed shared colour scale.
    bar=np.tile(np.linspace(-1.2,2.4,ms),(13,1));im.paste(field_image(bar),(mx,724));d=ImageDraw.Draw(im)
    for value in [-1,0,1,2]:d.text((mx+(value+1.2)/3.6*ms-6,742),f'{value:g}',font=small,fill='#607583')
    d.text((mx,765),'Concentration order parameter · fixed colour scale',font=small,fill='#607583')
    # Actual dense connected components, measured from this same frame.
    counts=[]
    for lo_y,hi_y in [(sim.pm_half+1,sim.pm_y-sim.ne_half),(sim.pm_y+sim.ne_half+1,sim.ne_y-sim.ne_half),(sim.ne_y+sim.ne_half+1,n)]:
        counts.append(ndimage.label(field[lo_y:hi_y]>.5)[1])
    d.text((25,792),f'Droplets: extracellular {counts[0]} · cytoplasm {counts[1]} · nucleus {counts[2]}',font=small,fill='#425e6d')
    return im
