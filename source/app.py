import json
import os
import re
import secrets
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from models import PhaseParams, FluidParams, simulate_phase, simulate_fluid
from extensions import NuclearParams, simulate_nuclear
SPACES={}

ROOT=Path(__file__).resolve().parent
DATA=Path(os.getenv('DATA_DIR',str(ROOT/'runtime'))).resolve()
DATA.mkdir(parents=True,exist_ok=True)
LOCK=threading.RLock()
JOBS={}
POOLS={k:ThreadPoolExecutor(max_workers=n,thread_name_prefix=k) for k,n in [('phase',1),('nuclear',1),('fluidity',1)]}
MAX_JOBS=int(os.getenv('MAX_PENDING_JOBS','256'))
TTL=86400
HISTORY=DATA/'_history'
HISTORY.mkdir(exist_ok=True)

def write_status(path,data):
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False)
    tmp=path.with_suffix('.tmp')
    for attempt in range(6):
        try:
            tmp.write_text(payload,encoding='utf8');tmp.replace(path);return
        except PermissionError:
            time.sleep(.015*(attempt+1))
    # Windows readers/scanners can temporarily deny rename while allowing writes.
    # A progress update must not abort the numerical calculation for that reason.
    try:path.write_text(payload,encoding='utf8')
    except PermissionError:
        import logging
        logging.getLogger(__name__).warning('Status persistence delayed by a temporary file lock.')

def persist(job):
    write_status(DATA/job['id']/'status.json',job)
    meta={k:v for k,v in job.items() if k not in ['live']}
    write_status(HISTORY/(job['id']+'.json'),meta)

def cleanup():
    now=time.time()
    with LOCK:
        for key,job in list(JOBS.items()):
            if job['state'] not in ['queued','running'] and now-job['created']>TTL:
                path=(DATA/key).resolve()
                if path.parent==DATA: shutil.rmtree(path,ignore_errors=True)
                del JOBS[key]

@asynccontextmanager
async def lifespan(app):
    for p in DATA.glob('*/status.json'):
        try:
            job=json.loads(p.read_text(encoding='utf-8'))
            if job['id']!=p.parent.name: continue
            if job['state'] in ['queued','running']:
                job.update(state='failed',error='The computation session ended before completion. Run the task again.',error_code='interrupted')
                persist(job)
            JOBS[job['id']]=job
        except (ValueError,KeyError): pass
    cleanup()
    yield
    for pool in POOLS.values(): pool.shutdown(wait=False,cancel_futures=True)

app=FastAPI(title='CondensateLab',lifespan=lifespan)

@app.middleware('http')
async def limits(request,call_next):
    owner=request.cookies.get('condensatelab_workspace','')
    if not re.fullmatch('[a-f0-9]{48}',owner):owner=secrets.token_hex(24)
    request.state.owner=owner
    if request.url.path.startswith('/api/') and request.method=='POST':
        if os.getenv('APP_ACCESS_KEY') and not secrets.compare_digest(request.headers.get('X-Access-Key',''),os.environ['APP_ACCESS_KEY']):
            return JSONResponse({'detail':'Access key required.'},status_code=401)
        try: size=int(request.headers.get('content-length','0'))
        except ValueError: return JSONResponse({'detail':'Invalid content length.'},status_code=400)
        if size>12*1024*1024: return JSONResponse({'detail':'Request exceeds 12 MB.'},status_code=413)
    response=await call_next(request)
    response.set_cookie('condensatelab_workspace',owner,max_age=31536000,httponly=True,samesite='strict',secure=request.url.scheme=='https')
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    if request.url.path.startswith('/api/'): response.headers['Cache-Control']='no-store'
    return response

def create_job(kind,owner):
    cleanup()
    with LOCK:
        if sum(j['state'] in ['queued','running'] for j in JOBS.values())>=MAX_JOBS:
            raise HTTPException(429,'Queue is full. Please try later.')
        key=secrets.token_hex(24)
        (DATA/key).mkdir()
        job={'id':key,'kind':kind,'state':'queued','progress':0,'created':time.time(),'owner':owner}
        JOBS[key]=job;persist(job)
        return key

def classify_failure(exc):
    import errno
    name=type(exc).__name__.lower()
    message=str(exc).lower()
    if isinstance(exc, PermissionError): return 'permission'
    if isinstance(exc, (ImportError, ModuleNotFoundError)): return 'environment'
    if isinstance(exc, FileNotFoundError): return 'missing'
    if isinstance(exc, MemoryError): return 'resources'
    if isinstance(exc, OSError) and getattr(exc,'errno',None) in [errno.ENOSPC,errno.EDQUOT]: return 'storage'
    if any(word in name for word in ['timeout','connect','network','proxy']): return 'connection'
    if any(word in message for word in ['connection refused','connection reset','name resolution','timed out','502 bad gateway','503 service unavailable','space is sleeping']): return 'connection'
    if any(word in message for word in ['queue is full','too many requests','rate limit','429']): return 'busy'
    if any(word in message for word in ['unstable','stable step','time step','denominator','nan/inf','mass conservation','finite-value']): return 'numerical'
    if isinstance(exc, ValueError) and any(word in message for word in ['fasta','sequence','file format','header','cif','pdb']): return 'input'
    return 'calculation'


def execute(key,func,*args):
    def update(**changes):
        with LOCK:
            JOBS[key].update(changes);persist(JOBS[key])
    with LOCK:
        if JOBS[key]['state']=='cancelled':return
    update(state='running',started=time.time())
    started=time.perf_counter()
    try:
        def on_progress(value):
            if isinstance(value,dict):
                value=dict(value);pct=value.pop('percent')
                update(progress=pct,live=value,preview_updated=time.time())
            else: update(progress=value)
        if (DATA/key/'cancel.flag').exists():
            raise InterruptedError('Cancelled before execution.')
        result=func(*args,DATA/key,on_progress)
        # Ensure results are serializable, including upstream values.
        result=json.loads(json.dumps(result,ensure_ascii=False,allow_nan=False))
        (DATA/key/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        update(state='complete',progress=100,elapsed_seconds=time.perf_counter()-started)
    except InterruptedError as exc:
        update(state='cancelled',error=str(exc))
    except Exception as exc:
        # No traceback or server paths are returned to visitors.
        message=str(exc).replace(str(DATA),'[job storage]')[:800]
        update(state='failed',error=type(exc).__name__+': '+message,error_code=classify_failure(exc))

@app.get('/api/health')
def health():
    return {'ok':True,'upstreams':SPACES,'pending':sum(j['state'] in ['queued','running'] for j in JOBS.values()),'retention_hours':24}

@app.post('/api/phase',status_code=202)
def phase(p:PhaseParams,request:Request):
    key=create_job('phase',request.state.owner);POOLS['phase'].submit(execute,key,simulate_phase,p);return {'id':key}

@app.post('/api/fluidity',status_code=202)
def fluid(p:FluidParams,request:Request):
    key=create_job('fluidity',request.state.owner);POOLS['fluidity'].submit(execute,key,simulate_fluid,p);return {'id':key}

@app.post('/api/nuclear',status_code=202)
def nuclear(p:NuclearParams,request:Request):
    with LOCK:
        if any(j['kind']=='nuclear' and j['owner']==request.state.owner and j['state'] in ['queued','running'] for j in JOBS.values()):
            raise HTTPException(429,'A simulation is already active. View or stop it in Job history before starting another.')
        key=create_job('nuclear',request.state.owner)
    try:POOLS['nuclear'].submit(execute,key,simulate_nuclear,p)
    except Exception:
        with LOCK:
            JOBS[key].update(state='failed',error='The computation service could not accept this task. Restart the software and retry.',error_code='resources');persist(JOBS[key])
        raise HTTPException(503,'The computation service is unavailable. Restart the software and retry.')
    return {'id':key}




def get_job(key):
    cleanup()
    with LOCK:
        if key not in JOBS: raise HTTPException(404,'Job not found or expired.')
        return dict(JOBS[key])

@app.get('/api/history')
def history(request:Request,software:str='all',q:str='',state:str='all'):
    cleanup()
    records=[]
    for path in HISTORY.glob('*.json'):
        try:
            j=json.loads(path.read_text(encoding='utf8'))
            if j.get('owner')!=request.state.owner:continue
            group='simulation' if j['kind'] in ['phase','fluidity','nuclear'] else 'protein'
            if software!='all' and group!=software:continue
            j.pop('owner',None)
            j['available']=j['id'] in JOBS
            records.append(j)
        except (ValueError,OSError):continue
    counts={k:sum(j['state']==k for j in records) for k in ['complete','failed','cancelled','queued','running']}
    filtered=[j for j in records if (state=='all' or j['state']==state) and q.lower() in (j['id']+' '+j['kind']).lower()]
    return {'counts':counts,'total':len(records),'jobs':sorted(filtered,key=lambda j:j['created'],reverse=True)}

@app.get('/api/jobs/{key}')
def status(key:str):
    job=get_job(key)
    job.pop('owner',None)
    if job['state']=='queued':
        job['queue_ahead']=sum(j['kind']==job['kind'] and j['created']<job['created'] and j['state'] in ['queued','running'] for j in JOBS.values())
    return job

@app.get('/api/jobs/{key}/result')
def result(key:str):
    if get_job(key)['state']!='complete': raise HTTPException(409,'Result is not complete.')
    return FileResponse(DATA/key/'result.json',media_type='application/json')

@app.get('/api/jobs/{key}/preview')
def preview(key:str):
    job=get_job(key)
    path=DATA/key/job.get('live',{}).get('preview_file','preview-not-ready.png')
    if not path.exists(): raise HTTPException(404,'The first preview is not ready yet.')
    return FileResponse(path,media_type='image/png',headers={'Cache-Control':'no-store'})

@app.post('/api/jobs/{key}/cancel')
def cancel(key:str):
    job=get_job(key)
    if job['kind'] not in ['phase','fluidity','nuclear']: raise HTTPException(400,'Only the local phase simulation can be cancelled here.')
    if job['state'] in ['queued','running']:
        (DATA/key/'cancel.flag').write_text('cancel',encoding='utf-8')
        with LOCK:
            if JOBS[key]['state']=='queued':
                JOBS[key].update(state='cancelled',error='Cancelled while waiting.');persist(JOBS[key])
    return {'requested':True}

@app.get('/api/jobs/{key}/files/{filename}')
def download(key:str,filename:str):
    if get_job(key)['state']!='complete': raise HTTPException(409,'Result is not complete.')
    data=json.loads((DATA/key/'result.json').read_text(encoding='utf-8'))
    allowed=set(data.get('files',[]))|{x['file'] for x in data.get('frames',[])}|{'result.json'}
    if filename not in allowed: raise HTTPException(404,'File not found.')
    return FileResponse(DATA/key/filename,filename=filename)

@app.get('/api/jobs/{key}/frames')
def live_frames(key:str,after:int=0):
    get_job(key)
    path=DATA/key/'frames.jsonl'
    rows=[]
    if path.exists():
        for line in path.read_text(encoding='utf8').splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    return {'frames':rows[max(0,after):max(0,after)+100],'total':len(rows)}

@app.get('/api/jobs/{key}/frame/{filename}')
def live_frame(key:str,filename:str):
    get_job(key)
    import re
    if not re.fullmatch(r'frame_[0-9]+\.png',filename): raise HTTPException(404)
    path=DATA/key/filename
    if not path.is_file(): raise HTTPException(404)
    return FileResponse(path,media_type='image/png')

app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
