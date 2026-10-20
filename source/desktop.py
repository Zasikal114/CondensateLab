"""Desktop host for a single standalone research application."""
import os,sys,json,socket,threading,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PRODUCT=json.loads((ROOT/'product.json').read_text(encoding='utf8'))['name']
STORE=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/PRODUCT
STORE.mkdir(parents=True,exist_ok=True)
os.environ['DATA_DIR']=str(STORE/'jobs')
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(STORE/'matplotlib')

def main():
    blocked_connections=[]
    if '--offline-check' in sys.argv:
        def offline_guard(event,args):
            if event=='socket.connect':
                address=args[1]
                if isinstance(address,tuple) and address[0] not in ('127.0.0.1','::1','localhost'):
                    blocked_connections.append(str(address))
                    raise OSError('Offline verification: external connections are disabled')
        sys.addaudithook(offline_guard)
    import uvicorn
    import app
    sock=socket.socket()
    sock.bind(('127.0.0.1',0));sock.listen(128);sock.setblocking(False)
    url='http://127.0.0.1:'+str(sock.getsockname()[1])
    server=uvicorn.Server(uvicorn.Config(app.app,log_config=None,access_log=False))
    worker=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True)
    worker.start()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPCookieProcessor())
    for _ in range(60):
        try:
            with opener.open(url+'/api/health',timeout=1) as response:
                health=json.load(response)
            if health.get('ok'):break
        except Exception:time.sleep(.25)
    else:raise RuntimeError('The local computation service did not start.')
    if '--self-test' in sys.argv:
        output=Path(sys.argv[sys.argv.index('--self-test')+1])
        with opener.open(url+'/',timeout=3) as response:page=response.read().decode('utf8')
        report={'product':PRODUCT,'health':health,'page_ok':PRODUCT in page}
        if '--self-test-models' in sys.argv:
            report['models']={}
            cases={'phase':{'N':32,'steps':80},'fluidity':{'N':32,'on_time':5,'off_time':5,'cycles':1},'nuclear':{'transport_duration':30}}
            for kind,parameters in cases.items():
                request=urllib.request.Request(url+'/api/'+kind,data=json.dumps(parameters).encode(),headers={'Content-Type':'application/json'},method='POST')
                with opener.open(request,timeout=10) as response:job=json.load(response)['id']
                for _ in range(480):
                    with opener.open(url+'/api/jobs/'+job,timeout=5) as response:state=json.load(response)
                    if state['state'] in ['complete','failed','cancelled']:break
                    time.sleep(.25)
                if state['state']!='complete':raise RuntimeError('Packaged '+kind+' check failed: '+str(state))
                with opener.open(url+'/api/jobs/'+job+'/result',timeout=5) as response:result=json.load(response)
                gif=next(f for f in result['files'] if f.endswith('.gif'))
                with opener.open(url+'/api/jobs/'+job+'/files/'+gif,timeout=5) as response:
                    from PIL import Image
                    from io import BytesIO
                    animation=Image.open(BytesIO(response.read()))
                    frames=animation.n_frames
                    if frames<2:raise RuntimeError('Animation contains fewer than two frames')
                report['models'][kind]={'state':state['state'],'gif_frames':frames,'elapsed_seconds':state['elapsed_seconds']}
            report['offline_guard_enabled']='--offline-check' in sys.argv
            report['external_connection_attempts']=blocked_connections
        output.write_text(json.dumps(report),encoding='utf8')
        server.should_exit=True;worker.join(timeout=5)
        return
    import webview
    webview.create_window(PRODUCT,url,width=1380,height=900,min_size=(900,650))
    try:webview.start(gui='edgechromium',private_mode=False,storage_path=str(STORE/'browser'))
    finally:
        server.should_exit=True
        worker.join(timeout=3)
        os._exit(0)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        error=traceback.format_exc()
        (STORE/'startup-error.log').write_text(error,encoding='utf8')
        if '--self-test' not in sys.argv:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0,'本地计算服务未能启动。请确认解压后的文件完整、用户目录可写，并已安装 Microsoft Edge WebView2 Runtime。详细记录：\n'+str(STORE/'startup-error.log'),PRODUCT,16)
        sys.exit(1)
