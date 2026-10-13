// One persistent Python engine. The event loop yields only between committed transitions.
let pyodide, bridge, generation=0, running=false, waitingForAck=false, tickTimer, readyPromise, steps=0;
const send=(type,data={})=>self.postMessage({type,generation,...data});
async function ready(base){
  if(readyPromise)return readyPromise;
  readyPromise=(async()=>{
    send('loading',{message:'Loading the local Python runtime…'});
    const runtime=new URL('runtime/',base).href;
    const {loadPyodide}=await import(/* @vite-ignore */new URL('pyodide.mjs',runtime).href);
    pyodide=await loadPyodide({indexURL:runtime});
    send('loading',{message:'Loading NumPy and SciPy…'});
    await pyodide.loadPackage(['numpy','scipy']);
    const response=await fetch(new URL('engine/posterior_lab.zip',base),{cache:'no-cache'});
    if(!response.ok)throw new Error(`Python engine download failed (HTTP ${response.status}).`);
    pyodide.unpackArchive(await response.arrayBuffer(),'zip',{extractDir:'/home/pyodide'});
    pyodide.runPython('import posterior_lab.bridge as _posterior_bridge');
    bridge=pyodide.runPython('_posterior_bridge.call');
    send('ready',{versions:{pyodide:pyodide.version,python:pyodide.runPython('import sys; sys.version.split()[0]')}});
  })();
  return readyPromise;
}
function call(request){
  const response=JSON.parse(bridge(JSON.stringify(request)));
  if(response.error)throw new Error(typeof response.error==='string'?response.error:JSON.stringify(response.error));
  return response;
}
function snapshot(diagnostics=false){return call({op:'snapshot',diagnostics});}
function schedule(){clearTimeout(tickTimer);tickTimer=setTimeout(tick,0);}
function tick(){
  if(!running)return;
  try{
    const state=call({op:'step',count:1,full:false}); steps++;
    // Each completed transition is checkpointed outside the worker. Diagnostic work
    // is periodic; a worker kill can only lose the transition currently executing.
    const full=steps%25===0||state.status!=='running';
    const result=full?snapshot(true):call({op:'delta'});
    send(full?'snapshot':'delta',{result});
    if(['completed','complete','cancelled','failed','budget_exhausted','paused','stopped'].includes(result.status)){running=false;send('stopped',{status:result.status});}
    else waitingForAck=true;
  }catch(error){running=false;send('error',{message:error.message});}
}
self.onmessage=async({data})=>{
  const {id,op}=data;
  try{
    if(op==='ack'){
      if(data.generation===generation&&running&&waitingForAck){waitingForAck=false;schedule();}
      return;
    }
    if(op==='init'){generation=data.generation;await ready(data.base);send('response',{id,result:{ready:true}});return;}
    if(data.generation<generation)return;
    if(data.generation>generation){running=false;clearTimeout(tickTimer);generation=data.generation;}
    if(!bridge)throw new Error('Python is still loading.');
    let result;
    if(op==='start'){
      running=false;waitingForAck=false;clearTimeout(tickTimer);steps=0;
      result=call({...data.request,op:'start'});send('snapshot',{result});running=true;schedule();
    }else if(op==='pause'||op==='cancel'){
      running=false;clearTimeout(tickTimer);
      call({op:'status',status:op==='pause'?'paused':'cancelled',reason:op==='pause'?'Paused between completed transitions.':'Cancelled between completed transitions.'});
      result=snapshot(true);send('snapshot',{result});
    }else if(op==='resume'){
      call({op:'status',status:'running'});running=true;schedule();result={status:'running'};
    }else if(op==='restore'){
      running=false;clearTimeout(tickTimer);result=call({op:'restore',checkpoint:data.checkpoint});if(result.status==='running')result=call({op:'status',status:'paused',reason:'Restored completed checkpoint; unfinished work discarded.'});send('snapshot',{result});
    }else if(op==='reset'){
      running=false;clearTimeout(tickTimer);result={status:'idle'};
    }else result=call(data.request??{op});
    send('response',{id,result});
  }catch(error){send('response',{id,error:error.message});if(op!=='init')send('error',{message:error.message});}
};
