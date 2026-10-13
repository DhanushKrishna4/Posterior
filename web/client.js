export class EngineClient extends EventTarget{
  constructor(){super();this.generation=0;this.serial=0;this.pending=new Map();this.checkpoint=null;this.result=null;this.ready=false;this.worker=null;}
  emit(type,detail){this.dispatchEvent(new CustomEvent(type,{detail}));}
  async boot(){
    this.worker?.terminate();this.ready=false;
    for(const {reject}of this.pending.values())reject(new Error('Worker restarted; unfinished work discarded.'));this.pending.clear();
    const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});this.worker=worker;
    worker.onmessage=({data})=>{
      if(worker!==this.worker||data.generation!==this.generation)return;
      if(data.type==='response'){
        const pending=this.pending.get(data.id);if(!pending)return;this.pending.delete(data.id);
        data.error?pending.reject(new Error(data.error)):pending.resolve(data.result);
      }else{
        if(data.type==='ready')this.ready=true;
        if(data.type==='delta'){
          data.result=this.mergeDelta(data.result);data.type='snapshot';
        }
        if(data.type==='snapshot'&&data.result?.checkpoint){this.result=data.result;this.checkpoint=data.result.checkpoint;}
        this.emit(data.type,data);
        if(data.type==='snapshot')worker.postMessage({op:'ack',generation:this.generation});
      }
    };
    worker.onerror=event=>{const message=event.message||'Worker failed. Use Recover worker to restore the last completed transition.';for(const p of this.pending.values())p.reject(new Error(message));this.pending.clear();this.emit('error',{message});};
    return this.request('init',{base:new URL(import.meta.env.BASE_URL,location.origin).href});
  }
  mergeDelta(delta){
    const r=this.result;if(!r||r.run_id!==delta.run_id)throw new Error('Incremental result has no matching base checkpoint.');
    const cp=r.checkpoint,i=delta.chain;
    const tail=(array,offset,values)=>{array.splice(offset,array.length-offset,...values);};
    tail(r.samples[i],delta.offset,delta.samples);tail(r.unconstrained_samples[i],delta.offset,delta.unconstrained_samples);
    tail(r.transition_summaries[i],delta.summary_offset,delta.transition_summaries);tail(r.transitions,delta.transition_offset,delta.transitions);
    if(delta.chain_state){cp.chains[i]=delta.chain_state;r.adaptation[i]=delta.chain_state.adapter;}
    if(delta.vi_state){
      const v=r.vi[i],oldHistory=v.history,oldEvaluations=v.evaluation_history;
      tail(oldHistory,delta.history_offset,delta.vi_state.history);tail(oldEvaluations,delta.evaluation_offset,delta.vi_state.evaluation_history);
      r.vi[i]={...delta.vi_state,history:oldHistory,evaluation_history:oldEvaluations};cp.vi[i]=r.vi[i];
    }
    cp.samples=r.samples;cp.unconstrained_samples=r.unconstrained_samples;cp.transition_summaries=r.transition_summaries;cp.transitions=r.transitions;
    cp.next_chain=delta.next_chain;cp.elapsed=delta.stats.elapsed_seconds;cp.evaluations=delta.stats.gradient_evaluations;
    cp.stats=Object.fromEntries(Object.keys(cp.stats).map(k=>[k,delta.stats[k]]));cp.status=delta.status;cp.reason=delta.reason;
    return {...r,status:delta.status,reason:delta.reason,progress:delta.progress,stats:delta.stats,checkpoint:cp};
  }
  request(op,extra={}){
    if(this.pending.size>=16)return Promise.reject(new Error('The bounded request queue is full. Wait for the current action.'));
    const id=++this.serial;
    return new Promise((resolve,reject)=>{this.pending.set(id,{resolve,reject});this.worker.postMessage({id,op,generation:this.generation,...extra});});
  }
  newGeneration(){
    this.generation++;
    for(const{reject}of this.pending.values())reject(new Error('Configuration changed; stale response discarded.'));this.pending.clear();
  }
  async recover(){
    const checkpoint=this.checkpoint;if(!checkpoint)throw new Error('There is no completed transition checkpoint to restore.');
    this.newGeneration();await this.boot();return this.request('restore',{checkpoint});
  }
}
