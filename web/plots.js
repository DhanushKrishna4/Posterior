export const COLORS=['#414c99','#c08a3c','#4e897e','#b66b79'];
const ink='#34404b', muted='#8a9396', grid='#e8e6de';
export const finite=x=>typeof x==='number'&&Number.isFinite(x);
const extent=values=>{let lo=Infinity,hi=-Infinity;for(const v of values)if(finite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}if(!finite(lo))return[-3,3];if(lo===hi)return[lo-1,hi+1];const pad=(hi-lo)*.08;return[lo-pad,hi+pad];};
const tick=v=>Math.abs(v)>1000?v.toExponential(1):Math.abs(v)<.01&&v!==0?v.toExponential(1):Number(v.toFixed(2)).toString();
function prepare(canvas,xDomain,yDomain,labels={}){
 const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),w=rect.width,h=rect.height;
 canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);
 const c=canvas.getContext('2d');c.scale(ratio,ratio);c.fillStyle='#fcfaf6';c.fillRect(0,0,w,h);
 const box={x:49,y:23,w:Math.max(1,w-72),h:Math.max(1,h-67)};
 const X=x=>box.x+(x-xDomain[0])/(xDomain[1]-xDomain[0])*box.w,Y=y=>box.y+box.h-(y-yDomain[0])/(yDomain[1]-yDomain[0])*box.h;
 c.font='9px -apple-system, sans-serif';c.lineWidth=1;
 for(let i=0;i<=4;i++){const x=box.x+box.w*i/4,y=box.y+box.h*i/4;c.strokeStyle=grid;c.beginPath();c.moveTo(x,box.y);c.lineTo(x,box.y+box.h);c.moveTo(box.x,y);c.lineTo(box.x+box.w,y);c.stroke();c.fillStyle=muted;c.textAlign='center';c.fillText(tick(xDomain[0]+(xDomain[1]-xDomain[0])*i/4),x,box.y+box.h+16);c.textAlign='right';c.fillText(tick(yDomain[1]-(yDomain[1]-yDomain[0])*i/4),box.x-8,y+3);}
 c.fillStyle='#6b757e';c.textAlign='center';c.font='10px -apple-system, sans-serif';c.fillText(labels.x||'',box.x+box.w/2,h-7);c.save();c.translate(12,box.y+box.h/2);c.rotate(-Math.PI/2);c.fillText(labels.y||'',0,0);c.restore();
 return {c,box,X,Y,w,h};
}
function line(p,x,y,color,width=1.7,dash=[]){const{c,X,Y}=p;c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.beginPath();let started=false;for(let i=0;i<x.length;i++){if(!finite(x[i])||!finite(y[i])){started=false;continue;}if(!started)c.moveTo(X(x[i]),Y(y[i]));else c.lineTo(X(x[i]),Y(y[i]));started=true;}c.stroke();c.setLineDash([]);}
function area(p,x,low,high,color){const{c,X,Y}=p;if(!x?.length)return;c.fillStyle=color;c.beginPath();x.forEach((v,i)=>i?c.lineTo(X(v),Y(low[i])):c.moveTo(X(v),Y(low[i])));for(let i=x.length-1;i>=0;i--)c.lineTo(X(x[i]),Y(high[i]));c.closePath();c.fill();}
function points(p,rows,color,alpha=.5,r=2,indices=[0,1],hollow=false){const{c,X,Y}=p;c.fillStyle=color;c.strokeStyle=color;c.lineWidth=1;c.globalAlpha=alpha;for(const row of rows){const a=row[indices[0]],b=row[indices[1]];if(!finite(a)||!finite(b))continue;c.beginPath();c.arc(X(a),Y(b),r,0,2*Math.PI);if(hollow)c.stroke();else c.fill();}c.globalAlpha=1;}
function note(p,text,sub=''){const{c,box}=p;c.fillStyle='#6c767f';c.textAlign='center';c.font='12px -apple-system, sans-serif';c.fillText(text,box.x+box.w/2,box.y+box.h/2-5);if(sub){c.font='10px -apple-system, sans-serif';c.fillStyle='#919995';c.fillText(sub,box.x+box.w/2,box.y+box.h/2+16);}}
function contour(p,ref){const g=ref?.grid;if(!g?.x?.length||!g?.y?.length||!g?.z?.length)return;
 const {x,y,z}=g,max=Math.max(...z.flat().filter(finite)),levels=[.05,.15,.35,.60,.85].map(v=>v*max),{c,X,Y}=p;
 // Equal-density contours, deliberately not labelled probability-mass regions.
 for(const level of levels){c.strokeStyle='#5b65796a';c.lineWidth=1;c.beginPath();for(let j=0;j<y.length-1;j++)for(let i=0;i<x.length-1;i++){
  const corners=[[x[i],y[j],z[j]?.[i]],[x[i+1],y[j],z[j]?.[i+1]],[x[i+1],y[j+1],z[j+1]?.[i+1]],[x[i],y[j+1],z[j+1]?.[i]]],cross=[];
  for(let edge=0;edge<4;edge++){const a=corners[edge],b=corners[(edge+1)%4];if((a[2]<level)!==(b[2]<level)){const t=(level-a[2])/(b[2]-a[2]);cross.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);}}
  for(let k=0;k+1<cross.length;k+=2){c.moveTo(X(cross[k][0]),Y(cross[k][1]));c.lineTo(X(cross[k+1][0]),Y(cross[k+1][1]));}
 }c.stroke();}
}
export function mainPlot(canvas,{preset,reference:ref={},run,view='density',parameter=0,transitionIndex=-1,comparison=null}){
 const chains=run?.samples??[],all=chains.flat(),names=run?.names??[];
 if(view==='trajectory'){
  const transitions=run?.transitions??[],t=transitions[transitionIndex<0?transitions.length-1:Math.min(transitionIndex,transitions.length-1)];
  const raw=t?.trajectory??[],limit=Number.isFinite(arguments[1].replayCount)?arguments[1].replayCount:raw.length,visible=raw.slice(0,limit),rows=raw.map(v=>v.position??[]),dims=rows[0]?.length??2;
  const coordinate=(point,index)=>[dims===1?index:point[0],dims===1?point[0]:point[1]];
  const xy=rows.map(coordinate),p=prepare(canvas,extent(xy.map(v=>v[0])),extent(xy.map(v=>v[1])),{x:dims===1?'Recorded integration index':'Working coordinate x₀',y:dims===1?'Working coordinate x₀':'Working coordinate x₁'});
  if(!rows.length){note(p,'No recorded transition yet','Run HMC or NUTS to inspect real integration steps.');return;}
  const start=t.starting_position??rows[0];
  if(t.algorithm==='hmc')line(p,xy.slice(0,limit).map(v=>v[0]),xy.slice(0,limit).map(v=>v[1]),COLORS[0],1.2,[3,3]);
  else{
   // Every expansion starts at its current forward/backward edge. Never join
   // traversal-order leaves across a switch to the other side of a NUTS tree.
   let minus={position:start,index:0},plus={position:start,index:0};
   for(const branch of (t.branches??[]).filter(v=>v.kind==='expansion')){
    const edge=branch.direction===-1?minus:plus;
    const members=raw.map((v,index)=>({...v,index})).filter(v=>v.integration_step>=branch.first_integration_step&&v.integration_step<=branch.last_integration_step);
    const shown=members.filter(v=>v.index<limit),segment=[coordinate(edge.position,edge.index),...shown.map(v=>coordinate(v.position,v.index))];
    if(shown.length)line(p,segment.map(v=>v[0]),segment.map(v=>v[1]),branch.direction===-1?COLORS[0]:COLORS[1],1.3,[3,3]);
    if(members.length){const last=members.at(-1);if(branch.direction===-1)minus=last;else plus=last;}
   }
  }
  for(let i=0;i<visible.length;i++){const q=coordinate(visible[i].position,i),{c,X,Y}=p;c.beginPath();c.arc(X(q[0]),Y(q[1]),2.7,0,Math.PI*2);c.strokeStyle=visible[i].continuing===false?COLORS[3]:COLORS[0];c.fillStyle=visible[i].eligible===false?'#fcfaf6':COLORS[0];c.fill();c.stroke();}
  points(p,[coordinate(start,0)],COLORS[2],1,4.5);
  if(limit>=raw.length&&t.selected_position){const selectedIndex=rows.findIndex(v=>v.every((x,i)=>Math.abs(x-t.selected_position[i])<1e-10));const q=coordinate(t.selected_position,Math.max(0,selectedIndex)),{c,X,Y}=p;c.fillStyle=COLORS[3];c.beginPath();c.moveTo(X(q[0]),Y(q[1])-6);c.lineTo(X(q[0])+6,Y(q[1]));c.lineTo(X(q[0]),Y(q[1])+6);c.lineTo(X(q[0])-6,Y(q[1]));c.closePath();c.fill();}
  return;
 }
 if(preset==='custom'&&(run?.names?.length??1)===1){
  const previous=comparison?.samples??[],values=all.map(r=>r[0]),domain=extent([...values,...previous.flat().map(r=>r[0])]),bins=30,width=(domain[1]-domain[0])/bins;
  const histogram=rows=>{const hist=Array(bins).fill(0);for(const r of rows)hist[Math.max(0,Math.min(bins-1,Math.floor((r[0]-domain[0])/width)))]++;return hist.map(v=>rows.length?v/rows.length/width:0);};
  const counts=chains.map(histogram),priorCounts=previous.map(histogram),p=prepare(canvas,domain,[0,Math.max(1,...counts.flat(),...priorCounts.flat())*1.1],{x:`${names[0]??'Parameter'} · model coordinates`,y:'Sample-based marginal density'});
  priorCounts.forEach((ys,i)=>{if(previous[i].length)line(p,ys.map((_,j)=>domain[0]+(j+.5)*width),ys,COLORS[i],1.5,[4,3]);});counts.forEach((ys,i)=>{if(chains[i].length)line(p,ys.map((_,j)=>domain[0]+(j+.5)*width),ys,COLORS[i],1.5);});if(!values.length&&!previous.some(rows=>rows.length))note(p,'No retained draws yet','This custom model has no independent analytic reference.');return;
 }
 if(preset==='beta'){
  const x=ref.x??[],ys=[ref.prior??[],ref.posterior??[],ref.likelihood??[]],bins=30;
  // Each chain or variational fit gets its own normalized histogram. In
  // particular, independently fitted VI approximations are never pooled.
  const histogram=rows=>{const counts=Array(bins).fill(0);for(const row of rows){const v=row[0];if(v>=0&&v<=1)counts[Math.min(bins-1,Math.floor(v*bins))]++;}return rows.length?counts.map(v=>v/rows.length*bins):counts;};
  const histograms=chains.map(histogram),previous=comparison?.samples??[],comparisonHistograms=previous.map(histogram);
  const max=Math.max(1,...ys.flat().filter(finite),...histograms.flat(),...comparisonHistograms.flat());const p=prepare(canvas,[0,1],[0,max*1.1],{x:'Success probability θ · model coordinates',y:'Probability density'});
  if(ref.prior)line(p,x,ref.prior,'#93a28f',1.4,[5,4]);if(ref.likelihood)line(p,x,ref.likelihood,'#b79974',1.3,[2,3]);if(ref.posterior){area(p,x,x.map(()=>0),ref.posterior,'#414c9910');line(p,x,ref.posterior,ink,2);}
  comparisonHistograms.forEach((ys,i)=>{if(previous[i].length)line(p,ys.map((_,j)=>(j+.5)/bins),ys,COLORS[i],1.5,[4,3]);});
  histograms.forEach((ys,i)=>{if(chains[i]?.length)line(p,ys.map((_,j)=>(j+.5)/bins),ys,COLORS[i],1.35);});
  if(!x.length&&!all.length)note(p,'The belief starts here','Python will compute the exact posterior when the runtime is ready.');return;
 }
 if(preset==='regression'){
  const x=ref.x_grid??[],xd=ref.x_data??[],yd=ref.y_data??[];const p=prepare(canvas,extent([...x,...xd]),extent([...yd,...(ref.predictive_lower??[]),...(ref.predictive_upper??[])]),{x:'Predictor x',y:'Response y'});
  if(x.length){area(p,x,ref.predictive_lower,ref.predictive_upper,'#bec4df65');area(p,x,ref.mean_lower,ref.mean_upper,'#717fc554');line(p,x,ref.mean_response,ink,1.8);}
  const regressionLines=(groups,budget,previous)=>{const perGroup=Math.max(1,Math.floor(budget/Math.max(1,groups.length)));groups.forEach((rows,i)=>rows.filter((_,j)=>j%Math.max(1,Math.ceil(rows.length/perGroup))===0).slice(0,perGroup).forEach(row=>line(p,x,x.map(v=>row[0]+row[1]*v),COLORS[i]+(previous?'65':'50'),1,previous?[3,3]:[])));};
  regressionLines(comparison?.samples??[],12,true);regressionLines(chains,16,false);points(p,xd.map((v,i)=>[v,yd[i]]),ink,1,3);
  if(!x.length)note(p,'Prediction keeps two uncertainties distinct','Load the local runtime to compute the analytic bands.');return;
 }
 const rx=ref.grid?.x,ry=ref.grid?.y,refs=ref.reference_samples??[];
 let xlim=rx?[rx[0],rx.at(-1)]:extent([...all,...refs.slice(0,500)].map(v=>v[0]));
 let ylim=ry?[ry[0],ry.at(-1)]:extent([...all,...refs.slice(0,500)].map(v=>v[1]));
 if(preset==='funnel'){xlim=[-9,9];ylim=[-15,15];}
 const p=prepare(canvas,xlim,ylim,{x:preset==='funnel'?'Hierarchy scale y · model coordinates':`${names[0]??'θ₀'} · model coordinates`,y:preset==='funnel'?'Child x₀ · model coordinates':`${names[1]??'θ₁'} · model coordinates`});
 p.c.save();p.c.beginPath();p.c.rect(p.box.x,p.box.y,p.box.w,p.box.h);p.c.clip();contour(p,ref);
 if(preset==='funnel')points(p,refs.slice(0,500),'#788391',.16,2);
 const previous=comparison?.samples??[],previousBudget=Math.max(1,Math.floor(800/Math.max(1,previous.length)));
 previous.forEach((rows,i)=>points(p,rows.filter((_,j)=>j%Math.max(1,Math.ceil(rows.length/previousBudget))===0),COLORS[i],.4,2.3,[0,1],true));
 chains.forEach((rows,i)=>points(p,rows.filter((_,j)=>j%Math.max(1,Math.ceil(rows.length/1200))===0),COLORS[i],.45,2));p.c.restore();
 if(!all.length&&!rx&&!refs.length)note(p,'No computed results yet','Choose a model, then run inference.');
}
export function lowerPlot(canvas,{run,parameter=0,view='trace'}){
 const chains=run?.samples??[],names=run?.names??[],isVI=run?.settings?.algorithm==='vi';
 if(view==='objective'||isVI){
  const runs=run?.vi??[],histories=runs.map(v=>v.history??[]);const values=histories.map(h=>h.map(v=>typeof v==='number'?v:(v.optimization_elbo??v.objective??v.elbo??v.value)));
  const p=prepare(canvas,[0,Math.max(1,...histories.map(v=>v.length))],extent(values.flat()),{x:'Optimization update',y:'Noisy objective'});
  values.forEach((ys,i)=>line(p,ys.map((_,j)=>j+1),ys,COLORS[i%4],1.1));if(!values.flat().length)note(p,'No variational updates recorded');return;
 }
 if(view==='rank'){
  const vals=chains.flatMap((rows,chain)=>rows.map(row=>({v:row[parameter],chain}))).filter(v=>finite(v.v)).sort((a,b)=>a.v-b.v);
  const bins=20,counts=chains.map(()=>Array(bins).fill(0));for(let i=0;i<vals.length;){let j=i+1;while(j<vals.length&&vals[j].v===vals[i].v)j++;const rank=(i+j-1)/2;for(let k=i;k<j;k++)counts[vals[k].chain][Math.min(bins-1,Math.floor(rank/vals.length*bins))]++;i=j;}
  const p=prepare(canvas,[0,1],[0,Math.max(1,...counts.flat())*1.1],{x:'Pooled rank / retained draw count',y:'Draws per rank bin'});counts.forEach((ys,i)=>line(p,ys.map((_,j)=>(j+.5)/bins),ys,COLORS[i],1.3));if(!vals.length)note(p,'Rank comparison needs retained draws');return;
 }
 const values=chains.map(rows=>rows.map(v=>v[parameter]));const p=prepare(canvas,[0,Math.max(1,...values.map(v=>v.length))],extent(values.flat()),{x:'Retained draw · warmup excluded',y:names[parameter]??'Parameter'});
 values.forEach((ys,i)=>line(p,ys.map((_,j)=>j+1),ys,COLORS[i],.9));if(!values.flat().length)note(p,'Your chains will appear here','Warmup draws are excluded from these traces.');
}
