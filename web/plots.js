export const COLORS=['#414c99','#c08a3c','#4e897e','#b66b79'];
const ink='#34404b',muted='#6f7a7f',grid='#e6e4dc',paper='#fcfaf6';
const font='-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const plots=new WeakMap();
export const finite=x=>typeof x==='number'&&Number.isFinite(x);
const extent=values=>{let lo=Infinity,hi=-Infinity;for(const v of values)if(finite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}if(!finite(lo))return[-3,3];if(lo===hi)return[lo-1,hi+1];const pad=(hi-lo)*.08;return[lo-pad,hi+pad];};
const tick=v=>Math.abs(v)>1000?v.toExponential(1):Math.abs(v)<.01&&v!==0?v.toExponential(1):Number(v.toFixed(2)).toString();
function ticks(domain,count){const range=domain[1]-domain[0],raw=range/count,power=10**Math.floor(Math.log10(raw)),factor=raw/power,multiples=[1,2,2.5,5,10],step=multiples.reduce((best,n)=>Math.abs(n-factor)<Math.abs(best-factor)?n:best)*power;const values=[];for(let v=Math.ceil(domain[0]/step)*step;v<=domain[1]+step*1e-8;v+=step){values.push(Math.abs(v)<step*1e-8?0:v);if(values.length>15)break;}return values;}
const precise=v=>finite(v)?Math.abs(v)>1e4||Math.abs(v)<.001&&v!==0?v.toExponential(3):Number(v.toFixed(4)).toString():'Unavailable';
const chainLabel=(run,i,previous=false)=>`${previous?'Previous ':''}${run?.settings?.algorithm==='vi'?'VI fit':'Chain'} ${i+1}`;
function prepare(canvas,xDomain,yDomain,labels={}){
 const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),w=Math.max(1,rect.width),h=Math.max(1,rect.height);
 canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);
 const c=canvas.getContext('2d');c.scale(ratio,ratio);c.fillStyle=paper;c.fillRect(0,0,w,h);
 const compact=w<480,short=h<250,left=compact?57:67,right=compact?17:26;
 const box={x:left,y:short?25:37,w:Math.max(1,w-left-right),h:Math.max(1,h-(short?25:37)-(compact?61:64))};
 const X=x=>box.x+(x-xDomain[0])/(xDomain[1]-xDomain[0])*box.w,Y=y=>box.y+box.h-(y-yDomain[0])/(yDomain[1]-yDomain[0])*box.h;
 c.font=`${compact||short?12:13}px ${font}`;c.lineWidth=1;
 const nx=compact?4:5,ny=short?3:4;
 for(const value of ticks(yDomain,ny)){
  const y=Y(value);c.strokeStyle=grid;c.beginPath();c.moveTo(box.x,y+.5);c.lineTo(box.x+box.w,y+.5);c.stroke();
  c.fillStyle=muted;c.textAlign='right';c.fillText(tick(value),box.x-11,y+4);
 }
 c.strokeStyle='#c9cdc8';c.beginPath();c.moveTo(box.x,box.y+box.h+.5);c.lineTo(box.x+box.w,box.y+box.h+.5);c.stroke();
 for(const value of ticks(xDomain,nx)){
  const x=X(value);c.strokeStyle='#c9cdc8';c.beginPath();c.moveTo(x,box.y+box.h);c.lineTo(x,box.y+box.h+5);c.stroke();
  c.fillStyle=muted;c.textAlign=x<box.x+10?'left':x>box.x+box.w-10?'right':'center';c.fillText(tick(value),x,box.y+box.h+22);
 }
 c.fillStyle=muted;c.textAlign='center';c.font=`${compact?12:13}px ${font}`;
 fitText(c,labels.x||'',box.x+box.w/2,h-11,box.w+15);
 c.save();c.translate(compact?14:17,box.y+box.h/2);c.rotate(-Math.PI/2);fitText(c,labels.y||'',0,0,box.h+25);c.restore();
 const p={c,box,X,Y,w,h,xDomain,yDomain,labels,series:[],clouds:[],grid:null,summary:''};plots.set(canvas,p);
 canvas.setAttribute('aria-label',`${labels.y||'Plot'} by ${labels.x||'coordinate'}. Focus to inspect values with the arrow keys.`);
 return p;
}
function fitText(c,text,x,y,width){if(c.measureText(text).width<=width)c.fillText(text,x,y);else{const words=text.split(' ');let first='',second='';for(const word of words){if(!second&&c.measureText(`${first} ${word}`).width<width)first+=(first?' ':'')+word;else second+=(second?' ':'')+word;}c.fillText(first,x,y-7);c.fillText(second,x,y+7);}}
function describe(canvas,p,summary){p.summary=summary;canvas.setAttribute('aria-label',`${summary} ${p.labels.y} by ${p.labels.x}. Focus and use the arrow keys to inspect plotted values.`);}
function within(p,draw){const{c,box}=p;c.save();c.beginPath();c.rect(box.x,box.y,box.w,box.h);c.clip();draw();c.restore();}
function line(p,x,y,color,width=1.7,dash=[],label){
 const{c,X,Y}=p;within(p,()=>{c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.lineJoin='round';c.beginPath();let started=false;for(let i=0;i<x.length;i++){if(!finite(x[i])||!finite(y[i])){started=false;continue;}if(!started)c.moveTo(X(x[i]),Y(y[i]));else c.lineTo(X(x[i]),Y(y[i]));started=true;}c.stroke();});
 if(label)p.series.push({label,color,x,y});
}
function area(p,x,low,high,color){const{c,X,Y}=p;if(!x?.length||!low?.length||!high?.length)return;within(p,()=>{c.fillStyle=color;c.beginPath();x.forEach((v,i)=>i?c.lineTo(X(v),Y(low[i])):c.moveTo(X(v),Y(low[i])));for(let i=x.length-1;i>=0;i--)c.lineTo(X(x[i]),Y(high[i]));c.closePath();c.fill();});}
function points(p,rows,color,alpha=.5,r=2,indices=[0,1],hollow=false,label){const{c,X,Y}=p;within(p,()=>{c.fillStyle=color;c.strokeStyle=color;c.lineWidth=1.2;c.globalAlpha=alpha;for(const row of rows){const a=row[indices[0]],b=row[indices[1]];if(!finite(a)||!finite(b))continue;c.beginPath();c.arc(X(a),Y(b),r,0,2*Math.PI);if(hollow)c.stroke();else c.fill();}});if(label)p.clouds.push({label,color,rows,indices});}
function note(p,text,sub=''){const{c,box}=p;c.fillStyle=ink;c.textAlign='center';c.font=`${p.w<480?12:14}px ${font}`;fitText(c,text,box.x+box.w/2,box.y+box.h/2-10,box.w-14);if(sub){c.font=`${p.w<480?11:12}px ${font}`;c.fillStyle=muted;fitText(c,sub,box.x+box.w/2,box.y+box.h/2+23,box.w-14);}}
function contour(p,ref){const g=ref?.grid;if(!g?.x?.length||!g?.y?.length||!g?.z?.length)return;
 const{x,y,z}=g,max=Math.max(...z.flat().filter(finite)),levels=[.05,.15,.35,.60,.85].map(v=>v*max),{c,X,Y}=p;p.grid=g;
 // These are equal-density contours, never probability-mass regions.
 for(const level of levels){c.strokeStyle='#737fa56b';c.lineWidth=1.1;c.beginPath();for(let j=0;j<y.length-1;j++)for(let i=0;i<x.length-1;i++){
  const corners=[[x[i],y[j],z[j]?.[i]],[x[i+1],y[j],z[j]?.[i+1]],[x[i+1],y[j+1],z[j+1]?.[i+1]],[x[i],y[j+1],z[j+1]?.[i]]],cross=[];
  for(let edge=0;edge<4;edge++){const a=corners[edge],b=corners[(edge+1)%4];if((a[2]<level)!==(b[2]<level)){const t=(level-a[2])/(b[2]-a[2]);cross.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);}}
  for(let k=0;k+1<cross.length;k+=2){c.moveTo(X(cross[k][0]),Y(cross[k][1]));c.lineTo(X(cross[k+1][0]),Y(cross[k+1][1]));}
 }c.stroke();}
}
function betaReference(p,ref){
 const{c,X,Y,box}=p,x=ref.x??[],posterior=ref.posterior;
 if(posterior){
  area(p,x,x.map(()=>0),posterior,'#414c9918');
  const interval=ref.interval??[];
  if(interval.length===2&&interval.every(finite)){
   // Clip the exact density fill at the independently computed 5% and 95%
   // quantiles. No histogram mass or sampled quantile is implied here.
   c.save();c.beginPath();c.rect(X(interval[0]),box.y,X(interval[1])-X(interval[0]),box.h);c.clip();area(p,x,x.map(()=>0),posterior,'#414c9923');c.restore();
   const y=box.y+box.h+1;c.strokeStyle='#414c99';c.lineWidth=2;c.beginPath();c.moveTo(X(interval[0]),y);c.lineTo(X(interval[1]),y);c.stroke();
   for(const q of interval){c.beginPath();c.moveTo(X(q),y-4);c.lineTo(X(q),y+4);c.stroke();}
  }
 }
 if(ref.prior)line(p,x,ref.prior,'#93a28f',1.8,[7,5],'Prior density');
 if(ref.likelihood)line(p,x,ref.likelihood,'#bd9451',1.6,[2,5],'Scaled likelihood');
 if(posterior)line(p,x,posterior,COLORS[0],2.6,[],'Exact posterior density');
 const mean=ref.mean?.[0];
 if(finite(mean)&&posterior?.length){
  const index=nearest(x,mean),height=posterior[index];
  within(p,()=>{c.strokeStyle='#414c998c';c.lineWidth=1;c.setLineDash([3,4]);c.beginPath();c.moveTo(X(mean),box.y+box.h);c.lineTo(X(mean),Y(height)-9);c.stroke();});
  const text=`mean ${precise(mean)}`;c.font=`${p.w<480?11:12}px ${font}`;c.textAlign='center';c.fillStyle=COLORS[0];const labelX=Math.max(box.x+c.measureText(text).width/2,Math.min(box.x+box.w-c.measureText(text).width/2,X(mean)));c.fillText(text,labelX,box.y-15);
  if(ref.interval?.length===2){const annotation='Shaded: exact 90% interval',meanRight=labelX+c.measureText(text).width/2;c.font=`11px ${font}`;if(meanRight+18<box.x+box.w-c.measureText(annotation).width){c.textAlign='right';c.fillStyle=muted;c.fillText(annotation,box.x+box.w,box.y-15);}}
 }
}
export function mainPlot(canvas,{preset,reference:ref={},run,view='density',parameter=0,transitionIndex=-1,comparison=null}){
 const chains=run?.samples??[],all=chains.flat(),names=run?.names??[];
 if(view==='trajectory'){
  const transitions=run?.transitions??[],t=transitions[transitionIndex<0?transitions.length-1:Math.min(transitionIndex,transitions.length-1)];
  const raw=t?.trajectory??[],limit=Number.isFinite(arguments[1].replayCount)?arguments[1].replayCount:raw.length,visible=raw.slice(0,limit),rows=raw.map(v=>v.position??[]),dims=rows[0]?.length??2;
  const coordinate=(point,index)=>[dims===1?index:point[0],dims===1?point[0]:point[1]];
  const xy=rows.map(coordinate),p=prepare(canvas,extent(xy.map(v=>v[0])),extent(xy.map(v=>v[1])),{x:dims===1?'Recorded integration index':'Working coordinate x₀',y:dims===1?'Working coordinate x₀':'Working coordinate x₁'});
  describe(canvas,p,'Recorded numerical integration positions. Green marks the start; a rose diamond marks the selected endpoint.');
  if(!rows.length){note(p,'No recorded transition yet','Run HMC or NUTS to inspect integration steps.');return;}
  const start=t.starting_position??rows[0];
  if(t.algorithm==='hmc')line(p,xy.slice(0,limit).map(v=>v[0]),xy.slice(0,limit).map(v=>v[1]),COLORS[0],1.6,[4,4]);
  else{
   // Never join traversal-order leaves across opposite sides of a NUTS tree.
   let minus={position:start,index:0},plus={position:start,index:0};
   for(const branch of(t.branches??[]).filter(v=>v.kind==='expansion')){
    const edge=branch.direction===-1?minus:plus;
    const members=raw.map((v,index)=>({...v,index})).filter(v=>v.integration_step>=branch.first_integration_step&&v.integration_step<=branch.last_integration_step);
    const shown=members.filter(v=>v.index<limit),segment=[coordinate(edge.position,edge.index),...shown.map(v=>coordinate(v.position,v.index))];
    if(shown.length)line(p,segment.map(v=>v[0]),segment.map(v=>v[1]),branch.direction===-1?COLORS[0]:COLORS[1],1.6,[4,4]);
    if(members.length){const last=members.at(-1);if(branch.direction===-1)minus=last;else plus=last;}
   }
  }
  within(p,()=>{for(let i=0;i<visible.length;i++){const q=coordinate(visible[i].position,i),{c,X,Y}=p;c.beginPath();c.arc(X(q[0]),Y(q[1]),3.3,0,Math.PI*2);c.strokeStyle=visible[i].continuing===false?COLORS[3]:COLORS[0];c.fillStyle=visible[i].eligible===false?paper:COLORS[0];c.fill();c.stroke();}});
  p.clouds.push({label:'Recorded integration position',color:COLORS[0],rows:visible.map((v,i)=>coordinate(v.position,i)),indices:[0,1]});
  points(p,[coordinate(start,0)],COLORS[2],1,5,[0,1],false,'Starting position');
  if(limit>=raw.length&&t.selected_position){const selectedIndex=rows.findIndex(v=>v.every((x,i)=>Math.abs(x-t.selected_position[i])<1e-10));const q=coordinate(t.selected_position,Math.max(0,selectedIndex)),{c,X,Y}=p;within(p,()=>{c.fillStyle=COLORS[3];c.beginPath();c.moveTo(X(q[0]),Y(q[1])-7);c.lineTo(X(q[0])+7,Y(q[1]));c.lineTo(X(q[0]),Y(q[1])+7);c.lineTo(X(q[0])-7,Y(q[1]));c.closePath();c.fill();});p.clouds.push({label:'Selected endpoint',color:COLORS[3],rows:[q],indices:[0,1]});}
  return;
 }
 if(preset==='custom'&&(run?.names?.length??1)===1){
  const previous=comparison?.samples??[],values=all.map(r=>r[0]),domain=extent([...values,...previous.flat().map(r=>r[0])]),bins=30,width=(domain[1]-domain[0])/bins;
  const histogram=rows=>{const hist=Array(bins).fill(0);for(const r of rows)hist[Math.max(0,Math.min(bins-1,Math.floor((r[0]-domain[0])/width)))]++;return hist.map(v=>rows.length?v/rows.length/width:0);};
  const counts=chains.map(histogram),priorCounts=previous.map(histogram),p=prepare(canvas,domain,[0,Math.max(1,...counts.flat(),...priorCounts.flat())*1.1],{x:`${names[0]??'Parameter'} · model coordinates`,y:'Marginal density'});
  describe(canvas,p,'Each curve is an independently normalized 30-bin draw histogram. This custom model has no independent analytic reference.');
  priorCounts.forEach((ys,i)=>{if(previous[i].length)line(p,ys.map((_,j)=>domain[0]+(j+.5)*width),ys,COLORS[i],1.5,[4,3],`${chainLabel(comparison,i,true)} histogram`);});counts.forEach((ys,i)=>{if(chains[i].length)line(p,ys.map((_,j)=>domain[0]+(j+.5)*width),ys,COLORS[i],1.7,[],`${chainLabel(run,i)} histogram`);});if(!values.length&&!previous.some(rows=>rows.length))note(p,'No retained draws yet','This model has no independent analytic reference.');return;
 }
 if(preset==='beta'){
  const x=ref.x??[],ys=[ref.prior??[],ref.posterior??[],ref.likelihood??[]],bins=30;
  // Each chain or variational fit has its own normalized histogram.
  const histogram=rows=>{const counts=Array(bins).fill(0);for(const row of rows){const v=row[0];if(v>=0&&v<=1)counts[Math.min(bins-1,Math.floor(v*bins))]++;}return rows.length?counts.map(v=>v/rows.length*bins):counts;};
  const histograms=chains.map(histogram),previous=comparison?.samples??[],comparisonHistograms=previous.map(histogram);
  const max=Math.max(1,...ys.flat().filter(finite),...histograms.flat(),...comparisonHistograms.flat()),p=prepare(canvas,[0,1],[0,max*1.1],{x:'Success probability θ',y:'Probability density'});
  betaReference(p,ref);
  describe(canvas,p,`Prior, scaled likelihood and exact Beta posterior.${finite(ref.mean?.[0])?` Exact mean ${precise(ref.mean[0])}.`:''}${ref.interval?.length===2?` The darker shaded 90% reference interval is ${ref.interval.map(precise).join(' to ')}.`:''} Draw histograms remain separate for each chain or variational fit.`);
  comparisonHistograms.forEach((ys,i)=>{if(previous[i].length)line(p,ys.map((_,j)=>(j+.5)/bins),ys,COLORS[i],1.4,[4,3],`${chainLabel(comparison,i,true)} histogram`);});
  histograms.forEach((ys,i)=>{if(chains[i]?.length)line(p,ys.map((_,j)=>(j+.5)/bins),ys,COLORS[i],1.5,[],`${chainLabel(run,i)} histogram`);});
  if(!x.length&&!all.length)note(p,'Every belief begins somewhere','The exact reference appears when Python is ready.');return;
 }
 if(preset==='regression'){
  const x=ref.x_grid??[],xd=ref.x_data??[],yd=ref.y_data??[],p=prepare(canvas,extent([...x,...xd]),extent([...yd,...(ref.predictive_lower??[]),...(ref.predictive_upper??[])]),{x:'Predictor x',y:'Response y'});
  describe(canvas,p,'Analytic regression. The inner 90% mean-response credible band excludes observation noise; the outer 90% predictive band includes it.');
  if(x.length){area(p,x,ref.predictive_lower,ref.predictive_upper,'#bec4df65');area(p,x,ref.mean_lower,ref.mean_upper,'#717fc554');line(p,x,ref.mean_response,COLORS[0],2.2,[],'Exact mean response');p.series.push({label:'Mean 90% lower',x,y:ref.mean_lower},{label:'Mean 90% upper',x,y:ref.mean_upper},{label:'Predictive 90% lower',x,y:ref.predictive_lower},{label:'Predictive 90% upper',x,y:ref.predictive_upper});}
  const regressionLines=(groups,budget,previous)=>{const perGroup=Math.max(1,Math.floor(budget/Math.max(1,groups.length)));groups.forEach((rows,i)=>rows.filter((_,j)=>j%Math.max(1,Math.ceil(rows.length/perGroup))===0).slice(0,perGroup).forEach(row=>line(p,x,x.map(v=>row[0]+row[1]*v),COLORS[i]+(previous?'65':'50'),1,previous?[3,3]:[])));};
  regressionLines(comparison?.samples??[],12,true);regressionLines(chains,16,false);points(p,xd.map((v,i)=>[v,yd[i]]),ink,1,3.7,[0,1],false,'Observed data');
  if(!x.length)note(p,'Two kinds of uncertainty','The analytic prediction bands appear when Python is ready.');return;
 }
 const rx=ref.grid?.x,ry=ref.grid?.y,refs=ref.reference_samples??[];
 let xlim=rx?[rx[0],rx.at(-1)]:extent([...all,...refs.slice(0,500)].map(v=>v[0]));
 let ylim=ry?[ry[0],ry.at(-1)]:extent([...all,...refs.slice(0,500)].map(v=>v[1]));
 if(preset==='funnel'){xlim=[-9,9];ylim=[-15,15];}
 const p=prepare(canvas,xlim,ylim,{x:preset==='funnel'?'Hierarchy scale y':`${names[0]??'θ₀'} · model coordinates`,y:preset==='funnel'?'Child x₀':`${names[1]??'θ₁'} · model coordinates`});
 describe(canvas,p,preset==='funnel'?'Model-space marginal projection. Faint independent generative reference draws and retained chain draws.':`Equal-density reference contours and retained draws, shown in model coordinates. Contours are not probability-mass regions.`);
 within(p,()=>contour(p,ref));
 if(preset==='funnel')points(p,refs.slice(0,500),'#788391',.19,2.1,[0,1],false,'Independent reference draw');
 const previous=comparison?.samples??[],previousBudget=Math.max(1,Math.floor(800/Math.max(1,previous.length)));
 previous.forEach((rows,i)=>points(p,rows.filter((_,j)=>j%Math.max(1,Math.ceil(rows.length/previousBudget))===0),COLORS[i],.4,2.6,[0,1],true,chainLabel(comparison,i,true)));
 chains.forEach((rows,i)=>points(p,rows.filter((_,j)=>j%Math.max(1,Math.ceil(rows.length/1200))===0),COLORS[i],.5,2.3,[0,1],false,chainLabel(run,i)));
 if(!all.length&&!rx&&!refs.length)note(p,'A distribution, waiting to be explored','Choose a model, then run inference.');
}
export function lowerPlot(canvas,{run,parameter=0,view='trace'}){
 const chains=run?.samples??[],names=run?.names??[],isVI=run?.settings?.algorithm==='vi';
 if(view==='objective'||isVI){
  const runs=run?.vi??[],histories=runs.map(v=>v.history??[]),values=histories.map(h=>h.map(v=>typeof v==='number'?v:(v.optimization_elbo??v.objective??v.elbo??v.value)));
  const p=prepare(canvas,[0,Math.max(1,...histories.map(v=>v.length))],extent(values.flat()),{x:'Optimization update',y:'Noisy objective'});
  describe(canvas,p,'Recorded noisy variational optimization objectives. Independent fits are shown separately; a stable objective does not prove posterior accuracy.');
  values.forEach((ys,i)=>line(p,ys.map((_,j)=>j+1),ys,COLORS[i%4],1.4,[],`VI fit ${i+1}`));if(!values.flat().length)note(p,'No variational updates recorded');return;
 }
 if(view==='rank'){
  const vals=chains.flatMap((rows,chain)=>rows.map(row=>({v:row[parameter],chain}))).filter(v=>finite(v.v)).sort((a,b)=>a.v-b.v);
  const bins=20,counts=chains.map(()=>Array(bins).fill(0));for(let i=0;i<vals.length;){let j=i+1;while(j<vals.length&&vals[j].v===vals[i].v)j++;const rank=(i+j-1)/2;for(let k=i;k<j;k++)counts[vals[k].chain][Math.min(bins-1,Math.floor(rank/vals.length*bins))]++;i=j;}
  const p=prepare(canvas,[0,1],[0,Math.max(1,...counts.flat())*1.1],{x:'Pooled rank / retained draw count',y:'Draws per rank bin'});
  describe(canvas,p,'Twenty-bin pooled rank histograms, one per chain. Tied values receive their average pooled rank.');
  counts.forEach((ys,i)=>line(p,ys.map((_,j)=>(j+.5)/bins),ys,COLORS[i],1.6,[],`Chain ${i+1} rank-bin count`));if(!vals.length)note(p,'Rank comparison needs retained draws');return;
 }
 const values=chains.map(rows=>rows.map(v=>v[parameter])),p=prepare(canvas,[0,Math.max(1,...values.map(v=>v.length))],extent(values.flat()),{x:'Retained draw · warmup excluded',y:names[parameter]??'Parameter'});
 describe(canvas,p,`Retained ${names[parameter]??'parameter'} draws in sequence for each chain. Warmup is excluded.`);
 values.forEach((ys,i)=>line(p,ys.map((_,j)=>j+1),ys,COLORS[i],1.05,[],`Chain ${i+1}`));if(!values.flat().length)note(p,'The path of your chains','Retained draws appear here as inference progresses.');
}
function nearest(values,value){let best=-1,distance=Infinity;for(let i=0;i<values.length;i++){const d=Math.abs(values[i]-value);if(finite(values[i])&&d<distance){best=i;distance=d;}}return best;}
function inspect(p,px,py){
 const x=p.xDomain[0]+(px-p.box.x)/p.box.w*(p.xDomain[1]-p.xDomain[0]),rows=[];
 if(p.series.length){for(const s of p.series){const i=nearest(s.x,x);if(i>=0&&finite(s.y?.[i]))rows.push(`${s.label}: ${precise(s.y[i])} · x ${precise(s.x[i])}`);}return{heading:'Nearest plotted values',rows};}
 if(p.clouds.length){let best=null,distance=Infinity;for(const cloud of p.clouds)for(const row of cloud.rows){const a=row[cloud.indices[0]],b=row[cloud.indices[1]],d=Math.hypot(p.X(a)-px,p.Y(b)-py);if(finite(a)&&finite(b)&&a>=p.xDomain[0]&&a<=p.xDomain[1]&&b>=p.yDomain[0]&&b<=p.yDomain[1]&&d<distance){best={cloud,a,b};distance=d;}}if(best&&distance<35)return{heading:best.cloud.label,rows:[`${p.labels.x}: ${precise(best.a)}`,`${p.labels.y}: ${precise(best.b)}`]};}
 if(p.grid){const i=nearest(p.grid.x,x),value=p.yDomain[1]-(py-p.box.y)/p.box.h*(p.yDomain[1]-p.yDomain[0]),j=nearest(p.grid.y,value);if(i>=0&&j>=0&&finite(p.grid.z[j]?.[i]))return{heading:'Nearest evaluated reference point',rows:[`x ${precise(p.grid.x[i])} · y ${precise(p.grid.y[j])}`,`Joint density: ${precise(p.grid.z[j][i])}`]};}
 return null;
}
/** Wire once after mounting. The current render supplies exact plotted values;
 * inspection never interpolates or generates draws. Returns an event cleanup. */
export function installPlotInspection(canvas,tooltip){
 let keyboardIndex=50;canvas.tabIndex=0;canvas.style.cursor='crosshair';tooltip.hidden=true;
 if(!tooltip.id)tooltip.id=`${canvas.id||'plot'}-inspection`;tooltip.setAttribute('role','tooltip');tooltip.setAttribute('aria-live','polite');const describedBy=new Set((canvas.getAttribute('aria-describedby')||'').split(/\s+/).filter(Boolean));describedBy.add(tooltip.id);canvas.setAttribute('aria-describedby',[...describedBy].join(' '));
 const hide=()=>{tooltip.hidden=true;};
 const show=(px,py,clientX,clientY)=>{
  const p=plots.get(canvas);if(!p||px<p.box.x||px>p.box.x+p.box.w||py<p.box.y||py>p.box.y+p.box.h){hide();return;}
  const result=inspect(p,px,py);if(!result?.rows.length){hide();return;}
  const title=document.createElement('div');title.className='plot-tooltip-label';title.textContent=result.heading;
  const entries=result.rows.slice(0,11).map(value=>{const row=document.createElement('div');row.className='plot-tooltip-row';row.textContent=value;return row;});tooltip.replaceChildren(title,...entries);tooltip.hidden=false;
  const tw=tooltip.offsetWidth,th=tooltip.offsetHeight;tooltip.style.left=`${Math.max(10,Math.min(clientX+16,innerWidth-tw-10))}px`;tooltip.style.top=`${Math.max(10,Math.min(clientY+18,innerHeight-th-10))}px`;
 };
 const pointer=event=>{const r=canvas.getBoundingClientRect();show(event.clientX-r.left,event.clientY-r.top,event.clientX,event.clientY);};
 const keyboard=()=>{const p=plots.get(canvas);if(!p)return;const r=canvas.getBoundingClientRect();let px=p.box.x+p.box.w*keyboardIndex/100,py=p.box.y+p.box.h/2;if(!p.series.length&&p.clouds.length){const visible=p.clouds.flatMap(cloud=>cloud.rows.map(row=>[row[cloud.indices[0]],row[cloud.indices[1]]])).filter(([x,y])=>finite(x)&&finite(y)&&x>=p.xDomain[0]&&x<=p.xDomain[1]&&y>=p.yDomain[0]&&y<=p.yDomain[1]);if(visible.length){const point=visible[Math.round((visible.length-1)*keyboardIndex/100)];px=p.X(point[0]);py=p.Y(point[1]);}}show(px,py,r.left+px,r.top+py);};
 const key=event=>{if(event.key==='Escape'){hide();return;}if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();keyboardIndex=event.key==='Home'?0:event.key==='End'?100:Math.max(0,Math.min(100,keyboardIndex+(event.key==='ArrowRight'?1:-1)));keyboard();}};
 const events=[['pointermove',pointer],['pointerdown',pointer],['pointerleave',hide],['focus',keyboard],['blur',hide],['keydown',key]];for(const[event,handler]of events)canvas.addEventListener(event,handler);
 return()=>{for(const[event,handler]of events)canvas.removeEventListener(event,handler);hide();};
}
