"""Run the same engine used by the browser without any service."""
import argparse
from pathlib import Path
from .examples import preset
from .runtime import Experiment
from .serialization import dumps,loads,draws_csv,extract_checkpoint

def main():
    p=argparse.ArgumentParser(description='Posterior CPU inference workbench')
    p.add_argument('--preset',choices=['beta','regression','gaussian','funnel','mixture'],default='beta'); p.add_argument('--model'); p.add_argument('--algorithm',choices=['nuts','hmc','vi'],default='nuts'); p.add_argument('--chains',type=int,default=4); p.add_argument('--warmup',type=int,default=200); p.add_argument('--draws',type=int,default=200); p.add_argument('--iterations',type=int,default=1000); p.add_argument('--seed',type=int,default=42); p.add_argument('--max-seconds',type=float,default=120); p.add_argument('--resume'); p.add_argument('--output',default='experiment.json'); p.add_argument('--csv')
    a=p.parse_args()
    if a.resume:
        data=loads(Path(a.resume).read_text()); run=Experiment.restore(extract_checkpoint(data))
        if run.status!='running':run.set_status('running')
    else:
        spec=loads(Path(a.model).read_text()) if a.model else preset(a.preset)['spec']; run=Experiment(spec,{k:getattr(a,k) for k in ['algorithm','chains','warmup','draws','iterations','seed','max_seconds']})
    while run.status=='running': run.step(25)
    out=run.snapshot(); Path(a.output).write_text(dumps(out)+'\n')
    if a.csv: Path(a.csv).write_text(draws_csv(run.samples,run.model.reporting_names))
    print(dumps(run.compact()))
    if run.status!='complete': raise SystemExit(2)

if __name__=='__main__': main()
