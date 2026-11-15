"""Run integration checks and the experiment queue in the same persistent process."""
import time
from experiments.common.runtime import gpu_guard,atomic_json
from experiments.robust.design import OUT
from experiments.robust.training import read
from experiments.robust.check import checks,pretraining_checks
from experiments.robust.run import execute


def main():
    check_out=OUT.parent/'checks_v2_tgn'
    check_out.mkdir(parents=True,exist_ok=True)
    while True:
        guard=gpu_guard('cuda:0',0)
        try: guard.__enter__();break
        except BlockingIOError:
            atomic_json(OUT/'progress.json',dict(status='waiting_for_gpu_lock',stage='preflight',updated_at=time.time()))
            time.sleep(15)
    try:
        for file,fn in [('checks.json',checks),('pretraining_checks.json',pretraining_checks)]:
            if (check_out/file).exists() and read(check_out/file)['status']=='passed': continue
            atomic_json(OUT/'progress.json',dict(status='preflight',stage=file,updated_at=time.time()))
            fn(check_out.resolve())
    finally: guard.__exit__(None,None,None)
    execute(OUT.resolve())


if __name__=='__main__': main()
