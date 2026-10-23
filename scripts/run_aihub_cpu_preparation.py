"""Continue the selected-snapshot CPU pipeline, then hand off to the GPU waiter."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    pending = path.with_name(path.name+'.pending')
    with pending.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
    pending.replace(path)


def bound_report(root):
    if (root/'.incomplete').exists() or (root/'INCOMPLETE').exists():
        raise ValueError(f'Incomplete output preserved: {root}')
    marker, report = read(root/'.complete'), read(root/'report.json')
    if not report.get('complete') or marker['report_sha256'] != hashlib.sha256((root/'report.json').read_bytes()).hexdigest():
        raise ValueError(f'Unbound completed report: {root}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-pid', required=True, type=int)
    p.add_argument('--native-prepared', required=True, type=Path)
    a = p.parse_args()
    project = Path.cwd().resolve()
    output = project/'runs/aihub_followup_v1'
    native = a.native_prepared.resolve(strict=True)
    cohort = project/'data/aihub_replay_cohort_v1'
    exported = project/'data/yolo26_aihub_semantic_v1'
    contract = output/'evaluation_contract'
    lock = output/'cpu_driver.lock'
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'pid':os.getpid(),'created':psutil.Process().create_time()},f)
    state = {'pid':os.getpid(),'created':psutil.Process().create_time(), 'complete':False,
             'native_pid':a.native_pid,'training_started':False,'completed_steps':[]}
    child = None

    def update(status, **extra):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write(output/'cpu_driver.json',state)
        print(json.dumps({'status':status,**extra},ensure_ascii=False),flush=True)

    def command(name, args):
        nonlocal child
        log = output/f'{name}.log'
        update('running',step=name,log=str(log))
        with log.open('x',encoding='utf-8') as f:
            child = subprocess.Popen([sys.executable,'-X','utf8','-B','-u',*map(str,args)],cwd=project,
                env=dict(os.environ,PYTHONPATH=str(project/'src')),stdout=f,stderr=subprocess.STDOUT)
            update('running',child_pid=child.pid)
            rc = child.wait(); child = None
        if rc: raise RuntimeError(f'{name} exited {rc}; see {log}')
        state['completed_steps'].append(name)
        update('step_complete',child_pid=None)

    try:
        update('waiting_for_native_preparation')
        native_identity = None
        if not (native/'.complete').exists():
            proc = psutil.Process(a.native_pid)
            cmd = proc.cmdline()
            if ('scripts/prepare_aihub_road.py' not in ' '.join(cmd).replace('\\','/')
                    or '--output' not in cmd
                    or (project/Path(cmd[cmd.index('--output')+1])).resolve() != native):
                raise RuntimeError('Native process does not match this exact prepared output')
            native_identity = {'pid':proc.pid,'created':proc.create_time(),'command':cmd}
            update('waiting_for_native_preparation',native_process=native_identity)
        deadline = time.monotonic()+6*3600
        while not (native/'.complete').exists():
            report = read(native/'report.json') if (native/'report.json').exists() else {}
            if report.get('error'): raise RuntimeError(report['error'])
            try:
                proc = psutil.Process(a.native_pid)
                if (proc.create_time() != native_identity['created']
                        or proc.cmdline() != native_identity['command']):
                    raise RuntimeError('Native preparation PID no longer matches')
            except psutil.NoSuchProcess:
                if (native/'.complete').exists(): break
                raise RuntimeError('Native preparation exited without a completion marker')
            if time.monotonic()>deadline: raise TimeoutError('Native preparation exceeded six hours; process untouched')
            time.sleep(15)
        native_report = bound_report(native)
        replay_count = native_report['manifests']['train']['samples']
        bdd_count = min(10000, replay_count)
        vkitti_count = replay_count-bdd_count
        if vkitti_count < 1:
            raise ValueError('Too few eligible AIHub frames for the planned balanced replay; review cohort')
        extracted = Path(read(project/'configs/local/aihub_followup_v1.json')['extraction']['output'])
        command('data_audit',['scripts/audit_aihub_subset.py','--extracted',extracted,'--prepared',native,'--output',output/'data_audit'])
        command('cohort',['scripts/build_aihub_cohort.py','--baseline-config','configs/local/large_vkitti_stage1_v1.json',
                        '--aihub-prepared',native,'--output',cohort,'--bdd-count',bdd_count,'--vkitti-count',vkitti_count,'--seed','42'])
        bound_report(cohort)
        command('semantic_export',['scripts/prepare_yolo_semantic.py','--config',cohort/'config.json','--output',exported,'--workers','4'])
        bound_report(exported)
        inputs = output/'evaluation_inputs'; inputs.mkdir(exist_ok=False)
        sources = read(project/'configs/local/large_vkitti_stage1_v1.json')['data']['val_manifests']
        labels = {13:'aihub_13_day_city',14:'aihub_14_day_expressway',15:'aihub_15_night_expressway'}
        rows = [json.loads(line) for line in (native/'val.jsonl').read_text(encoding='utf-8').splitlines()]
        for number, name in labels.items():
            path = inputs/(name+'.jsonl')
            selected = [row for row in rows if row['source_dataset_number']==number]
            if not selected: raise ValueError(f'Empty evaluation source {name}')
            with path.open('x',encoding='utf-8') as f:
                f.writelines(json.dumps(row,ensure_ascii=False)+'\n' for row in selected)
            sources.append({'name':name,'manifest':str(path)})
        spec = {'schema_version':1,'sources':sources,'runtime_threshold':.65,
                'retain_source_duplicate_multiplicity':True,
                'obstacle_root':str(project.parent/'datasets/road_obstacle21/dataset_ObstacleTrack/dataset_ObstacleTrack'),
                'anomaly_root':str(project.parent/'datasets/road_anomaly21/dataset_AnomalyTrack/dataset_AnomalyTrack')}
        write(inputs/'spec.json',spec)
        command('freeze_evaluation',['scripts/prepare_evaluation_contract.py','--spec',inputs/'spec.json','--output',contract])
        args = ['scripts/run_aihub_followup.py','--project',project,'--dataset',exported,'--config',cohort/'config.json',
                '--evaluation-contract',contract/'contract.json','--init-training-report',
                project/'runs/yolo26x_sem_experiment_v2/training/training_report.json','--output',output]
        # The GPU child owns its own lock/status and refuses to overlap existing training.
        update('cpu_complete_gpu_handoff',cpu_complete=True)
        with (output/'gpu_driver.stdout.log').open('x',encoding='utf-8') as f:
            child = subprocess.Popen([sys.executable,'-X','utf8','-B','-u',*map(str,args)],cwd=project,
                env=dict(os.environ,PYTHONPATH=str(project/'src')),stdout=f,stderr=subprocess.STDOUT)
            update('gpu_runner_active',gpu_runner_pid=child.pid)
            rc=child.wait();child=None
        if rc: raise RuntimeError(f'GPU runner exited {rc}; see gpu_runner.json and gpu_driver.stdout.log')
        update('complete',complete=True,gpu_pipeline_complete=True)
    except BaseException as error:
        update('recovery_required',error=f'{type(error).__name__}: {error}',
               active_child_pid=child.pid if child is not None and child.poll() is None else None)
        raise
    finally:
        # Never terminate child work on a driver error. Retain the lock if work is still active.
        if child is None or child.poll() is not None:
            lock.unlink()


if __name__=='__main__':
    main()
