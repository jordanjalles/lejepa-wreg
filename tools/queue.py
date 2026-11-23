#!/usr/bin/env python3
"""Run queue.txt lines one after another; register each run in the train-viz experiment at launch.

queue.txt line: NAME ARM LAM SEED SPLIT EPOCHS [extra train.py args...]   ('#' = comment)
A line is done when its sink folder has DONE or FAILED. The file is re-read after every run, so lines
can be added while the queue runs. Start detached:

  systemd-run --user --unit lejepa-wreg-queue python3 <data-root>/workspaces/lejepa-wreg/tools/queue.py
"""
import shlex, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEUE = HERE.parent / 'queue.txt'
SINK = Path('<data-root>/outputs/training-runs')
LOG = Path('<data-root>/outputs/lejepa-wreg/queue.log')
EXP_ADD = Path('<data-root>/workspaces/train-viz/scripts/experiment_add.py')
EXPERIMENT = 'lejepa-wreg'
REG = {'A': 'SIGReg + MLP-BN projector (official)', 'B': 'Wasserstein, no projector', 'C': 'SIGReg, no projector'}


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(line + '\n')


def pending():
    for raw in QUEUE.read_text().splitlines():
        if raw.strip() and not raw.strip().startswith('#'):
            f = shlex.split(raw)
            if not ((SINK / f[0] / 'DONE').exists() or (SINK / f[0] / 'FAILED').exists()):
                return f
    return None


while (f := pending()):
    name, arm, lam, seed, split, epochs, *extra = f
    role = 'baseline' if arm == 'A' and split == 'final' else 'control' if arm == 'C' else 'treatment'
    tags = {'--reg': 'reg {}', '--wsquare': 'outer square {}', '--final_norm': 'final LayerNorm {}'}
    tag = ', '.join(tags[x].format(extra[i + 1]) for i, x in enumerate(extra) if x in tags)
    reg = 'Wasserstein on the MLP-BN projector output' if arm == 'A' and 'wasserstein' in extra else REG[arm]
    desc = f"arm {arm}: {reg}, λ {float(lam):g}, seed {seed}, {split} split, {epochs} epochs" + (f" [{tag}]" if tag else '')
    log(f'start {name}: {desc}')
    subprocess.run([sys.executable, str(EXP_ADD), '--id', EXPERIMENT, '--status', 'running', '--run', f'shared/{name}',
                    '--label', f'{arm} λ{float(lam):g} s{seed} {split}' + (f' [{tag}]' if tag else ''), '--role', role], capture_output=True)
    subprocess.run(['docker', '--context', 'rootless', 'rm', '-f', f'lejepa-wreg-{name}'[:120].replace('.', '_')], capture_output=True)
    r = subprocess.run([sys.executable, str(HERE / 'launch.py'), '--wait', '--', '--arm', arm, '--lam', lam, '--seed', seed,
                        '--split', split, '--epochs', epochs, '--name', name, '--desc', desc, *extra])
    if r.returncode and not (SINK / name / 'FAILED').exists():
        (SINK / name / 'FAILED').write_text(f'container exit {r.returncode}\n')
    log(f"end {name}: exit {r.returncode} {'DONE' if (SINK / name / 'DONE').exists() else 'FAILED'}")
log('queue empty')
