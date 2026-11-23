#!/usr/bin/env python3
"""Verify step: a short run of each arm (default 40 steps), one after another, logs to OUT/smoke-<arm>.log."""
import subprocess, sys
from pathlib import Path

OUT = Path('<data-root>/outputs/lejepa-wreg')
HERE = Path(__file__).resolve().parent
arms = sys.argv[1].split(',') if len(sys.argv) > 1 else ['B:40', 'A:40', 'C:40']   # arm:steps
OUT.mkdir(parents=True, exist_ok=True)
for spec in arms:
    arm, steps = spec.split(':')
    lam = '0.5' if arm == 'B' else '0.02'
    name = f'lejepa-wreg-smoke-{arm}'
    subprocess.run(['docker', '--context', 'rootless', 'rm', '-f', f'lejepa-wreg-{name}'], capture_output=True)
    r = subprocess.run([sys.executable, str(HERE / 'launch.py'), '--wait', '--', '--arm', arm, '--lam', lam, '--seed', '0',
                        '--split', 'tune', '--epochs', '100', '--max_steps', steps, '--name', name,
                        '--desc', f'SMOKE TEST arm {arm}, {steps} steps'], capture_output=True, text=True)
    logs = subprocess.run(['docker', '--context', 'rootless', 'logs', f'lejepa-wreg-{name}'], capture_output=True, text=True)
    (OUT / f'smoke-{arm}.log').write_text(r.stdout + r.stderr + '\n---\n' + logs.stdout + logs.stderr)
    print(arm, r.stdout.strip().splitlines()[-1:] or r.stderr[-300:], flush=True)
