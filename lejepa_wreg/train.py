"""One run of the projector-free Wasserstein vs LeJEPA experiment (see README.md).

  python -m lejepa_wreg.train --arm B --lam 0.1 --seed 0 --split tune --epochs 100 --name tune-B-lam0.1-s0

Writes a train-viz harness folder <sink>/<name>/{config.json, metrics.jsonl, DESCRIPTION.md, DONE|FAILED}
and checkpoints to <ckpt>/<name>/. Eval records carry test_clean = 1 - probe top-1 (lower is better).
Built-in checks abort the run (FAILED + exit 3) when a treatment is not applied as specified.
"""
import argparse, hashlib, json, math, os, subprocess, sys, time, traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from . import data, evals
from .model import Net, SIGReg, WREG_WEIGHTS, derived_seed, private_rng, trunk_hash, wreg

HERE = Path(__file__).resolve().parents[1]
COLLAPSE_FRAC = 0.05   # abort when the per-dimension std of probe-eval features falls below this share of its init value


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['A', 'B', 'C'], required=True)
    p.add_argument('--lam', type=float, required=True, help='loss = (1 - lam) * align + lam * reg')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', choices=['tune', 'final'], required=True)
    p.add_argument('--epochs', type=int, default=800)
    p.add_argument('--max_steps', type=int, default=0, help='stop early (smoke / benchmark); schedule still uses --epochs')
    p.add_argument('--name', required=True)
    p.add_argument('--desc', default='')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--V', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--wd', type=float, default=5e-2)
    p.add_argument('--proj_dim', type=int, default=16)
    p.add_argument('--final_norm', type=int, default=None, help="keep the ViT's final LayerNorm (default: off in B, on in A and C)")
    p.add_argument('--reg', choices=['sigreg', 'wasserstein'], default=None,
                   help='regularizer (default: wasserstein in B, sigreg in A and C); A + wasserstein = test of the projector')
    p.add_argument('--wsquare', type=int, default=1, help="1: notebook's outer .square() on each Wasserstein term; 0: undo it")
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--compile', type=int, default=1)
    p.add_argument('--eval_every', type=int, default=50, help='epochs between probe evals')
    p.add_argument('--drift_every', type=int, default=10, help='epochs between drift snapshots')
    p.add_argument('--ckpt_every', type=int, default=100, help='epochs between checkpoints')
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--data', default='/datasets/imagenette')
    p.add_argument('--sink', default='/sink')
    p.add_argument('--ckpt', default='/ckpt')
    return p.parse_args()


class Run:
    def __init__(self, a):
        self.dir = Path(a.sink) / a.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.dir / 'metrics.jsonl', 'a')

    def log(self, rec):
        self.f.write(json.dumps(rec) + '\n'); self.f.flush()

    def fail(self, msg):
        (self.dir / 'FAILED').write_text(msg + '\n')
        print('CHECK FAILED:', msg, flush=True)
        sys.exit(3)


def git_rev(path):
    try:
        return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    a = args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda'
    run = Run(a)

    ssl_items, ptr_items, pev_items = data.splits(a.data, a.split)
    ssl = data.Views(ssl_items, a.V)
    g_data = torch.Generator().manual_seed(derived_seed('data', a.seed))     # shuffle + worker aug seeds
    loader = DataLoader(ssl, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers,
                        generator=g_data, persistent_workers=True, pin_memory=True, prefetch_factor=4)
    mk = lambda items: DataLoader(data.Views(items, 1), batch_size=512, num_workers=a.workers)
    probe_tr, probe_ev = mk(ptr_items), mk(pev_items)
    fixed_x = torch.stack([data.Views(pev_items[:512], 1)[i][0] for i in range(512)]).to(dev)   # drift images

    if a.reg is None:
        a.reg = 'wasserstein' if a.arm == 'B' else 'sigreg'
    if a.arm == 'C' and a.reg != 'sigreg' or a.arm == 'B' and a.reg != 'wasserstein':
        run.fail(f'arm {a.arm} with --reg {a.reg}: only arm A may swap its regularizer')
    if a.final_norm is None:
        a.final_norm = int(a.arm != 'B')
    net = Net(a.arm, a.seed, a.proj_dim, final_norm=bool(a.final_norm)).to(dev)
    if bool(a.final_norm) == isinstance(net.backbone.norm, torch.nn.Identity):
        run.fail(f'final LayerNorm state wrong: final_norm={a.final_norm}, backbone.norm={type(net.backbone.norm).__name__}')
    sigreg = SIGReg().to(dev)
    th = trunk_hash(net)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': params, 'lr': a.lr, 'weight_decay': a.wd}])
    spe = len(loader)
    total = spe * a.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=spe)
    s2 = CosineAnnealingLR(opt, T_max=total - spe, eta_min=1e-3)    # absolute floor, as in the example's code
    sched = SequentialLR(opt, schedulers=[s1, s2], milestones=[spe])
    scaler = GradScaler(enabled=True)
    fwd = torch.compile(net) if a.compile else net

    reg_name = 'wasserstein' if a.reg == 'wasserstein' else 'sigreg'
    cfg = dict(vars(a), run_type='ssl', steps=a.max_steps or total, total_schedule_steps=total, steps_per_epoch=spe,
               n_ssl=len(ssl_items), n_probe_train=len(ptr_items), n_probe_eval=len(pev_items),
               probe_eval_set='official validation' if a.split == 'final' else f'train hold-out ({data.HOLDOUT_FRAC:.0%} per class, seed {data.HOLDOUT_SEED})',
               feature_point='timm vit_small_patch8_224 (num_classes=0) forward_head(pre_logits=True): CLS token '
                             + ('after the final LayerNorm' if a.final_norm else 'of the residual stream, final LayerNorm removed (intentional divergence, user 2026-11-09)'),
               feature_shape=[a.bs * a.V, net.feat_dim], loss_path_shape=[a.V, a.bs, a.proj_dim if a.arm == 'A' else net.feat_dim],
               loss_path={'A': 'native 384 -> Linear 512 (timm head init) -> MLP 2048-BN-ReLU-2048-BN-ReLU-proj_dim',
                          'B': 'native 384 features', 'C': 'native 384 features'}[a.arm],
               regularizer=reg_name, reg_batch='per view: N = bs source images; mean over V views',
               bn_batch='A only: BatchNorm1d over all V*bs rows of the flattened batch (official code)' if a.arm == 'A' else None,
               wasserstein=dict(source=json.loads((HERE / 'lejepa_wreg' / 'wreg_nb_sources.json').read_text()),
                                weights=WREG_WEIGHTS, outer_power='each term returns loss.square() (squared mean of weighted squared quantile gaps)'
                                if a.wsquare else 'outer .square() undone: sqrt of each notebook term per view (--wsquare 0)',
                                applied_to='A projector output z (proj_dim)' if a.arm == 'A' else 'native 384 features',
                                bulk_sigma=1.0, n_slices={'swd': 1024}, prior='torch.randn_like, N(0, 1) per element',
                                terms={'swd': 'notebook SWD on (N, 1, 384): 1024 random directions of the 384-d cloud',
                                       'bwd': "per image: its 384 values vs N(0, 1) samples (notebook basis_wasserstein_distance on (384, 1, N), = the notebook's commented-out loss_e)",
                                       'bwd_coord': "monitor only: the notebook's active loss_b, each coordinate across the N images"},
                                removed='c_swd and t_swd (user 2026-11-09: redundant / degenerate on image-level features)',
                                precision='fp32, outside autocast (notebook also calls it outside autocast)') if a.reg == 'wasserstein' else None,
               sigreg=dict(knots=17, directions=256, t_max=3, precision='inside bf16 autocast (official)') if a.reg != 'wasserstein' else None,
               lr_schedule=f'LinearLR 0.01 -> 1 over {spe} steps, then CosineAnnealingLR eta_min=1e-3 (absolute)',
               probe=evals.PROBE | {'features': 'eval mode, test transform, standardized with probe-train mean/std'},
               trunk_hash=th, trainable_params=sum(p.numel() for p in params),
               upstream_commit=git_rev(HERE / 'upstream'), code_commit=git_rev(HERE),
               torch=torch.__version__, image=os.environ.get('IMAGE', ''))
    (run.dir / 'config.json').write_text(json.dumps(cfg, indent=1))
    (run.dir / 'DESCRIPTION.md').write_text((a.desc or f'arm {a.arm}, {reg_name} lam {a.lam:g}, seed {a.seed}, {a.split} split, {a.epochs} epochs') + '\n')
    print(json.dumps({k: cfg[k] for k in ('arm', 'lam', 'seed', 'split', 'steps', 'trunk_hash', 'trainable_params')}), flush=True)

    ck = Path(a.ckpt) / a.name
    step, t_train, prev_drift, init_std = 0, 0.0, {}, {}

    def do_evals(final=False):
        with private_rng(derived_seed('eval', a.seed, step)):
            before = trunk_hash(net) if step == 0 else None
            ftr, etr, _, ytr = evals.extract(net, probe_tr, dev)
            fev, eev, _, yev = evals.extract(net, probe_ev, dev)
            if before is not None and trunk_hash(net) != before:
                run.fail('encoder weights changed during probe evaluation')
            acc = evals.linear_probe(ftr, ytr, fev, yev)
            rec = {'step': step, 'epoch': step / spe, 'test_clean': 1 - acc, 'probe_err': 1 - acc, 'probe_acc': acc, 'train_seconds': t_train}
            rec |= {f'feat_{k}': v for k, v in evals.collapse(fev).items()}
            init_std.setdefault('v', rec['feat_std_mean'])
            if step > 0 and rec['feat_std_mean'] < COLLAPSE_FRAC * init_std['v']:
                run.log(rec)
                run.fail(f"collapsed: per-dimension feature std {rec['feat_std_mean']:.4g} < {COLLAPSE_FRAC} x init {init_std['v']:.4g} at step {step}")
            if etr is not None:
                rec['probe_acc_emb512'] = evals.linear_probe(etr, ytr, eev, yev)
        run.log(rec)
        print(f"eval step {step}: probe {acc:.4f} erank {rec['feat_erank']:.1f} std {rec['feat_std_mean']:.3f}", flush=True)
        return rec

    @torch.inference_mode()
    def do_drift():
        net.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            native, z, _ = net(fixed_x)            # fixed_x: (512, 1, 3, H, W)
        net.train()
        cur = {'native': native[0].float()}
        if a.arm == 'A':
            cur['proj'] = z[0].float()
        rec = {'step': step, 'drift': 1}
        for k, f in cur.items():
            c = evals.collapse(f)
            rec[f'drift_{k}_erank'] = c['erank']; rec[f'drift_{k}_std'] = c['std_mean']
            if k in prev_drift:
                rec |= {f'drift_{k}_{m}': v for m, v in evals.drift(prev_drift[k], f).items()}
            prev_drift[k] = f
        run.log(rec)

    def save(tag):
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'step': step, 'cfg': cfg}, ck / f'{tag}.pt.tmp')
        os.replace(ck / f'{tag}.pt.tmp', ck / f'{tag}.pt')

    do_drift(); do_evals()
    win, t_win, wait_ms, t_last = {}, time.time(), 0.0, time.time()
    torch.cuda.reset_peak_memory_stats()
    done = False
    for epoch in range(a.epochs):
        for vs, _ in loader:
            t0 = time.time(); wait_ms += 1000 * (t0 - t_last)
            vs = vs.to(dev, non_blocking=True)
            if step == 0:
                run.log({'step': 0, 'batch0_hash': int(hashlib.sha256(vs.float().cpu().numpy().tobytes()).hexdigest()[:12], 16)})
            with torch.autocast('cuda', dtype=torch.bfloat16):
                native, z, _ = fwd(vs)
                align = (z.mean(0) - z).square().mean()
                rseed = derived_seed('reg', a.seed, step)
                if a.reg != 'wasserstein':
                    with private_rng(rseed):
                        reg = sigreg(z)
                    terms = {}
            if a.reg == 'wasserstein':
                with private_rng(rseed):
                    reg, terms = wreg(z.float(), square=bool(a.wsquare))
            loss = (1 - a.lam) * align + a.lam * reg
            if not torch.isfinite(loss):
                run.fail(f'non-finite loss at step {step}: align {align.item()} reg {reg.item()}')

            if step == 0:
                first_step_checks(a, run, net, z, align, reg, rseed, sigreg)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if step == 0:
                bb = [p.grad for p in net.backbone.parameters() if p.grad is not None]
                if not bb or not all(torch.isfinite(g).all() for g in bb) or sum(g.abs().sum() for g in bb) == 0:
                    run.fail('encoder gradients missing, zero or non-finite at step 0')
                if a.arm == 'A' and any(p.grad is None or p.grad.abs().sum() == 0 for p in net.proj.parameters()):
                    run.fail('projector gradients missing at step 0')
            scaler.step(opt); scaler.update(); sched.step()
            torch.cuda.synchronize() if step % a.log_every == a.log_every - 1 else None
            step += 1
            t_last = time.time()

            for k, v in {'loss': loss, 'align': align, 'reg': reg, **{f'w_{k}': v for k, v in terms.items()}}.items():
                win[k] = win.get(k, 0.0) + v.detach()
            if step % a.log_every == 0:
                dt = time.time() - t_win; t_train += dt
                rec = {'step': step, 'epoch': step / spe, **{k: (v / a.log_every).item() for k, v in win.items()},
                       'lr': opt.param_groups[0]['lr'], 'sps': a.log_every / dt, 'img_s': a.log_every * a.bs / dt,
                       'data_wait_ms': wait_ms / a.log_every, 'gpu_mem_gib': torch.cuda.max_memory_allocated() / 2**30,
                       'train_seconds': t_train}
                rec['clean'] = rec['loss']        # train-viz hero number
                run.log(rec)
                win, wait_ms, t_win = {}, 0.0, time.time()
            if a.max_steps and step >= a.max_steps:
                done = True
                break
        ep = epoch + 1
        t_pause = time.time()
        if ep % a.drift_every == 0 or done:
            do_drift()
        if ep % a.eval_every == 0 or ep == a.epochs or done:
            do_evals()
        if ep % a.ckpt_every == 0 or ep == a.epochs or done:
            save('last')
        t_win += time.time() - t_pause; t_last = time.time()
        if done:
            break
    save('final')
    (run.dir / 'DONE').write_text(f'{step} steps, {t_train:.0f} s training\n')


def first_step_checks(a, run, net, z, align, reg, rseed, sigreg):
    """Treatment checks at step 0, before the first update. Each failure aborts the run."""
    rec = {'step': 0, 'check': 1}
    # 1a. the regularizer is reproducible under its private RNG (so it cannot touch the data stream)
    # 1b. it reduces over source images within each view: the whole call == mean of single-view calls
    #     (SIGReg: same directions per call; wreg: views are separate calls by construction, checked
    #     by comparing V single-view calls on V copies of view 0 with one call on view 0)
    with torch.no_grad():
        if a.reg == 'wasserstein':
            with private_rng(rseed):
                whole, _ = wreg(z.float(), square=bool(a.wsquare))
            with private_rng(rseed):
                one, _ = wreg(z[:1].float(), square=bool(a.wsquare))
            with private_rng(rseed):
                rep, _ = wreg(z[:1].float().expand(z.shape[0], -1, -1).contiguous(), square=bool(a.wsquare))
            split, pooled_ok = one, True
            rec['reg_view0'], rec['reg_view0_xV'] = float(one), float(rep)
            # outer-square treatment: the unsquared SWD of view 0 squared must equal the squared one
            with private_rng(rseed):
                _, t_sq = wreg(z[:1].float(), square=True)
            with private_rng(rseed):
                _, t_rt = wreg(z[:1].float(), square=False)
            rec['swd_sq'], rec['swd_rt'] = float(t_sq['swd']), float(t_rt['swd'])
            if abs(rec['swd_rt'] ** 2 - rec['swd_sq']) > 1e-3 * rec['swd_sq']:
                run.fail(f"sqrt(swd) check failed: {rec['swd_rt']}^2 != {rec['swd_sq']}")
            used = rec['swd_sq'] if a.wsquare else rec['swd_rt']
            with private_rng(rseed):
                _, t_used = wreg(z[:1].float(), square=bool(a.wsquare))
            if abs(float(t_used['swd']) - used) > 1e-6 * max(used, 1e-12):
                run.fail('training wreg does not use the --wsquare setting')
            rec['wreg_dim'] = z.shape[-1]
            if a.arm == 'A' and z.shape[-1] != a.proj_dim:
                run.fail(f'A + wasserstein must regularize the projector output ({a.proj_dim}-d), got {z.shape[-1]}-d')
            # with fresh priors per view, V copies average to about the single-view value; pooling would not
            pooled_ok = abs(float(rep) - float(one)) < 0.5 * abs(float(one))
        else:
            with private_rng(rseed), torch.autocast('cuda', dtype=torch.bfloat16):
                whole = sigreg(z)
            views = []
            for v in range(z.shape[0]):
                with private_rng(rseed), torch.autocast('cuda', dtype=torch.bfloat16):
                    views.append(sigreg(z[v:v + 1]))
            split, pooled_ok = torch.stack(views).mean(), True
    rec['reg_whole'], rec['reg_split'] = float(whole), float(split)
    r0 = float(reg.detach())
    if abs(float(whole) - r0) > 1e-3 * max(abs(r0), 1e-8):
        run.fail(f'regularizer not reproducible under its private RNG: {float(whole)} vs {r0}')
    if a.reg != 'wasserstein' and abs(float(split) - float(whole)) > 2e-2 * abs(float(whole)):
        run.fail(f'SIGReg is not a mean of per-view statistics: whole {float(whole)} split {float(split)}')
    if not pooled_ok:
        run.fail(f"wreg of V copies of one view {rec['reg_view0_xV']} != one view {rec['reg_view0']}: views look pooled")
    # 2. both loss terms reach the encoder; record their gradient norms
    bb = [p for p in net.backbone.parameters() if p.requires_grad]
    g_al = torch.autograd.grad((1 - a.lam) * align, bb, retain_graph=True, allow_unused=True)
    g_rg = torch.autograd.grad(a.lam * reg, bb, retain_graph=True, allow_unused=True)
    norm = lambda gs: math.sqrt(sum(float(g.float().square().sum()) for g in gs if g is not None))
    rec['g_align'], rec['g_reg'] = norm(g_al), norm(g_rg)
    if a.lam > 0 and not (rec['g_reg'] > 0 and math.isfinite(rec['g_reg'])):
        run.fail(f"regularizer gradient on the encoder is {rec['g_reg']}")
    if a.lam < 1 and not (rec['g_align'] > 0 and math.isfinite(rec['g_align'])):
        run.fail(f"alignment gradient on the encoder is {rec['g_align']}")
    run.log(rec)
    print('step-0 checks passed:', json.dumps(rec), flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
