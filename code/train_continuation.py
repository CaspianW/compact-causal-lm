"""Controlled CUDA warm restarts from one checkpoint, retaining its R-Drop coefficient."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))
import torch
from torch.nn import functional as F
from common import PROTOCOL, autocast, device_metrics, load_data, make_model, setup, sha, windows
from evaluate import score

def rdrop_loss(first, second, targets, alpha):
    logp = F.log_softmax(first.float(), dim=-1)
    logq = F.log_softmax(second.float(), dim=-1)
    ce = .5 * (F.nll_loss(logp.flatten(0, 1), targets.flatten()) +
               F.nll_loss(logq.flatten(0, 1), targets.flatten()))
    symmetric_kl = .5 * ((logp.exp() - logq.exp()) * (logp - logq)).sum(-1).mean()
    return ce + alpha * symmetric_kl, ce.detach(), symmetric_kl.detach()

IMPLEMENTATION = 'student'


def learning_rate_at(step, steps, peak, schedule):
    if schedule == 'constant':
        return peak
    if schedule == 'cosine-zero':
        return peak * .5 * (1 + math.cos(math.pi * step / max(1, steps - 1)))
    raise ValueError('Unknown continuation schedule.')


def sample_starts(length, previous_targets, steps, batch_size=32):
    if previous_targets < 0 or previous_targets % 256:
        raise ValueError('Ancestor target count must be a nonnegative multiple of 256.')
    generator = torch.Generator().manual_seed(17)
    torch.randint(length - 257, (previous_targets // 256,), generator=generator)
    return torch.randint(length - 257, (steps, batch_size), generator=generator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--schedule', choices=['constant', 'cosine-zero'], required=True)
    parser.add_argument('--learning-rate', type=float, default=.0001)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4800)
    parser.add_argument('--eval-every', type=int, default=1200)
    parser.add_argument('--snapshot-every', type=int, default=0)
    parser.add_argument('--snapshot-start', type=int, default=2400)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error('R-Drop coefficient must be finite and nonnegative.')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Learning rate must be finite and positive.')
    if args.steps < 1 or args.eval_every < 1:
        parser.error('Steps and evaluation interval must be positive.')
    if args.snapshot_every < 0 or args.snapshot_start < 0:
        parser.error('Snapshot interval and start must be nonnegative.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Use an empty output directory.')
    process_started = time.perf_counter()
    device, precision = setup('cuda', 'bf16', 4)
    ancestor = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ancestor['protocol'] != PROTOCOL or ancestor['seed'] != 17:
        parser.error('Expected a seed-17 checkpoint with the unchanged coursework protocol.')
    config = dict(ancestor['config'])
    config.pop('temperature', None)
    torch.manual_seed(17)
    model, implementation_sha = make_model(IMPLEMENTATION, config, device)
    model.load_state_dict(ancestor['model'])
    data = load_data()
    tokens = data['train'][0].to(device)
    previous_targets = int(ancestor['train_tokens'])
    previous_steps = previous_targets // (32 * 256)
    if previous_targets % (32 * 256):
        parser.error('Expected an ancestor trained with batches of 32 sequences.')
    starts = sample_starts(len(tokens), previous_targets, args.steps).to(device)
    offsets = torch.arange(257, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.1, fused=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    initial = score(model, *data['validation'], device, 'fp32')
    initial.pop('window_nll_nats')
    history, validation_history = [], [{'step': previous_steps, 'additional_step': 0, **initial}]
    evaluation_seconds = 0.
    best = {'bpb': initial['bpb'], 'step': previous_steps, 'additional_step': 0}
    metadata = {'protocol': PROTOCOL, 'implementation': IMPLEMENTATION, 'config': config,
                'variant': config.get('normalization_variant', 'control'),
                'training_mode': 'warm_restart', 'rdrop_alpha': args.alpha, 'seed': 17,
                'snapshot_every': args.snapshot_every, 'snapshot_start': args.snapshot_start,
                'batch_size': 32, 'steps': args.steps, 'completed_steps': 0,
                'train_tokens': previous_targets + args.steps * 32 * 256,
                'forward_train_targets': 2 * (previous_targets + args.steps * 32 * 256),
                'ancestor_train_tokens': previous_targets, 'ancestor_steps': previous_steps,
                'additional_sampled_targets': args.steps * 32 * 256,
                'additional_forward_targets': 2 * args.steps * 32 * 256,
                'ancestor': str(args.checkpoint.resolve()), 'ancestor_sha256': sha(args.checkpoint),
                'initial_validation': initial, 'optimizer_state_restored': False,
                'dropout_rng_reset_seed': 17, 'same_sampled_windows_for_all_candidates': True,
                'precision': precision, 'validation_precision': 'fp32',
                'optimizer': 'adamw', 'fused': True, 'initialization': 'same ancestor weights',
                'learning_rate': args.learning_rate, 'schedule': args.schedule, 'warmup_steps': 0,
                'cosine_lr_floor': 0. if args.schedule == 'cosine-zero' else 1.,
                'weight_decay': .1, 'gradient_clip': 1.,
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainer_sha256': sha(__file__), 'implementation_sha256': implementation_sha,
                'student_sha256': sha(CODE_ROOT / 'student.py'),
                'experimental_models_sha256': sha(CODE_ROOT / 'experimental_models.py'), 'torch_version': str(torch.__version__),
                'sources': ['https://arxiv.org/abs/2106.14448', 'https://github.com/dropreg/R-Drop']}
    (args.run_dir / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def save_checkpoint(path, additional_step, temperature=1.):
        torch.save({'protocol': PROTOCOL, 'implementation': IMPLEMENTATION,
                    'config': {**config, 'temperature': temperature},
                    'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'seed': 17, 'train_tokens': previous_targets + additional_step * 32 * 256,
                    'step': previous_steps + additional_step,
                    'followup': {'ancestor_sha256': metadata['ancestor_sha256'],
                                 'additional_steps': additional_step, 'schedule': args.schedule,
                                 'learning_rate': args.learning_rate, 'rdrop_alpha': args.alpha,
                                 'optimizer_state_restored': False},
                    'trainer_sha256': metadata['trainer_sha256']}, path)

    save_checkpoint(args.run_dir / 'checkpoint-best.pt', 0)
    best['checkpoint_sha256'] = sha(args.run_dir / 'checkpoint-best.pt')
    # Validation uses no dropout. Reset its stream identically for all three branches.
    torch.manual_seed(17)
    model.train()
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    def write_metrics(step, complete=False):
        result = {**metadata, 'completed_steps': step, 'completed': complete,
                  'history': history, 'validation_history': validation_history, 'best': best,
                  'train_seconds': time.perf_counter() - started - evaluation_seconds,
                  'evaluation_and_saving_seconds': evaluation_seconds,
                  'process_seconds': time.perf_counter() - process_started, **device_metrics(device)}
        (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    print(json.dumps({'starting': args.schedule, 'learning_rate': args.learning_rate,
                      'initial_validation_bpb': initial['bpb'], 'ancestor_steps': previous_steps}), flush=True)
    for step in range(args.steps):
        batch = tokens[starts[step, :, None] + offsets]
        lr = learning_rate_at(step, args.steps, args.learning_rate, args.schedule)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            first, second = model(batch[:, :-1]), model(batch[:, :-1])
            loss, ce, kl = rdrop_loss(first, second, batch[:, 1:], args.alpha)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (args.snapshot_every and step + 1 >= args.snapshot_start
                and (step + 1) % args.snapshot_every == 0):
            saved = time.perf_counter()
            save_checkpoint(args.run_dir / f'snapshot-{step + 1:05d}.pt', step + 1)
            torch.cuda.synchronize(device)
            evaluation_seconds += time.perf_counter() - saved
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            if not torch.isfinite(loss.detach()) or not torch.isfinite(gradient_norm):
                raise RuntimeError('Non-finite loss or gradient norm.')
            row = {'additional_step': step + 1, 'step': previous_steps + step + 1,
                   'loss': loss.item(), 'ce': ce.item(), 'symmetric_kl': kl.item(),
                   'learning_rate': lr, 'train_seconds': time.perf_counter() - started - evaluation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluated = time.perf_counter()
            val = score(model, *data['validation'], device, 'fp32')
            val.pop('window_nll_nats')
            validation_history.append({'step': previous_steps + step + 1, 'additional_step': step + 1, **val})
            if val['bpb'] < best['bpb']:
                path = args.run_dir / 'checkpoint-best.pt'
                save_checkpoint(path, step + 1)
                best = {'step': previous_steps + step + 1, 'additional_step': step + 1,
                        'bpb': val['bpb'], 'checkpoint_sha256': sha(path)}
            torch.cuda.synchronize(device)
            evaluation_seconds += time.perf_counter() - evaluated
            write_metrics(step + 1)
            print(json.dumps({'validation': validation_history[-1], 'best': best}), flush=True)
    save_checkpoint(args.run_dir / 'checkpoint-final.pt', args.steps)
    torch.save({'optimizer': optimizer.state_dict(), 'cuda_rng_state': torch.cuda.get_rng_state(device),
                'cpu_rng_state': torch.get_rng_state(), 'completed_steps': args.steps,
                'checkpoint_sha256': sha(args.run_dir / 'checkpoint-final.pt')},
               args.run_dir / 'training-state.pt')
    training_result = write_metrics(args.steps, complete=True)
    del optimizer, loss, batch, first, second, ce, kl
    checkpoint = torch.load(args.run_dir / 'checkpoint-best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    temperatures = [.8, .85, .9, .925, .95, .975, 1., 1.025, 1.05, 1.075, 1.1, 1.15, 1.2]
    totals = torch.zeros(len(temperatures), device=device, dtype=torch.float64)
    val_tokens, byte_count = data['validation']
    with torch.inference_mode():
        for x, y in windows(val_tokens):
            x, y = x.to(device), y.to(device)
            logits = model(x).float()
            for index, temperature in enumerate(temperatures):
                logp = F.log_softmax(logits / temperature, dim=-1)
                losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                totals[index] += losses.masked_fill(y == -100, 0).double().sum()
    rows = [{'temperature': t, 'bpb': nll / math.log(2) / byte_count}
            for t, nll in zip(temperatures, totals.cpu().tolist())]
    selected = min(rows, key=lambda row: row['bpb'])
    calibrated_path = args.run_dir / 'checkpoint-calibrated.pt'
    save_checkpoint(calibrated_path, best['additional_step'], selected['temperature'])
    exported = torch.load(calibrated_path, map_location='cpu', weights_only=True)
    reloaded, _ = make_model(IMPLEMENTATION, exported['config'], device)
    reloaded.load_state_dict(exported['model'])
    verified = score(reloaded, *data['validation'], device, 'fp32')
    verified.pop('window_nll_nats')
    if abs(verified['bpb'] - selected['bpb']) > 1e-9:
        raise RuntimeError('Calibrated export failed score verification.')
    calibration = {'rows': rows, 'selected': selected, 'verified': verified,
                   'checkpoint_sha256': sha(calibrated_path)}
    (args.run_dir / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    x = torch.randint(0, 2048, (2, 64), device=device)
    changed = x.clone(); changed[:, 32:] = (changed[:, 32:] + 19) % 2048
    reloaded.eval()
    with torch.inference_mode():
        a, b = reloaded.predict_log_probs(x), reloaded.predict_log_probs(changed)
        assert a.shape == (2, 64, 2048) and torch.isfinite(a).all()
        torch.testing.assert_close(a[:, :32], b[:, :32], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a.logsumexp(-1), torch.zeros(2, 64, device=device), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a[:1], reloaded.predict_log_probs(x[:1]), atol=2e-5, rtol=1e-5)
        reloaded.predict_log_probs((x + 31) % 2048)
        torch.testing.assert_close(a, reloaded.predict_log_probs(x), atol=1e-6, rtol=1e-6)
        assert reloaded.head.weight is reloaded.token.weight
    result = {**training_result, 'calibration': calibration, 'inference_checks_passed': True,
              'process_seconds': time.perf_counter() - process_started}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'completed': True, 'raw_best': best, 'calibrated': selected}), flush=True)


if __name__ == '__main__':
    main()
