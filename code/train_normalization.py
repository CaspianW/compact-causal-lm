"""Fresh CUDA training with controlled normalization variants and validation calibration."""
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

IMPLEMENTATION = 'student'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['control', 'qknorm', 'normformer'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=9600)
    parser.add_argument('--eval-every', type=int, default=2400)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        parser.error('Steps and evaluation interval must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Use an empty output directory.')
    process_started = time.perf_counter()
    device, precision = setup('cuda', 'bf16', 4)
    torch.manual_seed(17)
    config = json.loads((CODE_ROOT / 'configs/rope_swiglu_w256_d5_attndropout01.json').read_text())
    config['normalization_variant'] = args.variant
    model, implementation_sha = make_model(IMPLEMENTATION, config, device)
    data = load_data()
    tokens = data['train'][0].to(device)
    generator = torch.Generator().manual_seed(17)
    starts = torch.randint(len(tokens) - 257, (args.steps, 32), generator=generator).to(device)
    offsets = torch.arange(257, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1, fused=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    history, validation_history = [], []
    evaluation_seconds = 0.
    best = {'bpb': float('inf'), 'step': None}
    metadata = {'protocol': PROTOCOL, 'implementation': IMPLEMENTATION, 'config': config,
                'variant': args.variant, 'seed': 17, 'batch_size': 32, 'steps': args.steps,
                'train_tokens': args.steps * 32 * 256, 'precision': precision,
                'validation_precision': 'fp32', 'optimizer': 'adamw', 'fused': True,
                'initialization': 'random; common parameters identical across variants',
                'learning_rate': .001, 'weight_decay': .1, 'warmup_steps': 100,
                'cosine_lr_floor': .1, 'gradient_clip': 1.,
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainer_sha256': sha(__file__), 'implementation_sha256': implementation_sha,
                'student_sha256': sha(CODE_ROOT / 'student.py'),
                'experimental_models_sha256': sha(CODE_ROOT / 'experimental_models.py'), 'torch_version': str(torch.__version__)}
    (args.run_dir / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    def save_checkpoint(path, step, temperature=1.):
        torch.save({'protocol': PROTOCOL, 'implementation': IMPLEMENTATION,
                    'config': {**config, 'temperature': temperature},
                    'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    'seed': 17, 'train_tokens': step * 32 * 256, 'step': step,
                    'trainer_sha256': metadata['trainer_sha256']}, path)

    def write_metrics(step, complete=False):
        result = {**metadata, 'completed_steps': step, 'completed': complete,
                  'history': history, 'validation_history': validation_history, 'best': best,
                  'train_seconds': time.perf_counter() - started - evaluation_seconds,
                  'evaluation_and_saving_seconds': evaluation_seconds,
                  'process_seconds': time.perf_counter() - process_started, **device_metrics(device)}
        (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    print(json.dumps({'starting': args.variant, 'parameters': metadata['parameters']}), flush=True)
    for step in range(args.steps):
        batch = tokens[starts[step, :, None] + offsets]
        lr = .001 * min(1., (step + 1) / 100) * (
            .1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            if not torch.isfinite(loss.detach()) or not torch.isfinite(gradient_norm):
                raise RuntimeError('Non-finite loss or gradient norm.')
            row = {'step': step + 1, 'loss': loss.item(), 'learning_rate': lr,
                   'train_seconds': time.perf_counter() - started - evaluation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluated = time.perf_counter()
            val = score(model, *data['validation'], device, 'fp32')
            val.pop('window_nll_nats')
            validation_history.append({'step': step + 1, **val})
            if val['bpb'] < best['bpb']:
                path = args.run_dir / 'checkpoint-best.pt'
                save_checkpoint(path, step + 1)
                best = {'step': step + 1, 'bpb': val['bpb'], 'checkpoint_sha256': sha(path)}
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
    del optimizer, loss, batch
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
    save_checkpoint(calibrated_path, best['step'], selected['temperature'])
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
    print(json.dumps({'completed': True, 'variant': args.variant, 'raw_best': best,
                      'calibrated': selected, 'process_seconds': result['process_seconds']}), flush=True)


if __name__ == '__main__':
    main()
