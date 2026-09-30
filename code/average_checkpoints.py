"""Average late weights from one training trajectory, then calibrate on validation only."""
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
from common import PROTOCOL, load_data, make_model, setup, sha, windows
from evaluate import score

IMPLEMENTATION = 'student'


def mean_weights(checkpoints):
    keys = checkpoints[0]['model'].keys()
    if any(c['model'].keys() != keys for c in checkpoints):
        raise ValueError('Checkpoint state dictionaries have different keys.')
    output = {}
    for key in keys:
        values = [c['model'][key] for c in checkpoints]
        if any(v.shape != values[0].shape or v.dtype != values[0].dtype for v in values):
            raise ValueError('Checkpoint tensors have different shapes or dtypes.')
        if values[0].is_floating_point():
            output[key] = torch.stack([v.double() for v in values]).mean(0).to(values[0].dtype)
        else:
            if any(not torch.equal(v, values[0]) for v in values[1:]):
                raise ValueError('Cannot average unequal non-floating buffers.')
            output[key] = values[0].clone()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-dir', required=True, type=Path)
    parser.add_argument('--count', required=True, type=int)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.count < 2:
        parser.error('Average at least two snapshots.')
    paths = sorted(args.source_run_dir.glob('snapshot-*.pt'))[-args.count:]
    if len(paths) != args.count:
        parser.error('Not enough snapshots in the source trajectory.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Use an empty output directory.')
    started = time.perf_counter()
    parent = json.loads((args.source_run_dir / 'metrics.json').read_text(encoding='utf-8'))
    if not parent.get('inference_checks_passed'):
        parser.error('The source run has not completed its inference checks.')
    checkpoints = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    reference = checkpoints[-1]
    if any(c['protocol'] != PROTOCOL or c['config'] != reference['config'] or c['seed'] != reference['seed']
           or c['followup'] != {**reference['followup'], 'additional_steps': c['followup']['additional_steps']}
           for c in checkpoints):
        parser.error('Snapshots must come from one configuration and continuation trajectory.')
    average = mean_weights(checkpoints)
    config = dict(reference['config']); config['temperature'] = 1.
    device, _ = setup('cuda', 'fp32', 4)
    model, _ = make_model(IMPLEMENTATION, config, device)
    model.load_state_dict(average); model.eval()
    data = load_data()
    raw = score(model, *data['validation'], device, 'fp32'); raw.pop('window_nll_nats')
    temperatures = [.8, .85, .9, .925, .95, .975, 1., 1.025, 1.05, 1.075, 1.1, 1.15, 1.2]
    totals = torch.zeros(len(temperatures), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for x, y in windows(data['validation'][0]):
            x, y = x.to(device), y.to(device)
            logits = model(x).float()
            for index, temperature in enumerate(temperatures):
                logp = F.log_softmax(logits / temperature, dim=-1)
                losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                totals[index] += losses.masked_fill(y == -100, 0).double().sum()
    rows = [{'temperature': t, 'bpb': nll / math.log(2) / data['validation'][1]}
            for t, nll in zip(temperatures, totals.cpu().tolist())]
    selected = min(rows, key=lambda r: r['bpb'])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    exported = {**reference, 'model': average, 'implementation': IMPLEMENTATION,
                'config': {**config, 'temperature': selected['temperature']},
                'weight_average': {'count': args.count, 'source_steps': [c['step'] for c in checkpoints],
                                   'source_sha256': [sha(p) for p in paths]},
                'averaging_script_sha256': sha(__file__)}
    path = args.run_dir / 'checkpoint-calibrated.pt'
    torch.save(exported, path)
    reloaded, _ = make_model(IMPLEMENTATION, exported['config'], device)
    reloaded.load_state_dict(exported['model']); reloaded.eval()
    verified = score(reloaded, *data['validation'], device, 'fp32'); verified.pop('window_nll_nats')
    if abs(verified['bpb'] - selected['bpb']) > 1e-9:
        raise RuntimeError('Averaged checkpoint does not reproduce its validation score.')
    x = torch.randint(0, 2048, (2, 64), device=device)
    changed = x.clone(); changed[:, 32:] = (changed[:, 32:] + 19) % 2048
    with torch.inference_mode():
        a, b = reloaded.predict_log_probs(x), reloaded.predict_log_probs(changed)
        assert a.shape == (2, 64, 2048) and torch.isfinite(a).all()
        torch.testing.assert_close(a[:, :32], b[:, :32], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a.logsumexp(-1), torch.zeros(2, 64, device=device), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a[:1], reloaded.predict_log_probs(x[:1]), atol=2e-5, rtol=1e-5)
        reloaded.predict_log_probs((x + 31) % 2048)
        torch.testing.assert_close(a, reloaded.predict_log_probs(x), atol=1e-6, rtol=1e-6)
        assert reloaded.head.weight is reloaded.token.weight
    result = {**parent, 'weight_average': exported['weight_average'],
              'averaging_script_sha256': sha(__file__), 'additional_training_updates': 0,
              'best': {'step': reference['step'], 'additional_step': reference['followup']['additional_steps'],
                       'bpb': raw['bpb']},
              'calibration': {'rows': rows, 'selected': selected, 'verified': verified,
                              'checkpoint_sha256': sha(path)},
              'averaging_seconds': time.perf_counter() - started, 'inference_checks_passed': True}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'completed': True, 'count': args.count, 'raw': raw['bpb'],
                      'calibrated': selected}), flush=True)


if __name__ == '__main__':
    main()
