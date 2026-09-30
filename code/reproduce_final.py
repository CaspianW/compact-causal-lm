"""Replay a recorded train-only recipe; use the supplied checkpoint for direct scoring."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

CODE_ROOT = Path(__file__).resolve().parent
SCRIPTS = {'train_capacity.py', 'train_continuation.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding='utf-8'))
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Use an empty run directory.')
    for phase in recipe['phases']:
        if phase['script'] not in SCRIPTS:
            parser.error('Unknown training phase.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / 'base-config.json'
    config_path.write_text(json.dumps(recipe['config'], indent=2) + '\n')
    previous = None
    for index, phase in enumerate(recipe['phases']):
        output = args.run_dir / f'phase-{index + 1:02d}'
        command = [sys.executable, '-u', str(Path(__file__).resolve().parent / phase['script']),
                   '--run-dir', str(output)]
        if phase['script'] == 'train_capacity.py':
            command.extend(['--config', str(config_path), '--alpha', str(phase['alpha']),
                            '--steps', str(phase['steps']), '--eval-every', str(phase['eval_every'])])
        elif phase['script'] == 'train_continuation.py':
            if previous is None:
                parser.error('Continuation requires a preceding training phase.')
            command.extend(['--checkpoint', str(previous / 'checkpoint-calibrated.pt'),
                            '--alpha', str(phase['alpha']), '--schedule', phase['schedule'],
                            '--learning-rate', str(phase['learning_rate']),
                            '--steps', str(phase['steps']), '--eval-every', str(phase['eval_every']),
                            '--snapshot-every', str(phase.get('snapshot_every', 0)),
                            '--snapshot-start', str(phase.get('snapshot_start', 2400))])
        else:
            parser.error('Unknown final training phase.')
        subprocess.run(command, cwd=CODE_ROOT, check=True)
        previous = output
    if previous is None:
        parser.error('The recipe contains no phases.')
    shutil.copyfile(previous / 'checkpoint-calibrated.pt', args.run_dir / 'checkpoint-calibrated.pt')
    (args.run_dir / 'training_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')


if __name__ == '__main__':
    main()
