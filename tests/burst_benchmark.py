#!/usr/bin/env python3
"""Measure the application after its production burst emitter finishes spawning."""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('build/bin/verlet_cuda'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--particles', type=int, nargs='+', default=[2000000, 3000000, 4000000])
    parser.add_argument('--warmup', type=int, default=120)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--presentation', choices=['visible', 'offscreen'], default='visible')
    parser.add_argument('--shuffle-storage', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--packing', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1 or any(n < 1 for n in args.particles):
        parser.error('particle counts, warmup and samples must be positive')
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for count in args.particles:
        name = str(count)
        result_path = output / f'{name}.csv'
        result_path.unlink(missing_ok=True)
        config = {
            'version': 1, 'presentation': args.presentation,
            'clock': {'mode': 'fixed', 'step_seconds': 1 / 60},
            'exit': {'frame': args.warmup + args.samples + 1},
            'application': {'burst_benchmark': {
                'particles': count, 'warmup': args.warmup, 'samples': args.samples,
                'shuffle_storage': args.shuffle_storage, 'packing': args.packing,
                'output': str(result_path),
            }},
        }
        if args.presentation == 'offscreen':
            config['framebuffer_size'] = [1920, 1080]
            config['captures'] = [{'frame': args.warmup + args.samples + 1,
                                   'path': str(output / f'{name}.ppm'), 'include_ui': False}]
        config_path = output / f'{name}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        with (output / f'{name}.log').open('w') as log:
            subprocess.run([str(binary), '--klvk-diagnostics', str(config_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        if not result_path.exists():
            raise RuntimeError(f'Benchmark produced no results; inspect {output / (name + ".log")}')
        with result_path.open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise RuntimeError(f'Incomplete benchmark; inspect {output / (name + ".log")}')
        row = rows[0]
        assert int(row['particles']) == count and int(row['frames']) == args.samples
        result = {'particles': count, 'mean_ms': float(row['total_ms']) / args.samples,
                  'presentation': args.presentation, 'shuffle_storage': args.shuffle_storage,
                  'packing': args.packing, 'warmup': args.warmup, 'samples': args.samples}
        result['framebuffer_width'] = int(row['framebuffer_width'])
        result['framebuffer_height'] = int(row['framebuffer_height'])
        results.append(result)
        (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f"{count:,}: {result['mean_ms']:.3f} ms ({args.presentation})", flush=True)


if __name__ == '__main__':
    main()
