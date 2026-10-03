#!/usr/bin/env python3
"""Plot recorded RL losses, without smoothing or inventing missing steps.

Usage: uv run --with matplotlib python -m rl.plot_metrics
Only reads existing metrics; does not start training or contact model services.
"""
import argparse
import json
from pathlib import Path


def load_metric_rows(root, name):
    """Read complete live JSONL records, falling back to historical JSON exports."""
    journal = root / f'{name}.jsonl'
    if journal.exists():
        rows = []
        with journal.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    break
                rows.append(json.loads(line))
        return rows
    path = root / f'{name}.json'
    return json.loads(path.read_text()) if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, default=[
        Path('output/rl_runs/ppo_qat_v4'), Path('output/rl_runs/grpo_qat_v4')])
    parser.add_argument('--output', type=Path, default=Path('output/rl_runs/loss_curves.png'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#f8fafc'})
    fig, axes = plt.subplots(1, len(args.runs), figsize=(6 * len(args.runs), 4.4), squeeze=False)
    for index, (ax, root) in enumerate(zip(axes[0], args.runs)):
        metrics = load_metric_rows(root, 'metrics')
        optimizer_rows = load_metric_rows(root, 'optimizer_metrics')
        rows = optimizer_rows or [m for m in metrics if 'update' in m and 'loss' in m]
        step_key = 'optimizer_step' if optimizer_rows else 'update'
        if not rows:
            raise ValueError(f'no recorded loss updates: {root}')
        config = json.loads((root / 'config.json').read_text())
        x, y = [m[step_key] for m in rows], [m['loss'] for m in rows]
        color = ['#2563eb', '#d97706'][index % 2]
        ax.plot(x, y, '-o', color=color, linewidth=2.2, markersize=8)
        for step, loss in zip(x, y):
            ax.annotate(f'{loss:.6f}', (step, loss), xytext=(0, 12),
                        textcoords='offset points', ha='center', color=color, weight='bold')
        ax.set_title(f"{config['algorithm'].upper()} + {config['tuning'].upper()}", loc='left', weight='bold')
        ax.set_xlabel('Optimizer step' if optimizer_rows else 'Policy update (legacy aggregate)')
        ax.set_ylabel('Weighted objective loss' if optimizer_rows else 'Legacy logged loss')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(x)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.margins(x=.3, y=.35)
        ax.grid(axis='y', alpha=.2)
        skipped = sum('skipped' in m for m in metrics)
        ax.text(.02, .04, f'{len(rows)} recorded points | {skipped} skipped groups',
                transform=ax.transAxes, fontsize=9, color='#64748b')
    fig.suptitle('Agent RL training loss | Qwen3-4B | Apple Metal', fontsize=15, weight='bold')
    fig.text(.5, .025, 'Recorded losses; independent y-scales. No smoothing. Short runs do not establish convergence.',
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .07, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix('.svg'))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
