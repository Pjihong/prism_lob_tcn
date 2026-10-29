#!/usr/bin/env python3
"""Re-render a pinned set of uploaded paper figures; never train or overwrite originals."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

BUNDLE = Path(__file__).resolve().parent

def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def check(p, digest):
    if not p.is_file():
        raise SystemExit(f'Missing input: {p}\nThe PNG-only figures folder does not include original analysis inputs. Supply the files recorded in reference/input_manifest.json; see docs/REPRODUCE.md.')
    if sha256(p) != digest:
        raise SystemExit(f'Input differs from the supplied paper result: {p}')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd())
    ap.add_argument('--only', choices=['all', 'cf', 'factorization', 'cards', 'gallery'], default='all')
    ap.add_argument('--device', default=os.environ.get('DEVICE', 'cpu'))
    ap.add_argument('--output', type=Path, default=Path('figure_revision_output'))
    args = ap.parse_args()
    root = args.root.resolve()
    if not (root / 'lob_xai').is_dir():
        raise SystemExit('Run from the repository root or pass --root /path/to/prism_lob_tcn')
    manifest = json.loads((BUNDLE / 'reference/input_manifest.json').read_text())
    if args.only in ('all', 'cf'):
        for rel, digest in manifest['files'].items():
            check(root / rel, digest)
    missing = [m for m in ['numpy', 'matplotlib'] if importlib.util.find_spec(m) is None]
    if args.only != 'cf':
        missing += [m for m in ['torch', 'pandas', 'scipy', 'sklearn', 'yaml'] if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit('Missing Python modules: ' + ', '.join(missing) + '\nActivate the original training environment. CF-only: python -m lob_xai_viz.render --only cf')
    ck = {}
    if args.only != 'cf':
        keys = ['toxic_seed1', 'toxic_seed2', 'toxic_seed3'] if args.only == 'gallery' else ['toxic_seed1', 'vacuum_seed1']
        if args.only == 'all':
            keys = list(manifest['checkpoints'])
        for key in keys:
            ref = manifest['checkpoints'][key]
            ck[key] = root / ref['path']; check(ck[key], ref['sha256'])
        for name in ['FI2010_train.csv', 'FI2010_test.csv']:
            if not (root / 'data/FI2010' / name).is_file():
                raise SystemExit(f'Missing data/FI2010/{name}')
    out = args.output if args.output.is_absolute() else root / args.output
    out.mkdir(parents=True, exist_ok=True)
    # Each run is isolated. A failed run cannot leave apparently complete old figures.
    from datetime import datetime
    run = out / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.only)
    run.mkdir(exist_ok=False)
    fig = run / 'fig'; fig.mkdir()
    work = run / 'work'; work.mkdir()
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('MPLBACKEND', 'Agg')
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    commands = []
    def execute(script, *opts):
        cmd = [sys.executable, '-m', 'lob_xai_viz.' + Path(script).stem, *map(str, opts)]
        commands.append(cmd)
        print('\n[RUN]', script, flush=True)
        subprocess.run(cmd, cwd=root, env=env, check=True)
    if args.only in ('all', 'cf'):
        staged = work / 'cf_inputs'
        for rel, digest in manifest['files'].items():
            src = root / rel; check(src, digest)
            dst = staged / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        execute('plot_cf_mirror.py', '--models', 'ours', 'deeplob', '--seeds', 1, 2, 3,
                '--carpet-root', staged / 'figures/cf_response_carpet/final/full',
                '--insertion-template', str(staged / 'figures/cf_mirror_src_positional/{model}_{task}_seed{seed}_insertion.curves.npz'),
                '--out-dir', work / 'cf')
        sources = json.loads((work/'cf/cf_mirror.provenance.json').read_text())['sources']
        def signature(rows):
            return sorted((r['direction'], r['task'], r['cls'], r['model'], r['seed'], r['n'], r['checkpoint_sha256']) for r in rows)
        if signature(sources) != signature(manifest['cf_sources']):
            raise SystemExit('CF input/cohort mismatch; export stopped')
    if args.only in ('all', 'factorization', 'cards'):
        for task in ['toxic', 'vacuum']:
            common = ['--task', task, '--target', 1, '--seed', 1, '--rank', 1,
                      '--ckpt', ck[task+'_seed1'], '--device', args.device]
            if args.only in ('all', 'factorization'):
                execute('plot_factorization.py', *common, '--out-dir', work/'factorization')
            if args.only in ('all', 'cards'):
                execute('plot_alert_card.py', *common, '--out-dir', work/'cards')
    if args.only in ('all', 'gallery'):
        execute('gallery_qualitative.py', '--task', 'toxic', '--ckpt-s1', ck['toxic_seed1'],
                '--ckpt-s2', ck['toxic_seed2'], '--ckpt-s3', ck['toxic_seed3'],
                '--model-name', 'Prism-LOB', '--device', args.device, '--out-dir', work/'gallery')
    # Verify key scientific identities against supplied results before exporting.
    checks = []
    for actual in work.rglob('*.provenance.json'):
        ref = BUNDLE / 'reference' / actual.name
        if not ref.is_file():
            continue
        a, b = json.loads(actual.read_text()), json.loads(ref.read_text())
        for key in ['checkpoint_sha256', 'window_index', 'representative_window_index',
                    'cam_span_ticks_1based', 'random_span_ticks_1based', 'pool_kind', 'verdict']:
            if key in b and a.get(key) != b[key]:
                raise SystemExit(f'{actual.name}: {key} changed: {a.get(key)} vs {b[key]}. Export stopped; inspect work files.')
        for key in ['p_original', 'p_cam_deleted', 'p_random_deleted']:
            if key in b:
                err = max(abs(x-y) for x,y in zip(a[key],b[key]))
                if err > 0.005:
                    raise SystemExit(f'{actual.name}: {key} drift {err:.4g}; export stopped')
        checks.append(actual.name)
    gallery_stats = work/'gallery/caption_stats.json'
    if gallery_stats.is_file():
        a=json.loads(gallery_stats.read_text()); b=json.loads((BUNDLE/'reference/caption_stats.json').read_text())
        for seed, classes in b['per_seed'].items():
            for cls, stats in classes.items():
                if a['per_seed'][seed][cls]['rank1_window'] != stats['rank1_window']:
                    raise SystemExit(f'Gallery representative window changed: seed {seed} {cls}')
        checks.append('gallery representative windows')
    for f in work.rglob('*'):
        if f.suffix in ('.png', '.pdf'):
            shutil.copy2(f, fig/f.name)
    shutil.copy2(BUNDLE/'README.md', run/'FIGURE_GUIDE.md')
    notes = run/'notes';notes.mkdir()
    for f in work.rglob('*'):
        if f.suffix in ('.csv','.json') and 'cf_inputs' not in f.parts:
            shutil.copy2(f, notes/f.name)
    (run/'execution.json').write_text(json.dumps({'root':str(root),'device':args.device,'commands':commands,'checks':checks},indent=2))
    with zipfile.ZipFile(run/'overleaf_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(fig.iterdir()): z.write(f, 'fig/'+f.name)
    (out/'LATEST_OUTPUT.txt').write_text(str(run)+'\n')
    print('\nDONE\nUpload files in:',fig,'\nZIP:',run/'overleaf_figures.zip',flush=True)
    print('Original scripts and images were not overwritten. See FIGURE_GUIDE.md and notes/.')

if __name__ == '__main__':
    main()
