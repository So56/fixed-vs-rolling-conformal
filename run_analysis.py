"""Run numbered research scripts in an explicitly prepared, independent workspace."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import shutil


CODE_DIR = Path(__file__).resolve().parent / 'analysis'
MARKER = '_workspace_manifest.json'


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_workspace(input_dir, output_dir):
    """Copy saved inputs/results once; never run analysis or overwrite a destination."""
    source, destination = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not (source / 'docs/research_design_lock.json').is_file():
        raise ValueError('Input must contain the saved research snapshot and design lock.')
    if not (source / 'data/raw/download_manifest.json').is_file():
        raise ValueError('Saved FRED snapshots are required; this command does not download data.')
    if destination.exists():
        raise FileExistsError(f'Output directory must not exist: {destination}')
    if destination == source or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Use a separate output directory outside the input snapshot.')
    if destination == CODE_DIR.parent or CODE_DIR.parent.is_relative_to(destination) or destination.is_relative_to(CODE_DIR.parent):
        raise ValueError('Do not use the public code directory as an analysis workspace.')

    # Copy only research inputs and saved outputs, never Git history or local browser/vendor files.
    folders = ['data', 'docs', 'models', 'reports', 'figures', 'scripts']
    files = [p for name in folders for p in (source / name).rglob('*') if p.is_file()]
    files += list(source.glob('[0-9][0-9]_*.py'))
    files += [source / name for name in ['requirements.txt', 'file_inventory.csv', 'README.md']
              if (source / name).is_file()]
    files = sorted(set(p for p in files if not any(
        part in {'.git', '__pycache__', 'vendor'} or part.startswith('browser_profile')
        for part in p.relative_to(source).parts)))
    # Reject links before writing anything: input and output must be genuinely separate.
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f'Symlink or external input is not supported: {path}')
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = file_hash(path)
        if file_hash(target) != digest:
            raise OSError(f'Copied file differs: {relative}')
        copied[relative.as_posix()] = digest
    manifest = {'format': 1, 'input_dir': str(source), 'workspace': str(destination),
                'copied_files_sha256': copied,
                'policy': 'Only workspace copies are updated. No data retrieval. No Git history copied.'}
    (destination / MARKER).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination


def run_stage(workspace, stage):
    """Run preprocessing or a numbered analysis stage in the prepared workspace."""
    workspace = Path(workspace).resolve()
    marker = workspace / MARKER
    if not marker.is_file():
        raise ValueError('Run the prepare command first; an original research directory cannot be used.')
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    source = Path(manifest['input_dir']).resolve()
    if manifest.get('format') != 1 or Path(manifest['workspace']).resolve() != workspace:
        raise ValueError('Workspace marker does not match this directory.')
    if workspace == source or workspace.is_relative_to(source) or source.is_relative_to(workspace):
        raise ValueError('Workspace and input snapshot must be separate.')
    if not (workspace / 'data/raw/SP500_fred.csv').is_file() or not (workspace / 'data/raw/VIXCLS_fred.csv').is_file():
        raise FileNotFoundError('Saved FRED snapshots are required; no download is attempted.')
    # Cache and temporary files also belong to the copy, not the research original.
    cache = workspace / '.runtime'
    (cache / 'tmp').mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache / 'matplotlib')
    os.environ['TEMP'] = os.environ['TMP'] = str(cache / 'tmp')
    stages = ['prepare'] + [str(number) for number in range(10, 20)] if stage == 'all' else [stage]
    previous_directory = Path.cwd()
    try:
        os.chdir(workspace)
        for selected in stages:
            executed = []
            if selected == 'prepare':
                path = CODE_DIR / 'prepare_data.py'
                print(f'[prepare] {path.name}', flush=True)
                spec = importlib.util.spec_from_file_location('research_preprocessing', path)
                preprocessing = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(preprocessing)
                preprocessing.prepare_data(workspace)
                executed.append({'code': str(path), 'sha256': file_hash(path)})
            else:
                path = next(CODE_DIR.glob(f'{int(selected):02d}_*.py'))
                print(f'[{selected}] {path.name}', flush=True)
                # Match separate original script executions: plot styles must not leak between stages.
                import matplotlib
                with matplotlib.rc_context():
                    matplotlib.rcdefaults()
                    runpy.run_path(str(path), init_globals={'ANALYSIS_WORKSPACE': str(workspace)}, run_name='__main__')
                executed.append({'code': str(path), 'sha256': file_hash(path)})
            # Record the code actually executed, separately from archived code in the snapshot.
            log_path = workspace / '_execution_log.json'
            records = json.loads(log_path.read_text(encoding='utf-8')) if log_path.exists() else []
            records.append({'stage': selected, 'workspace': str(workspace), 'executed_code': executed})
            log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        os.chdir(previous_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Copy a saved snapshot into a new workspace; no analysis.')
    prepare.add_argument('--input-dir', type=Path, required=True)
    prepare.add_argument('--output-dir', type=Path, required=True)
    run = commands.add_parser('run', help='Analyze workspace copies only.')
    run.add_argument('--workspace', type=Path, required=True)
    run.add_argument('--stage', choices=['prepare', 'all'] + [str(n) for n in range(10, 20)], required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare_workspace(args.input_dir, args.output_dir))
    else:
        run_stage(args.workspace, args.stage)


if __name__ == '__main__':
    main()
