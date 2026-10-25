"""Small public tests; no research data, model fitting, or Bootstrap required."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

PUBLIC = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_imports():
    """Audit hooks apply only in this child process and cannot affect other tests."""
    def guard(event, arguments):
        if event == 'open':
            mode, flags = arguments[1:3]
            if (mode and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise AssertionError('IMPORT_WRITE')
        if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.system',
                     'subprocess.Popen', 'socket.connect', 'socket.getaddrinfo', 'urllib.Request'}:
            raise AssertionError('IMPORT_SIDE_EFFECT: ' + event)

    sys.dont_write_bytecode = True
    sys.addaudithook(guard)
    paths = [PUBLIC / 'run_analysis.py'] + sorted((PUBLIC / 'analysis').glob('*.py'))
    for index, path in enumerate(paths):
        load_module(path, 'public_import_' + str(index))
    print(json.dumps({'imported_modules': len(paths), 'side_effects': 0}))


class EntryPointTests(unittest.TestCase):
    def test_imports_do_not_analyze_write_or_connect(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), '--probe'],
                                    cwd=directory, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {'imported_modules': 12, 'side_effects': 0})
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prepare_refuses_existing_or_overlapping_destination(self):
        cli = load_module(PUBLIC / 'run_analysis.py', 'public_cli_protection')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'saved'
            (source / 'docs').mkdir(parents=True)
            (source / 'data/raw').mkdir(parents=True)
            for relative in ['docs/research_design_lock.json', 'data/raw/download_manifest.json']:
                (source / relative).write_text('{}', encoding='utf-8')
            existing = Path(directory) / 'existing'
            existing.mkdir()
            sentinel = existing / 'keep.txt'
            sentinel.write_bytes(b'unchanged')
            with self.assertRaisesRegex(FileExistsError, 'must not exist'):
                cli.prepare_workspace(source, existing)
            with self.assertRaisesRegex(ValueError, 'separate output directory'):
                cli.prepare_workspace(source, source / 'nested')
            self.assertEqual(sentinel.read_bytes(), b'unchanged')
            self.assertFalse((source / 'nested').exists())

    def test_run_rejects_original_without_marker(self):
        cli = load_module(PUBLIC / 'run_analysis.py', 'public_cli_no_original')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'prepare command first'):
                cli.run_stage(directory, '19')
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    if sys.argv[1:] == ['--probe']:
        probe_imports()
    else:
        unittest.main(verbosity=2)
