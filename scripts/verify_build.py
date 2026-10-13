"""Verify saved evidence and the static release without rerunning benchmarks."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    sources = {p.name: digest(p) for p in sorted((ROOT / 'posterior_lab').glob('*.py'))}
    lock = json.loads((ROOT / 'scripts/runtime-lock.json').read_text())
    for asset in lock['assets']:
        for base in ['public', 'dist']:
            path = ROOT / base / 'runtime' / asset['name']
            assert path.stat().st_size == asset['bytes'], str(path)
            assert digest(path) == asset['sha256'], str(path)
    checks.append({'check': 'pinned runtime assets in source and production', 'assets': len(lock['assets'])})

    archive = ROOT / 'public/engine/posterior_lab.zip'
    before = digest(archive)
    subprocess.run(['node', 'scripts/package-engine.mjs'], cwd=ROOT, check=True, capture_output=True)
    assert digest(archive) == before == digest(ROOT / 'dist/engine/posterior_lab.zip')
    with zipfile.ZipFile(archive) as package:
        assert set(package.namelist()) == {'posterior_lab/' + name for name in sources}
        for name in sources:
            assert package.read('posterior_lab/' + name) == (ROOT / 'posterior_lab' / name).read_bytes()
    checks.append({'check': 'reproducible Python archive matches current source and production', 'sha256': before})

    native = json.loads((ROOT / 'reports/native-evaluation.json').read_text())
    assert len(native['cases']) == 12
    for case in native['cases']:
        assert case['source_hashes'] == sources
        assert digest(ROOT / case['raw_artifact']) == case['raw_sha256']
        assert case['status'] == 'complete'
    checks.append({'check': 'native evidence source identity and raw checkpoints', 'cases': len(native['cases'])})

    samplers = json.loads((ROOT / 'reports/sampler-validation.json').read_text())
    assert samplers['all_required_cases_passed']
    for name, value in samplers['environment']['source_sha256'].items():
        assert sources[name] == value
    checks.append({'check': 'quality-gated sampler evidence', 'cases': len(samplers['cases'])})

    browser = json.loads((ROOT / 'reports/browser/report.json').read_text())
    assert browser['summary']['failed'] == 0
    checks.append({'check': 'production HTTP browser evidence', **browser['summary']})
    reference = json.loads((ROOT / 'reports/reference-inference.json').read_text())
    # An observed statistical failure remains visible; never convert it to a pass.
    checks.append({'check': 'independent inference comparison outcome', 'status': reference['status']})
    files = sorted(p for p in (ROOT / 'dist').rglob('*') if p.is_file())
    report = {'status': 'passed', 'checks': checks, 'production_files': len(files),
              'production_bytes': sum(p.stat().st_size for p in files),
              'source_sha256': sources,
              'production_sha256': {str(p.relative_to(ROOT / 'dist')): digest(p) for p in files},
              'limits': ['Saved evidence audit, not a new numerical experiment.',
                         'This local audit does not exercise hosted Pages or remote CI; see deployment evidence separately.',
                         'Browser execution verified in Chromium on macOS; other browsers unverified.']}
    (ROOT / 'reports/build-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks),
                      'production_files': len(files), 'production_bytes': report['production_bytes']}))


if __name__ == '__main__':
    main()
