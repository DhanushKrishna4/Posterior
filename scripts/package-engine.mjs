import { execFileSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
await mkdir('public/engine', {recursive:true});
// Sorted members, stable timestamps and permissions make the package reproducible.
execFileSync(process.env.PYTHON || 'python3', ['-c', `
from pathlib import Path
import zipfile
root = Path('posterior_lab')
with zipfile.ZipFile('public/engine/posterior_lab.zip', 'w', compression=zipfile.ZIP_DEFLATED) as out:
    for path in sorted(root.rglob('*.py')):
        info = zipfile.ZipInfo(str(path), date_time=(2020,1,1,0,0,0))
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        out.writestr(info, path.read_bytes())
`], {stdio:'inherit'});
console.log('Packaged Python source → public/engine/posterior_lab.zip');
