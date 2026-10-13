import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
const lock=JSON.parse(await readFile(new URL('./runtime-lock.json',import.meta.url),'utf8'));
for(const asset of lock.assets){
 let bytes;try{bytes=await readFile('public/runtime/'+asset.name);}catch{throw new Error(`Pinned runtime is missing ${asset.name}. Run npm run vendor first.`);}
 if(bytes.length!==asset.bytes||createHash('sha256').update(bytes).digest('hex')!==asset.sha256)throw new Error(`Runtime integrity check failed for ${asset.name}. Run npm run vendor to restore pinned assets.`);
}
execFileSync(process.execPath,['scripts/package-engine.mjs'],{stdio:'inherit'});
execFileSync(process.execPath,['node_modules/vite/bin/vite.js','build'],{stdio:'inherit'});
