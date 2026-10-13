import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const lock=JSON.parse(await readFile(new URL('./runtime-lock.json',import.meta.url),'utf8'));
const root='public/runtime/';await mkdir(root,{recursive:true});
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
for(const asset of lock.assets){
 let bytes;try{bytes=await readFile(root+asset.name);if(sha(bytes)!==asset.sha256)bytes=null;}catch{}
 if(!bytes){const response=await fetch(lock.source+asset.name);if(!response.ok)throw new Error(`Runtime asset ${asset.name}: HTTP ${response.status}`);bytes=Buffer.from(await response.arrayBuffer());if(sha(bytes)!==asset.sha256||bytes.length!==asset.bytes)throw new Error(`Pinned integrity check failed: ${asset.name}`);await writeFile(root+asset.name,bytes);}
 console.log(`${asset.name}: ${(bytes.length/1024/1024).toFixed(2)} MiB · SHA-256 verified`);
}
await writeFile(root+'manifest.json',JSON.stringify(lock,null,2)+'\n');
console.log(`Self-hosted Pyodide ${lock.pyodide}: ${(lock.assets.reduce((n,a)=>n+a.bytes,0)/1024/1024).toFixed(1)} MiB. Packages: ${lock.packages.join(', ')}.`);
