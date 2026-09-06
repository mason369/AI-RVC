import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
execFileSync(process.execPath, ['node_modules/@tailwindcss/cli/dist/index.mjs', '-i', 'src/style.css', '-o', 'dist/player.css', '--minify'], {stdio:'inherit'});
await build({entryPoints:['src/main.tsx'], bundle:true, minify:true, outfile:'dist/player.js', jsx:'automatic', define:{'process.env.NODE_ENV':'"production"'}, legalComments:'eof'});
const css=readFileSync('dist/player.css','utf8');
const js=readFileSync('dist/player.js','utf8').replaceAll('</script', '<\\/script');
writeFileSync('dist/player.html', `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${css}</style></head><body><div id="root"></div><script>${js}</script></body></html>`);
