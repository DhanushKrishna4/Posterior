/** Visual and interaction evidence for the redesigned workbench.
 * Build first; this script starts a production preview automatically. Set
 * POSTERIOR_URL to check an existing server or the deployed /Posterior/ site.
 * These are real Python-worker UI runs, not fixtures.
 */
import {chromium} from '@playwright/test';
import {access, mkdir, readdir, writeFile} from 'node:fs/promises';
import os from 'node:os';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';

const url = process.env.POSTERIOR_URL || 'http://127.0.0.1:4179/Posterior/';
const out = process.env.POSTERIOR_REPORT_DIR || 'docs/redesign/after';
await mkdir(out, {recursive:true});
let server;
if (!process.env.POSTERIOR_URL) {
  server = spawn(process.execPath, ['node_modules/vite/bin/vite.js','preview','--host','127.0.0.1','--port','4179','--strictPort'], {stdio:['ignore','pipe','pipe']});
  let output = '', exited = false, available = false;
  const capture = chunk => { output = (output+chunk.toString()).slice(-4096); };
  server.stdout.on('data',capture);
  server.stderr.on('data',capture);
  server.on('exit',() => { exited = true; });
  server.on('error',error => { output += error.message; exited = true; });
  process.once('exit',() => server.kill());
  const deadline = Date.now()+15000;
  while (Date.now() < deadline && !exited) {
    try { if ((await fetch(url,{signal:AbortSignal.timeout(750)})).ok) { available = true; break; } }
    catch {}
    await new Promise(resolve => setTimeout(resolve,100));
  }
  if (!available || exited) { server.kill(); throw new Error(`Preview HTTP endpoint did not become ready: ${output}`); }
}
let executablePath = process.env.PLAYWRIGHT_EXECUTABLE;
if (!executablePath) {
  try { await access(chromium.executablePath()); }
  catch {
    const cache = process.platform === 'darwin' ? `${os.homedir()}/Library/Caches/ms-playwright` : `${os.homedir()}/.cache/ms-playwright`;
    const dirs = (await readdir(cache)).filter(v => /^chromium-\d+$/.test(v)).sort((a,b) => Number(b.split('-')[1])-Number(a.split('-')[1]));
    if (dirs.length) executablePath = process.platform === 'darwin'
      ? `${cache}/${dirs[0]}/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`
      : `${cache}/${dirs[0]}/chrome-linux/chrome`;
  }
}
const browser = await chromium.launch({headless:true, executablePath});
const context = await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1,acceptDownloads:true});
const page = await context.newPage();
const report = {
  schema:1, timestamp:new Date().toISOString(), url,
  environment:{node:process.version,platform:process.platform,arch:process.arch,chromium:browser.version()},
  screenshot_convention:'Viewport size, full-page/viewport scope and scroll position are recorded for each capture. Full-page captures start at the top. Inference screenshots contain real worker results.',
  checks:[], screenshots:[], runs:[], page_errors:[], console_errors:[], failed_requests:[],
};
page.on('pageerror',error => report.page_errors.push(error.message));
page.on('console',message => { if (message.type() === 'error') report.console_errors.push(message.text()); });
page.on('requestfailed',request => report.failed_requests.push({url:request.url(),failure:request.failure()?.errorText}));
const save = () => writeFile(`${out}/report.json`, JSON.stringify(report,null,2)+'\n');
const ready = () => page.waitForFunction(() => window.posterior?.engine?.ready && window.posterior.state.spec, null, {timeout:90000});
const terminal = () => page.waitForFunction(() => ['complete','failed','budget_exhausted','cancelled'].includes(window.posterior.state.run?.status), null, {timeout:180000});
async function settled() {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(() => !window.posterior?.state.renderPending);
  await page.waitForTimeout(200);
}
async function check(name, fn) {
  const start = performance.now();
  try {
    const details = await fn();
    report.checks.push({name,status:'passed',seconds:(performance.now()-start)/1000,...(details ? {details} : {})});
    console.log(`PASS ${name}`);
  } catch (error) {
    report.checks.push({name,status:'failed',seconds:(performance.now()-start)/1000,error:error.stack});
    console.log(`FAIL ${name}: ${error.message}`);
    await page.evaluate(() => document.querySelectorAll('dialog[open]').forEach(dialog => dialog.close())).catch(() => {});
  }
  await save();
}
async function screenshot(name, width=1440, height=1000, {fullPage=true,scrollTarget}={}) {
  await page.setViewportSize({width,height});
  await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo({top:0,left:0,behavior:'instant'}); });
  await settled();
  if (scrollTarget) {
    await page.locator(scrollTarget).evaluate(element => element.scrollIntoView({block:'start',behavior:'instant'}));
    await settled();
  }
  const metrics = await page.evaluate(() => {
    const run = document.querySelector('#run').getBoundingClientRect();
    const plot = document.querySelector('#main-canvas').getBoundingClientRect();
    return {
      document_width:document.documentElement.scrollWidth, document_height:document.documentElement.scrollHeight,
      scroll_y:scrollY, run_button:{top:run.top,bottom:run.bottom,left:run.left,right:run.right},
      main_plot:{top:plot.top,bottom:plot.bottom,width:plot.width,height:plot.height},
      primary_action_within_viewport:run.top >= 0 && run.bottom <= innerHeight,
    };
  });
  await page.screenshot({path:`${out}/${name}.png`,fullPage});
  const result = {name,width,height,full_page:fullPage,scroll_target:scrollTarget??null,...metrics};
  report.screenshots.push(result);
  assert.ok(metrics.document_width <= width+1, `${name}: horizontal overflow ${metrics.document_width} > ${width}`);
  if (!scrollTarget) assert.equal(metrics.scroll_y,0,`${name}: capture must begin at the top`);
  return result;
}
async function runUI(settings, label) {
  for (const [id,value] of Object.entries(settings)) {
    const locator = page.locator(`#${id}`);
    if (id === 'chains') await locator.selectOption(String(value));
    else await locator.fill(String(value));
  }
  await page.locator('#run').click();
  await terminal();
  await settled();
  const run = await page.evaluate(() => {
    const r = posterior.state.run;
    return {status:r.status,reason:r.reason,settings:r.settings,progress:r.progress,stats:r.stats,diagnostics:r.diagnostics,model_hash:r.model_hash,retained_per_chain:r.samples.map(chain => chain.length)};
  });
  report.runs.push({label,...run});
  assert.equal(run.status,'complete',run.reason || label);
  return run;
}

await check('Runtime, document identity and initial responsive composition', async () => {
  await page.goto(url);
  await ready();
  await settled();
  const documentState = await page.evaluate(() => {
    const ids = [...document.querySelectorAll('[id]')].map(element => element.id);
    return {
      duplicate_ids:ids.filter((id,index) => ids.indexOf(id) !== index),
      empty_ids:ids.filter(id => !id),
      broken_label_targets:[...document.querySelectorAll('label[for]')].filter(label => !document.getElementById(label.htmlFor)).map(label => label.htmlFor),
      model_count:document.querySelectorAll('[data-model]').length,
      runtime:document.querySelector('#runtime-label').textContent,
      figure:document.querySelector('#figure-number').textContent,
    };
  });
  assert.deepEqual(documentState.duplicate_ids,[]);
  assert.deepEqual(documentState.empty_ids,[]);
  assert.deepEqual(documentState.broken_label_targets,[]);
  assert.equal(documentState.model_count,5);
  assert.deepEqual(await page.locator('dialog').evaluateAll(dialogs=>dialogs.filter(d=>!document.getElementById(d.getAttribute('aria-labelledby'))?.textContent.trim()).map(d=>d.id)),[]);
  const desktop = await screenshot('desktop-initial',1440,1000);
  await screenshot('desktop-first-viewport',1440,1000,{fullPage:false});
  await screenshot('desktop-1280-initial',1280,900);
  await screenshot('tablet-initial',768,1024);
  const mobile = await screenshot('mobile-initial',390,844);
  await screenshot('mobile-first-viewport',390,844,{fullPage:false});
  assert.ok(desktop.primary_action_within_viewport,'Desktop primary action must be in the first viewport');
  assert.ok(mobile.primary_action_within_viewport,'Mobile primary action must be in the first viewport');
  return {document:documentState,widths:[1440,1280,768,390],primary_action_visible:{desktop:true,mobile:true}};
});
await check('Keyboard figure inspection displays plotted values and dismisses', async () => {
  await page.setViewportSize({width:1440,height:1000});
  await settled();
  await page.locator('#main-canvas').focus();
  await page.keyboard.press('ArrowRight');
  const tooltip = page.locator('#plot-tooltip');
  await tooltip.waitFor({state:'visible'});
  const content = await tooltip.textContent();
  assert.match(content,/posterior|prior|density/i);
  assert.doesNotMatch(content,/NaN|Infinity/);
  const bounds = await tooltip.boundingBox();
  assert.ok(bounds.x >= 0 && bounds.y >= 0 && bounds.x+bounds.width <= 1441 && bounds.y+bounds.height <= 1001);
  await page.keyboard.press('Escape');
  assert.equal(await tooltip.isVisible(),false);
  return {content,tooltip_within_viewport:true,escape_dismissed:true};
});
await check('Guide keyboard activation, dialog focus and focus restoration', async () => {
  await page.locator('#help-button').focus();
  await page.keyboard.press('Enter');
  assert.equal(await page.locator('#help-dialog').isVisible(),true);
  assert.equal(await page.evaluate(() => document.querySelector('#help-dialog').contains(document.activeElement)),true);
  await screenshot('desktop-guide');
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#help-dialog').isVisible(),false);
  assert.equal(await page.evaluate(() => document.activeElement.id),'help-button');
  return {opened_with:'Enter',closed_with:'Escape',focus_returned_to:'help-button'};
});
await check('Default real Beta NUTS result and matching desktop/mobile evidence', async () => {
  const run = await runUI({warmup:200,draws:200,chains:4,seed:42},'Default Beta NUTS');
  assert.deepEqual(run.retained_per_chain,[200,200,200,200]);
  assert.equal(run.settings.algorithm,'nuts');
  await screenshot('desktop-result');
  await screenshot('desktop-workspace-result',1440,1000,{fullPage:false,scrollTarget:'#workbench'});
  await screenshot('mobile-result',390,844);
  return {retained:800,settings:run.settings,diagnostic_status:run.diagnostics.status};
});
await check('Trajectory and diagnostic navigation expose correct selected state', async () => {
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('[data-view="trajectory"]').click();
  await page.locator('[data-lower="transition"]').click();
  await settled();
  assert.equal(await page.locator('[data-view="trajectory"]').getAttribute('aria-pressed'),'true');
  assert.equal(await page.locator('[data-view="density"]').getAttribute('aria-pressed'),'false');
  assert.equal(await page.locator('[data-lower="transition"]').getAttribute('aria-pressed'),'true');
  assert.ok(await page.locator('#transition-select option').count() > 0);
  const text = await page.locator('#transition-details').textContent();
  assert.match(text,/trajectory|integration|algorithm/);
  await screenshot('desktop-secondary');
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.locator('#replay').click();
  const replay = await page.evaluate(() => {
    const state = posterior.state;
    const transition = state.run.transitions[state.transitionIndex<0 ? state.run.transitions.length-1 : state.transitionIndex];
    return {visible_positions:state.replayCount,total_positions:transition.trajectory.length};
  });
  assert.equal(replay.visible_positions,replay.total_positions);
  await page.emulateMedia({reducedMotion:'no-preference'});
  return {main_view:'trajectory',diagnostic_view:'transition',recorded_transitions:await page.locator('#transition-select option').count(),reduced_motion_replay:replay};
});
await check('Parameter edits recompute the independent reference and clear old draws', async () => {
  const before = await page.evaluate(() => posterior.state.reference.mean[0]);
  await page.locator('#cfg-successes').fill('8');
  await page.locator('#cfg-successes').press('Tab');
  await page.waitForFunction(() => posterior.state.spec?.metadata?.config?.successes === 8 && !posterior.state.run && posterior.state.reference.mean?.[0] > 0.7);
  const after = await page.evaluate(() => posterior.state.reference.mean[0]);
  assert.ok(Math.abs(after-10/14) < 1e-12);
  assert.notEqual(before,after);
  assert.equal(await page.locator('#transition-details').textContent(),'No completed transition has been recorded.');
  assert.equal(await page.locator('#transition-details').isVisible(),true);
  assert.equal(await page.locator('#status-text').textContent(),'Ready for a new run');
  assert.equal(await page.locator('#diagnostic-status').textContent(),'Too few draws to assess.');
  assert.doesNotMatch(await page.locator('#diag-caption').textContent(),/ELBO|occupancy|Coverage y/);
  await page.locator('#cfg-successes').fill('7');
  await page.locator('#cfg-successes').press('Tab');
  await page.waitForFunction(() => posterior.state.spec?.metadata?.config?.successes === 7);
  await page.locator('[data-view="density"]').click();
  await page.locator('[data-lower="trace"]').click();
  return {before,after,analytic_expected:10/14,old_draws_cleared:true,old_transition_metadata_cleared:true,diagnostic_status_reset:true};
});
await check('Advanced model editor and resource controls remain accessible', async () => {
  await page.getByText('Advanced model editor',{exact:true}).click();
  await page.getByText('Integration & resource budgets',{exact:true}).click();
  assert.ok(await page.locator('#model-editor').isVisible());
  assert.ok(await page.locator('#gradient-budget').isVisible());
  const model = JSON.parse(await page.locator('#model-editor').inputValue());
  assert.equal(model.metadata.preset,'beta');
  await screenshot('desktop-advanced');
  await page.getByText('Advanced model editor',{exact:true}).click();
  await page.getByText('Integration & resource budgets',{exact:true}).click();
  return {editor_model:model.metadata.preset,resource_budget_controls:true};
});
await check('Regression real computation and analytic predictive bands', async () => {
  await page.locator('[data-model="regression"]').click();
  await page.waitForFunction(() => posterior.state.spec?.metadata?.preset === 'regression');
  const run = await runUI({warmup:25,draws:60,chains:2,seed:42},'Regression NUTS visual verification');
  assert.deepEqual(run.retained_per_chain,[60,60]);
  assert.ok(await page.locator('#open-csv').isVisible());
  const reference = await page.evaluate(() => ({mean:posterior.state.reference.mean_lower.length,predictive:posterior.state.reference.predictive_lower.length}));
  assert.ok(reference.mean > 20 && reference.predictive > 20);
  await screenshot('desktop-regression');
  return {retained:120,reference,interpretation:'Short execution run; no inference-quality claim.'};
});
await check('Gaussian VI preserves separate fit summaries and objective view', async () => {
  await page.locator('[data-model="gaussian"]').click();
  await page.waitForFunction(() => posterior.state.spec?.metadata?.preset === 'gaussian');
  await page.locator('[data-lower="transition"]').click();
  await page.locator('[data-algorithm="vi"]').click();
  assert.equal(await page.locator('[data-lower="transition"]').isDisabled(),true);
  assert.equal(await page.locator('[data-lower="objective"]').getAttribute('aria-pressed'),'true');
  assert.equal(await page.locator('.lower-plot').isVisible(),true);
  const run = await runUI({iterations:75,draws:60,chains:2,seed:42},'Gaussian VI visual verification');
  assert.equal(run.diagnostics.status,'not applicable');
  assert.deepEqual(run.retained_per_chain,[60,60]);
  await page.locator('[data-lower="objective"]').click();
  assert.equal(await page.locator('[data-lower="objective"]').getAttribute('aria-pressed'),'true');
  assert.equal(await page.locator('#vi-fit option').count(),2);
  assert.match(await page.locator('#diag-caption').textContent(),/ELBO/);
  await screenshot('desktop-variational');
  await screenshot('mobile-variational',390,844);
  return {fits:2,updates:75,draws_per_fit:60,mcmc_diagnostics:'not applicable',interpretation:'Short execution run; no inference-quality claim.'};
});
await check('Reduced motion, mobile dialog and clean console/static requests', async () => {
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.setViewportSize({width:390,height:844});
  await page.locator('#help-button').focus();
  await page.keyboard.press('Enter');
  assert.equal(await page.locator('#help-dialog').isVisible(),true);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#help-dialog').isVisible(),false);
  assert.equal(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches),true);
  assert.deepEqual(report.page_errors,[]);
  assert.deepEqual(report.console_errors,[]);
  assert.deepEqual(report.failed_requests,[]);
  return {reduced_motion_requested:true,page_errors:0,console_errors:0,failed_requests:0};
});
report.summary = {passed:report.checks.filter(c => c.status === 'passed').length,failed:report.checks.filter(c => c.status === 'failed').length};
await save();
await browser.close();
server?.kill();
console.log(JSON.stringify(report.summary));
process.exitCode = report.summary.failed ? 1 : 0;
