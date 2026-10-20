import test from 'node:test';
import assert from 'node:assert/strict';
import worker,{refresh,fullRefreshPolicy,REPO,DUE_MS,FULL_RETRY_WINDOW_MS,FULL_MAX_ATTEMPTS} from '../cloudflare/scheduler.mjs';
const now=Date.parse('2026-10-15T18:00:00Z');
function fake({age=15*60000,active=[],fullAge=3600000,fullAttemptAge=null,fullRuns=null,dispatch=204,status='ok',fail=false}={}){
  const calls=[];
  const fetchImpl=async(url,options={})=>{
    calls.push({url,options});
    if(fail)return new Response('',{status:503});
    if(options.method==='POST')return new Response(null,{status:dispatch});
    const success=fullAge===null?[]:[{conclusion:'success',updated_at:new Date(now-fullAge).toISOString(),created_at:new Date(now-fullAge).toISOString()}];
    const value=url.includes('raw.githubusercontent.com')?{generated:new Date(now-age).toISOString(),open:[],settled:[],refresh:{status}}:
      url.includes('status=success')?{workflow_runs:success}:
      url.includes('full.yml/runs')?{workflow_runs:fullRuns||[
        ...(fullAttemptAge===null?[]:[{conclusion:'failure',created_at:new Date(now-fullAttemptAge).toISOString()}]),
        ...success,
      ]}:{workflow_runs:active};
    return Response.json(value);
  };return{calls,fetchImpl};
}
const run=f=>refresh({GITHUB_TOKEN:'test-only-token'},{fetchImpl:f.fetchImpl,now});
test('independent tick dispatches only the fixed Sports workflow when due, not a trading or arbitrary target',async()=>{
  const f=fake();const r=await run(f);assert.equal(r.state,'dispatched');
  const post=f.calls.find(c=>c.options.method==='POST');
  assert.equal(post.url,`https://api.github.com/repos/${REPO}/actions/workflows/light.yml/dispatches`);
  assert.deepEqual(JSON.parse(post.options.body),{ref:'main'});
  assert.equal(f.calls.find(c=>c.url.includes('raw.')).options.headers.Authorization,undefined);
});
test('fresh snapshots do not dispatch; the due boundary and degraded source retry are explicit',async()=>{
  for(const age of [0,DUE_MS-1]){const f=fake({age});assert.equal((await run(f)).state,'fresh');}
  assert.equal((await run(fake({age:DUE_MS}))).state,'dispatched');
  assert.equal((await run(fake({age:0,status:'degraded'}))).state,'dispatched');
});
test('queued/running light OR full jobs prevent competing runners, and stuck queues fail loudly',async()=>{
  for(const path of ['.github/workflows/light.yml','.github/workflows/full.yml']){
    const active=[{id:1,path,status:'queued',created_at:new Date(now-60000).toISOString()}],f=fake({active});
    assert.equal((await run(f)).state,'running');assert.equal(f.calls.some(c=>c.options.method==='POST'),false);
    active[0].created_at=new Date(now-46*60000).toISOString();await assert.rejects(run(fake({active})),/stuck/);
  }
});
test('daily full data refresh is recovered even while the live board keeps updating',async()=>{
  for(const fullAge of [null,27*3600000]){
    const f=fake({age:0,fullAge});assert.equal((await run(f)).workflow,'full.yml');
  }
});
test('failed heavy full refresh backs off while due light refreshes can still run',async()=>{
  const f=fake({fullAge:27*3600000,fullAttemptAge:60000});
  assert.equal((await run(f)).workflow,'light.yml');
});

const attempt=(age,conclusion='failure',completedAge=age)=>({conclusion,
  created_at:new Date(now-age).toISOString(),updated_at:new Date(now-completedAge).toISOString()});
const policy=(runs,lastSuccess=null,at=now)=>fullRefreshPolicy(runs,lastSuccess,at);

test('first failed full run waits two hours FROM COMPLETION, second waits six',()=>{
  assert.equal(policy([attempt(3*3600000,'failure',3600000)]).status,'retry-backoff');
  assert.equal(policy([attempt(4*3600000,'failure',2*3600000)]).due,true);
  const runs=[attempt(7*3600000,'failure',5*3600000),attempt(10*3600000)];
  assert.equal(policy(runs).status,'retry-backoff');
  assert.equal(policy(runs,null,now+3600000).due,true);
});

test('three failed/cancelled attempts block heavy recovery but never suppress live refreshes',async()=>{
  const fullRuns=[attempt(3600000),attempt(4*3600000,'cancelled'),attempt(8*3600000,'timed_out')];
  const p=policy(fullRuns);
  assert.equal(p.status,'retry-limit');assert.equal(p.attempts24h,FULL_MAX_ATTEMPTS);
  assert.equal(p.retryAfter,new Date(now-8*3600000+FULL_RETRY_WINDOW_MS).toISOString());
  for(const age of [0,DUE_MS]){
    const f=fake({fullAge:null,fullRuns,age});const r=await run(f);
    assert.equal(r.state,age?'dispatched':'full-refresh-blocked');
    assert.equal(r.fullRefresh.status,'retry-limit');
    assert.equal(f.calls.some(c=>c.options.method==='POST'&&c.url.includes('full.yml')),false);
    if(age)assert.equal(r.workflow,'light.yml');
  }
});

test('legacy retry storm survives redeploy and waits until fewer than three attempts remain in rolling 24h',()=>{
  const runs=Array.from({length:23},(_,i)=>attempt((i+1)*3600000));
  assert.equal(policy(runs).status,'retry-limit');
  assert.equal(policy(runs).retryAfter,new Date(now+21*3600000).toISOString());
  assert.equal(policy(runs,null,now+21*3600000-1).status,'retry-limit');
  assert.equal(policy(runs,null,now+21*3600000).due,true);
});

test('separate last-success lookup prevents older success falling off first history page',async()=>{
  const f=fake({fullAge:3600000,fullRuns:Array.from({length:12},(_,i)=>attempt((i+2)*3600000)),age:0});
  const r=await run(f);assert.equal(r.state,'fresh');assert.equal(r.fullRefresh.status,'current');
  assert.equal(f.calls.some(c=>c.url.includes('status=success&per_page=1')),true);
});

test('success makes full refresh current; another light board never resets failed budget',async()=>{
  const f=fake({fullAge:60000,fullRuns:[attempt(60000,'success'),attempt(3600000)],age:0});
  assert.equal((await run(f)).fullRefresh.status,'current');
  assert.equal(policy([attempt(3600000),attempt(2*3600000),attempt(3*3600000)]).status,'retry-limit');
});

test('incomplete or corrupt full-run evidence cannot silently launch heavy recovery',()=>{
  assert.equal(policy(Array.from({length:100},(_,i)=>attempt((i+1)*60000))).status,'history-incomplete');
  assert.throws(()=>policy([{created_at:'bad'}]),/timestamps/);
  assert.throws(()=>policy(null),/history/);
  assert.throws(()=>policy([], {updated_at:'bad'}),/successful/);
});

test('health discloses exhausted full recovery even when the live board is fresh',async t=>{
  const f=fake({age:0,fullAge:5*24*3600000,fullRuns:[attempt(3600000),attempt(2*3600000),attempt(3*3600000)]});
  t.mock.method(Date,'now',()=>now);t.mock.method(globalThis,'fetch',f.fetchImpl);
  const response=await worker.fetch(new Request('https://worker.example/health'),{GITHUB_TOKEN:'test-only-token'});
  const body=await response.json();
  assert.equal(response.status,200);assert.equal(body.stale,false);
  assert.equal(body.fullRefreshHealthy,false);assert.equal(body.fullRefresh.status,'retry-limit');
  assert.equal(body.fullRefresh.maxAttempts24h,3);assert.ok(body.fullRefresh.retryAfter);
  assert.equal(f.calls.some(c=>c.options.method==='POST'),false);
  assert.equal(JSON.stringify(body).includes('test-only-token'),false);
});
test('authentication, source and dispatch failures never claim refresh success',async()=>{
  await assert.rejects(refresh({},{}),/not configured/);
  await assert.rejects(run(fake({fail:true})),/HTTP 503/);
  await assert.rejects(run(fake({dispatch:403})),/dispatch failed/);
  await assert.rejects(run(fake({age:-180000})),/invalid/);
});
test('public requests can never trigger a workflow',async()=>{
  for(const method of ['GET','POST','PUT']){
    const result=await worker.fetch(new Request('https://worker.example/refresh',{method}),{});
    assert.equal(result.status,404);
  }
});
