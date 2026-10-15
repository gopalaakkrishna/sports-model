import test from 'node:test';
import assert from 'node:assert/strict';
import worker,{refresh,REPO,DUE_MS} from '../cloudflare/scheduler.mjs';
const now=Date.parse('2026-10-15T18:00:00Z');
function fake({age=15*60000,active=[],fullAge=3600000,fullAttemptAge=null,dispatch=204,status='ok',fail=false}={}){
  const calls=[];
  const fetchImpl=async(url,options={})=>{
    calls.push({url,options});
    if(fail)return new Response('',{status:503});
    if(options.method==='POST')return new Response(null,{status:dispatch});
    const value=url.includes('raw.githubusercontent.com')?{generated:new Date(now-age).toISOString(),open:[],settled:[],refresh:{status}}:
      url.includes('full.yml/runs')?{workflow_runs:[
        ...(fullAttemptAge===null?[]:[{conclusion:'failure',created_at:new Date(now-fullAttemptAge).toISOString()}]),
        ...(fullAge===null?[]:[{conclusion:'success',updated_at:new Date(now-fullAge).toISOString(),created_at:new Date(now-fullAge).toISOString()}]),
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
test('failed heavy full refresh backs off for an hour while due light refreshes can still run',async()=>{
  const f=fake({fullAge:27*3600000,fullAttemptAge:60000});
  assert.equal((await run(f)).workflow,'light.yml');
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
