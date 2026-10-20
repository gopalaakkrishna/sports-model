// Only starts the existing Sports workflows. No trading API, account keys,
// ledger writes, prediction logic or public mutation endpoint lives here.
export const REPO='gopalaakkrishna/sports-model';
export const BOARD_URL=`https://raw.githubusercontent.com/${REPO}/main/public/sports.json`;
export const DUE_MS=12*60*1000;
export const VERSION='2026.10.20-sports-recovery-v2';
export const FULL_RETRY_WINDOW_MS=24*3600000;
export const FULL_MAX_ATTEMPTS=3;
const FULL_BACKOFF_MS=[2*3600000,6*3600000];
const ACTIVE=new Set(['queued','in_progress','requested','waiting','pending']);
const WORKFLOWS=new Set(['.github/workflows/light.yml','.github/workflows/full.yml']);

async function read(url,fetchImpl,headers={}) {
  const response=await fetchImpl(url,{headers,cache:'no-store',signal:AbortSignal.timeout(12000)});
  if(!response.ok)throw Error(`Sports scheduler upstream HTTP ${response.status}`);
  return response.json();
}

export function fullRefreshPolicy(runs,lastSuccess,now) {
  if(!Array.isArray(runs))throw Error('Sports full-refresh history is invalid.');
  const dated=runs.map(run=>({run,created:Date.parse(run.created_at),
    completed:Date.parse(run.updated_at||run.created_at)}));
  if(dated.some(r=>!Number.isFinite(r.created)||!Number.isFinite(r.completed)||r.created>now+120000))
    throw Error('Sports full-refresh history has invalid timestamps.');
  dated.sort((a,b)=>b.created-a.created);
  const lastFull=lastSuccess?Date.parse(lastSuccess.updated_at):NaN;
  if(lastSuccess&&(!Number.isFinite(lastFull)||lastFull>now+120000))
    throw Error('Sports last successful full refresh is invalid.');
  const needed=!Number.isFinite(lastFull)||now-lastFull>=26*3600000;
  // Include scheduled, manual, failed AND cancelled attempts. A new Worker
  // deployment or a fresh light board never resets the heavy-run budget.
  const recent=dated.filter(r=>now-r.created<FULL_RETRY_WINDOW_MS&&r.run.conclusion!=='success');
  const latest=dated[0];
  // If a whole page is newer than the budget window, history is incomplete:
  // block heavy recovery rather than infer that unseen failures do not exist.
  const complete=runs.length<100||dated.at(-1).created<=now-FULL_RETRY_WINDOW_MS;
  let reason=null,retryAt=null;
  if(needed&&!complete)reason='history-incomplete';
  else if(needed&&recent.length>=FULL_MAX_ATTEMPTS){
    reason='retry-limit';
    retryAt=recent[FULL_MAX_ATTEMPTS-1].created+FULL_RETRY_WINDOW_MS;
  }else if(needed&&latest&&latest.run.conclusion!=='success'){
    const delay=FULL_BACKOFF_MS[Math.min(Math.max(recent.length,1)-1,FULL_BACKOFF_MS.length-1)];
    retryAt=latest.completed+delay;
    if(now<retryAt)reason='retry-backoff';
  }
  return {needed,due:needed&&!reason,status:!needed?'current':reason||'due',
    attempts24h:recent.length,maxAttempts24h:FULL_MAX_ATTEMPTS,historyComplete:complete,
    retryAfter:reason&&Number.isFinite(retryAt)?new Date(retryAt).toISOString():null,
    lastSuccessAt:Number.isFinite(lastFull)?new Date(lastFull).toISOString():null,
    lastAttemptAt:latest?new Date(latest.created).toISOString():null};
}

export async function inspect(env,{fetchImpl=fetch,now=Date.now()}={}) {
  if(!env.GITHUB_TOKEN)throw Error('Sports workflow credential is not configured.');
  const headers={Authorization:`Bearer ${env.GITHUB_TOKEN}`,Accept:'application/vnd.github+json',
    'X-GitHub-Api-Version':'2022-11-28','User-Agent':`Tara-Sports-refresh/${VERSION}`};
  const api=`https://api.github.com/repos/${REPO}`;
  // Do not send the GitHub credential to the public snapshot host.
  const [board,runs,full,success]=await Promise.all([
    read(`${BOARD_URL}?scheduler=${Math.floor(now/60000)}`,fetchImpl),
    read(`${api}/actions/runs?per_page=30`,fetchImpl,headers),
    read(`${api}/actions/workflows/full.yml/runs?per_page=100`,fetchImpl,headers),
    // Last success may be older than the recent-attempt page. Light refreshes
    // cannot stand in for this independent heavy-data success clock.
    read(`${api}/actions/workflows/full.yml/runs?status=success&per_page=1`,fetchImpl,headers),
  ]);
  const generated=Date.parse(board.generated),age=now-generated;
  if(!Number.isFinite(generated)||age < -120000||!Array.isArray(board.open)||!Array.isArray(board.settled))
    throw Error('Sports board has invalid identity or freshness evidence.');
  const active=(runs.workflow_runs||[]).filter(r=>WORKFLOWS.has(r.path)&&ACTIVE.has(r.status));
  if(!Array.isArray(success.workflow_runs)||success.workflow_runs.some(r=>r.conclusion!=='success'))
    throw Error('Sports successful full-refresh history is invalid.');
  const fullRefresh=fullRefreshPolicy(full.workflow_runs,success.workflow_runs[0],now);
  const due=age>=DUE_MS||board.refresh?.status==='degraded';
  return {headers,api,active,fullDue:fullRefresh.due,fullRefresh,due,ageMs:Math.max(0,age),generated:board.generated,
    sourceStatus:board.refresh?.status||'legacy'};
}

export async function refresh(env,options={}) {
  const now=options.now??Date.now(),fetchImpl=options.fetchImpl??fetch;
  const status=await inspect(env,{fetchImpl,now});
  if(status.active.length){
    if(status.active.some(r=>now-Date.parse(r.created_at)>45*60000))
      throw Error('Sports workflow is stuck or queued over45minutes; inspect GitHub Actions.');
    return {state:'running',ageMs:status.ageMs,active:status.active.map(r=>({id:r.id,status:r.status}))};
  }
  if(!status.fullDue&&!status.due)return {state:status.fullRefresh.needed?'full-refresh-blocked':'fresh',
    generated:status.generated,ageMs:status.ageMs,fullRefresh:status.fullRefresh};
  const workflow=status.fullDue?'full.yml':'light.yml';
  const response=await fetchImpl(`${status.api}/actions/workflows/${workflow}/dispatches`,{
    method:'POST',headers:{...status.headers,'Content-Type':'application/json'},
    body:JSON.stringify({ref:'main'}),signal:AbortSignal.timeout(12000),
  });
  if(response.status!==204)throw Error(`Sports workflow dispatch failed: HTTP ${response.status}`);
  // Accepted means queued, not refreshed or published. The next tick/health
  // check independently verifies the source timestamp and running job.
  return {state:'dispatched',workflow,generated:status.generated,ageMs:status.ageMs,fullRefresh:status.fullRefresh};
}

export default {
  async scheduled(event,env,ctx){
    try{const result=await refresh(env);console.log(JSON.stringify({at:new Date().toISOString(),...result}));}
    catch(error){console.error(error.message);throw error;}
  },
  async fetch(request,env){
    const url=new URL(request.url);
    if(request.method!=='GET'||url.pathname!=='/health')return new Response('Not found',{status:404});
    try{
      const s=await inspect(env);
      return Response.json({version:VERSION,checkedAt:new Date().toISOString(),generated:s.generated,
        ageMs:s.ageMs,sourceStatus:s.sourceStatus,running:s.active.length>0,
        stale:s.ageMs>30*60000||s.sourceStatus==='degraded',fullRefreshDue:s.fullDue,
        fullRefreshHealthy:!s.fullRefresh.needed,fullRefresh:s.fullRefresh},{headers:{'Cache-Control':'no-store'}});
    }catch(error){return Response.json({error:error.message},{status:503,headers:{'Cache-Control':'no-store'}});}
  },
};
