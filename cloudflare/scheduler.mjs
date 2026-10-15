// Only starts the existing Sports workflows. No trading API, account keys,
// ledger writes, prediction logic or public mutation endpoint lives here.
export const REPO='gopalaakkrishna/sports-model';
export const BOARD_URL=`https://raw.githubusercontent.com/${REPO}/main/public/sports.json`;
export const DUE_MS=12*60*1000;
const ACTIVE=new Set(['queued','in_progress','requested','waiting','pending']);
const WORKFLOWS=new Set(['.github/workflows/light.yml','.github/workflows/full.yml']);

async function read(url,fetchImpl,headers={}) {
  const response=await fetchImpl(url,{headers,cache:'no-store',signal:AbortSignal.timeout(12000)});
  if(!response.ok)throw Error(`Sports scheduler upstream HTTP ${response.status}`);
  return response.json();
}

export async function inspect(env,{fetchImpl=fetch,now=Date.now()}={}) {
  if(!env.GITHUB_TOKEN)throw Error('Sports workflow credential is not configured.');
  const headers={Authorization:`Bearer ${env.GITHUB_TOKEN}`,Accept:'application/vnd.github+json',
    'X-GitHub-Api-Version':'2022-11-28','User-Agent':'Tara-Sports-refresh/14.7.4'};
  const api=`https://api.github.com/repos/${REPO}`;
  // Do not send the GitHub credential to the public snapshot host.
  const [board,runs,full]=await Promise.all([
    read(`${BOARD_URL}?scheduler=${Math.floor(now/60000)}`,fetchImpl),
    read(`${api}/actions/runs?per_page=30`,fetchImpl,headers),
    read(`${api}/actions/workflows/full.yml/runs?per_page=10`,fetchImpl,headers),
  ]);
  const generated=Date.parse(board.generated),age=now-generated;
  if(!Number.isFinite(generated)||age < -120000||!Array.isArray(board.open)||!Array.isArray(board.settled))
    throw Error('Sports board has invalid identity or freshness evidence.');
  const active=(runs.workflow_runs||[]).filter(r=>WORKFLOWS.has(r.path)&&ACTIVE.has(r.status));
  const fullRuns=full.workflow_runs||[];
  const lastFull=Date.parse(fullRuns.find(r=>r.conclusion==='success')?.updated_at);
  const lastAttempt=Date.parse(fullRuns[0]?.created_at);
  const fullDue=(!Number.isFinite(lastFull)||now-lastFull>=26*3600000)&&
    (!Number.isFinite(lastAttempt)||now-lastAttempt>=3600000);
  const due=age>=DUE_MS||board.refresh?.status==='degraded';
  return {headers,api,active,fullDue,due,ageMs:Math.max(0,age),generated:board.generated,
    sourceStatus:board.refresh?.status||'legacy',lastFull:Number.isFinite(lastFull)?lastFull:null};
}

export async function refresh(env,options={}) {
  const now=options.now??Date.now(),fetchImpl=options.fetchImpl??fetch;
  const status=await inspect(env,{fetchImpl,now});
  if(status.active.length){
    if(status.active.some(r=>now-Date.parse(r.created_at)>45*60000))
      throw Error('Sports workflow is stuck or queued over45minutes; inspect GitHub Actions.');
    return {state:'running',ageMs:status.ageMs,active:status.active.map(r=>({id:r.id,status:r.status}))};
  }
  if(!status.fullDue&&!status.due)return {state:'fresh',generated:status.generated,ageMs:status.ageMs};
  const workflow=status.fullDue?'full.yml':'light.yml';
  const response=await fetchImpl(`${status.api}/actions/workflows/${workflow}/dispatches`,{
    method:'POST',headers:{...status.headers,'Content-Type':'application/json'},
    body:JSON.stringify({ref:'main'}),signal:AbortSignal.timeout(12000),
  });
  if(response.status!==204)throw Error(`Sports workflow dispatch failed: HTTP ${response.status}`);
  // Accepted means queued, not refreshed or published. The next tick/health
  // check independently verifies the source timestamp and running job.
  return {state:'dispatched',workflow,generated:status.generated,ageMs:status.ageMs};
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
      return Response.json({version:'14.7.4',checkedAt:new Date().toISOString(),generated:s.generated,
        ageMs:s.ageMs,sourceStatus:s.sourceStatus,running:s.active.length>0,
        stale:s.ageMs>30*60000||s.sourceStatus==='degraded',fullRefreshDue:s.fullDue},{headers:{'Cache-Control':'no-store'}});
    }catch(error){return Response.json({error:error.message},{status:503,headers:{'Cache-Control':'no-store'}});}
  },
};
