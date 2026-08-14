#!/usr/bin/env node
// Render each LaTeX formula to a tight transparent PNG via headless Chrome + (inlined) KaTeX.
// Usage: node render_math.cjs <formulas.json> <out_dir> <map.json>
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const KATEX = "C:\\Users\\administrator\\.workbuddy\\binaries\\node\\wenyan-venv\\node_modules\\katex";

const formulas = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const outDir = process.argv[3];
const mapPath = process.argv[4];
fs.mkdirSync(outDir, { recursive: true });

const WS = globalThis.WebSocket;
if (!WS) { console.error("Node has no global WebSocket"); process.exit(2); }

const katexCSS = fs.readFileSync(KATEX + "\\dist\\katex.min.css", "utf8");
const katexJS  = fs.readFileSync(KATEX + "\\dist\\katex.min.js", "utf8");

function buildHTML(tex, display){
  return `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;background:transparent;}
#m{display:inline-block;padding:3px;font-size:17px;color:#2f2f2f;line-height:1.2;}
${katexCSS}</style></head><body><div id="m"></div>
<script>${katexJS}</script>
<script>
(function(){
  var tex=${JSON.stringify(tex)};
  try{ katex.render(tex, document.getElementById('m'), {displayMode:${display?"true":"false"}, throwOnError:false}); }
  catch(e){ document.getElementById('m').textContent=tex; }
  window.__ready=true;
})();
</script></body></html>`;
}

const sleep = (ms)=>new Promise(r=>setTimeout(r,ms));
const getJSON = async (u)=>{ const r=await fetch(u); return r.json(); };
function send(ws, method, params={}, id){ return new Promise((resolve)=>{
  const onMsg=(ev)=>{ const o=JSON.parse(ev.data); if(o.id===id){ ws.removeEventListener("message",onMsg); resolve(o); } };
  ws.addEventListener("message", onMsg); ws.send(JSON.stringify({id,method,params}));
}); }

async function main(){
  const userData = path.join(os.tmpdir(), "chrome_wx_" + Date.now());
  const chrome = spawn(CHROME, [
    "--headless=new","--disable-gpu","--no-sandbox","--disable-dev-shm-usage",
    "--remote-debugging-port=9222","--user-data-dir="+userData
  ]);
  let ver=null;
  for(let i=0;i<80;i++){ try{ ver=await getJSON("http://127.0.0.1:9222/json/version"); if(ver) break; }catch(e){} await sleep(250); }
  if(!ver){ console.error("chrome devtools not up"); chrome.kill("SIGKILL"); process.exit(3); }

  // must connect to a PAGE target, not the browser-level endpoint
  let targets = await getJSON("http://127.0.0.1:9222/json");
  let target = (targets||[]).find(t=>t.type==="page" && t.url!=="about:blank") || (targets||[]).find(t=>t.type==="page");
  if(!target){ target = await (await fetch("http://127.0.0.1:9222/json/new",{method:"PUT"})).json(); }
  if(!target||!target.webSocketDebuggerUrl){ console.error("no page target"); chrome.kill("SIGKILL"); process.exit(3); }

  const ws = new WS(target.webSocketDebuggerUrl);
  await new Promise(r=>ws.addEventListener("open", r));
  let id=1; const s=(m,p)=>send(ws,m,p,id++);
  await s("Page.enable"); await s("Runtime.enable");
  ws.addEventListener("message", (ev)=>{ const o=JSON.parse(ev.data);
    if(o.method==="Runtime.exceptionThrown"){ console.error("PAGE EXCEPTION:", JSON.stringify(o.params&&o.params.exceptionDetails)); }
  });

  const tmp = path.join(os.tmpdir(), "katex_html_" + Date.now());
  fs.mkdirSync(tmp, { recursive:true });
  const map = {};
  for(const f of formulas){
    const htmlPath = path.join(tmp, f.key + ".html");
    fs.writeFileSync(htmlPath, buildHTML(f.tex, f.display));
    const url = "file://" + htmlPath.replace(/\\/g,"/");
    await s("Page.navigate", { url });
    await sleep(400);
    if(globalThis.DEBUG){ const dbg = await s("Runtime.evaluate", { expression:"JSON.stringify({rs:document.readyState,href:location.href.slice(0,60),katex:typeof katex,ready:window.__ready,hash:location.hash.slice(0,40)})", returnByValue:true }); console.error("DBG", JSON.stringify(dbg&&dbg.result&&dbg.result.result)); }
    let ready=false;
    for(let i=0;i<30;i++){ await sleep(100);
      const r = await s("Runtime.evaluate", { expression:"window.__ready===true", returnByValue:true });
      if(r&&r.result&&r.result.result&&r.result.result.value===true){ ready=true; break; }
    }
    if(!ready){ console.error("timeout rendering", f.key); continue; }
    const box = await s("Runtime.evaluate", { expression:"(function(){var e=document.getElementById('m');var r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};})()", returnByValue:true });
    const b = box.result.result.value;
    const pad = 2;
    const clip = { x: Math.max(0,b.x-pad), y: Math.max(0,b.y-pad), width: Math.max(1,b.w+pad*2), height: Math.max(1,b.h+pad*2), scale: 2 };
    const shot = await s("Page.captureScreenshot", { format:"png", captureBeyondViewport:true, clip });
    if(!shot.result){ console.error("capture error for", f.key, JSON.stringify(shot).slice(0,300)); continue; }
    const b64 = shot.result.data;
    fs.writeFileSync(path.join(outDir, f.key + ".png"), Buffer.from(b64,"base64"));
    map[f.key] = "data:image/png;base64," + b64;
  }
  fs.writeFileSync(mapPath, JSON.stringify(map));
  try{ fs.unlinkSync(htmlPath);}catch(e){}
  ws.close(); chrome.kill("SIGKILL");
  console.log("RENDERED", Object.keys(map).length, "of", formulas.length, "formulas");
}
main().catch(e=>{ console.error(e); process.exit(1); });
