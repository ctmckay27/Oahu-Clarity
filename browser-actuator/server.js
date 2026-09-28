const express = require('express');
const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const { chromium } = require('playwright-core');
const { createProxyMiddleware } = require('http-proxy-middleware');

const PORT = Number(process.env.PORT || 10000);
const ACTUATOR_TOKEN = process.env.ACTUATOR_TOKEN;
const PROFILE_DIR = process.env.PROFILE_DIR || '/tmp/mari404-chrome-profile';
const ALLOWED_HOSTS = new Set([
  'chatgpt.com',
  'www.chatgpt.com',
  'mari404ever.carlitoesblanco.chatgpt.site'
]);

if (!ACTUATOR_TOKEN || ACTUATOR_TOKEN.length < 24) throw new Error('ACTUATOR_TOKEN must be at least 24 characters');
fs.mkdirSync(PROFILE_DIR, { recursive: true });

let contextPromise;
async function getContext() {
  if (!contextPromise) {
    contextPromise = chromium.launchPersistentContext(PROFILE_DIR, {
      headless: false,
      viewport: { width: 1400, height: 820 },
      args: ['--no-sandbox','--disable-dev-shm-usage','--window-size=1400,820']
    }).catch(e => { contextPromise = null; throw e; });
  }
  return contextPromise;
}
async function getPage() {
  const c = await getContext();
  const pages = c.pages();
  return pages[0] || await c.newPage();
}
function secureEq(a,b){
  const x=Buffer.from(String(a||'')), y=Buffer.from(String(b||''));
  return x.length===y.length && crypto.timingSafeEqual(x,y);
}
function auth(req,res,next){
  const q=req.query.token;
  const h=(req.headers.authorization||'').replace(/^Bearer\s+/i,'');
  if(!secureEq(q||h,ACTUATOR_TOKEN)) return res.status(401).json({ok:false,error:'unauthorized'});
  next();
}
function allowedTarget(raw){
  const u=new URL(raw);
  if(u.protocol!=='https:') throw new Error('https only');
  if(!ALLOWED_HOSTS.has(u.hostname)) throw new Error('host not allowed');
  return u;
}
function selector(v){
  const s=String(v||'');
  if(!s || s.length>500) throw new Error('selector required');
  return s;
}

const app=express();
app.disable('x-powered-by');
const wsProxy=createProxyMiddleware({
  target:'http://127.0.0.1:6080',
  ws:true,
  pathRewrite:{'^/viewer/websockify':'/'}
});
app.use('/viewer/websockify',wsProxy);
app.use('/viewer',express.static('/usr/share/novnc'));
app.get('/',(_req,res)=>res.redirect('/viewer/vnc.html?autoconnect=1&resize=remote&reconnect=1&path=viewer/websockify'));
app.get('/healthz',(_req,res)=>res.json({ok:true}));

app.get('/api/status',auth,async(_req,res)=>{
  try{const p=await getPage();res.json({ok:true,url:p.url(),title:await p.title()});}
  catch(e){res.status(500).json({ok:false,error:String(e)});}
});
app.get('/api/nav',auth,async(req,res)=>{
  try{
    const p=await getPage();
    const u=allowedTarget(String(req.query.url||''));
    const r=await p.goto(u.toString(),{waitUntil:'domcontentloaded',timeout:45000});
    res.json({ok:true,url:p.url(),title:await p.title(),status:r&&r.status()});
  }catch(e){res.status(500).json({ok:false,error:String(e)});}
});
app.get('/api/snapshot',auth,async(_req,res)=>{
  try{
    const p=await getPage();
    const items=await p.locator('a,button,input,textarea,select,[role=button],[contenteditable=true]').evaluateAll(els=>els.slice(0,300).map((el,i)=>({
      i,
      tag:el.tagName.toLowerCase(),
      type:el.getAttribute('type'),
      role:el.getAttribute('role'),
      text:(el.innerText||el.value||'').trim().slice(0,240),
      aria:el.getAttribute('aria-label'),
      placeholder:el.getAttribute('placeholder'),
      name:el.getAttribute('name'),
      id:el.id||null,
      href:el.getAttribute('href'),
      disabled:!!el.disabled
    })));
    res.json({ok:true,url:p.url(),title:await p.title(),items});
  }catch(e){res.status(500).json({ok:false,error:String(e)});}
});
app.get('/api/click',auth,async(req,res)=>{
  try{
    const p=await getPage();
    await p.locator(selector(req.query.selector)).first().click({timeout:15000});
    await p.waitForTimeout(400);
    res.json({ok:true,url:p.url(),title:await p.title()});
  }catch(e){res.status(500).json({ok:false,error:String(e)});}
});
app.get('/api/click-text',auth,async(req,res)=>{
  try{
    const text=String(req.query.text||'');
    if(!text || text.length>300) throw new Error('text required');
    const p=await getPage();
    await p.getByText(text,{exact:req.query.exact==='1'}).first().click({timeout:15000});
    await p.waitForTimeout(400);
    res.json({ok:true,url:p.url(),title:await p.title()});
  }catch(e){res.status(500).json({ok:false,error:String(e)});}
});
app.get('/api/fill',auth,async(req,res)=>{try{const value=String(req.query.value||'');if(value.length>30000)throw new Error('value too long');const p=await getPage();await p.locator(selector(req.query.selector)).first().fill(value,{timeout:15000});res.json({ok:true});}catch(e){res.status(500).json({ok:false,error:String(e)});}});
app.get('/api/press',auth,async(req,res)=>{
  try{
    const key=String(req.query.key||'');
    if(!key || key.length>100) throw new Error('key required');
    const p=await getPage();
    await p.locator(selector(req.query.selector||'body')).first().press(key,{timeout:15000});
    res.json({ok:true,url:p.url()});
  }catch(e){res.status(500).json({ok:false,error:String(e)});}
});
app.get('/api/screenshot',auth,async(_req,res)=>{
  try{const p=await getPage();res.type('image/png').send(await p.screenshot({fullPage:false}));}
  catch(e){res.status(500).json({ok:false,error:String(e)});}
});
app.get('/api/close',auth,async(_req,res)=>{
  try{
    if(contextPromise){const c=await contextPromise;await c.close();contextPromise=null;}
    res.json({ok:true});
  }catch(e){res.status(500).json({ok:false,error:String(e)});}
});

const server=http.createServer(app);
server.on('upgrade',wsProxy.upgrade);
server.listen(PORT,'0.0.0.0',()=>console.log('Mari404 browser actuator listening on',PORT));
