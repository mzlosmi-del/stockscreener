"""
Stock Screener API — FastAPI + yfinance + pandas-ta
Frontend served at / and /app — no CORS issues.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

app = FastAPI(title="Stock Screener API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=6)

DEFAULT_TICKERS = [
    "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
    "JPM", "JNJ", "XOM", "UNH", "BAC", "HD", "PG", "CVX",
    "MRK", "ABBV", "KO", "PFE", "BA", "CAT", "GE", "AMD", "CRM", "NOW",
]

SECTOR_MAP = {
    "NVDA": "Technology", "AAPL": "Technology", "MSFT": "Technology",
    "META": "Technology", "GOOGL": "Technology", "AMZN": "Technology",
    "AMD": "Technology", "CRM": "Technology", "NOW": "Technology",
    "TSLA": "Consumer", "HD": "Consumer", "PG": "Consumer", "KO": "Consumer",
    "JPM": "Financials", "BAC": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "PFE": "Healthcare",
    "XOM": "Energy", "CVX": "Energy",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
}

FRONTEND_HTML = open("/app/frontend.html").read() if False else r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>US Market Scanner</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#fff;--bg2:#f5f5f4;--bg3:#eeede9;--txt:#1a1a1a;--txt2:#666;--txt3:#999;--border:rgba(0,0,0,0.1);--border2:rgba(0,0,0,0.18);--green:#2d7a3a;--red:#b03030;--r:8px;--rl:12px}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg3);color:var(--txt);min-height:100vh}
.wrap{max-width:1100px;margin:0 auto;padding:1.5rem 1rem}
.header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.5rem;flex-wrap:wrap;gap:10px}
.title{font-size:20px;font-weight:600}.subtitle{font-size:13px;color:var(--txt2);margin-top:3px}
.ldot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);margin-right:6px;animation:lp 2s ease-in-out infinite}
@keyframes lp{0%,100%{opacity:1}50%{opacity:.3}}
.disc{font-size:11px;color:var(--txt3);background:var(--bg2);border:.5px solid var(--border);border-radius:var(--r);padding:6px 10px}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:1.25rem}
select,button{font-size:13px;padding:7px 12px;border-radius:var(--r);border:.5px solid var(--border2);background:var(--bg);color:var(--txt);cursor:pointer}
select:hover,button:hover{background:var(--bg2)}
.btnp{background:var(--txt)!important;color:var(--bg)!important;border-color:var(--txt)!important}
.btnp:hover{opacity:.85}
button:disabled{opacity:.5;cursor:not-allowed}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:1.5rem}
.mc{background:var(--bg);border-radius:var(--r);padding:12px 14px;border:.5px solid var(--border)}
.ml{font-size:11px;color:var(--txt2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.mv{font-size:22px;font-weight:600}.mv.g{color:var(--green)}.mv.r{color:var(--red)}
.ms{font-size:11px;color:var(--txt3);margin-top:2px}
.card{background:var(--bg);border-radius:var(--rl);border:.5px solid var(--border);overflow:hidden;margin-bottom:1.5rem}
table{width:100%;border-collapse:collapse;font-size:13px;table-layout:fixed}
th{background:var(--bg2);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--txt2);padding:10px 12px;text-align:left;border-bottom:.5px solid var(--border);cursor:pointer;user-select:none}
th:hover{color:var(--txt)}
td{padding:11px 12px;border-bottom:.5px solid var(--border);vertical-align:middle}
tr:last-child td{border-bottom:none}
tbody tr{cursor:pointer;transition:background .1s}
tbody tr:hover td{background:var(--bg2)}
.pill{display:inline-block;font-size:11px;font-weight:500;padding:3px 9px;border-radius:20px}
.sb{background:#d0f0d8;color:#1a5c28}.b{background:#e5f5e9;color:#2d6e3a}
.h{background:#fef9e7;color:#7a6520}.s{background:#fde8e8;color:#9b2a2a}.ss{background:#fbd5d5;color:#7a1c1c}
.gn{color:var(--green);font-weight:500}.rd{color:var(--red);font-weight:500}
.bar{display:flex;gap:2px;align-items:center}
.seg{width:8px;height:8px;border-radius:1px;background:var(--border2)}
.seg.bull{background:var(--green)}.seg.bear{background:var(--red)}
.tabs{display:flex;border-bottom:.5px solid var(--border);margin-bottom:1.25rem}
.tab{font-size:13px;padding:8px 16px;color:var(--txt2);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.tab.active{color:var(--txt);border-bottom-color:var(--txt);font-weight:500}
.loading{text-align:center;padding:3rem;color:var(--txt2);font-size:14px}
.dot{display:inline-block;animation:pulse 1.2s ease-in-out infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
.ebox{background:#fdeaea;border:.5px solid #e8aaaa;border-radius:var(--r);padding:14px 16px;color:#7a1c1c;font-size:13px;margin-bottom:1rem}
.ph{background:var(--bg2);padding:12px 16px;border-bottom:.5px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.pb{padding:16px}
.dg{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.dl{font-size:11px;color:var(--txt2);margin-bottom:3px}
.dv{font-size:15px;font-weight:600}
.sbox{border-radius:var(--r);padding:12px 14px;margin-bottom:10px}
.sbox.buy{background:#e8f5ea;border:.5px solid #a8d5b0}
.sbox.sell{background:#fdeaea;border:.5px solid #e8aaaa}
.sbox-t{font-size:12px;font-weight:600;margin-bottom:6px}
.sbox.buy .sbox-t{color:#1a5c28}.sbox.sell .sbox-t{color:#7a1c1c}
.sbox p{font-size:13px;line-height:1.7}
.inds{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px}
.itag{font-size:11px;padding:3px 8px;border-radius:20px;border:.5px solid var(--border2)}
.itag.bull{border-color:#a8d5b0;background:#f0faf2;color:#1a5c28}
.itag.bear{border-color:#e8aaaa;background:#fdf5f5;color:#7a1c1c}
.itag.neut{background:var(--bg2);color:var(--txt2)}
.rbox{margin-top:16px;background:var(--bg2);border-radius:var(--r);padding:12px 14px}
.rg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;font-size:13px;margin-top:8px}
.back{font-size:13px;color:var(--txt2);cursor:pointer;margin-bottom:1rem;display:inline-block;border:none;background:none;padding:0}
.back:hover{color:var(--txt);background:none}
.cc{position:relative;height:180px;width:100%;margin:14px 0}
.spin{display:inline-block;animation:sp .8s linear infinite}
@keyframes sp{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.note{font-size:11px;color:var(--txt3);margin-top:10px}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <div class="title"><span class="ldot"></span>US Market Scanner</div>
      <div class="subtitle" id="ts">Connecting to live data...</div>
    </div>
    <div class="disc">15-min delayed · Yahoo Finance · Not financial advice</div>
  </div>
  <div id="main">
    <div class="controls">
      <select id="fsig"><option value="all">All signals</option><option value="strong-buy">Strong buy</option><option value="buy">Buy</option><option value="hold">Hold</option><option value="sell">Sell</option><option value="strong-sell">Strong sell</option></select>
      <select id="fsec"><option value="all">All sectors</option><option value="Technology">Technology</option><option value="Healthcare">Healthcare</option><option value="Financials">Financials</option><option value="Energy">Energy</option><option value="Consumer">Consumer</option><option value="Industrials">Industrials</option></select>
      <select id="fsort"><option value="score">Sort: Score</option><option value="rsi">Sort: RSI</option><option value="change">Sort: 1D chg</option><option value="volume">Sort: Volume</option></select>
      <button class="btnp" id="rbtn">Refresh</button>
    </div>
    <div class="metrics">
      <div class="mc"><div class="ml">Strong buys</div><div class="mv g" id="msb">-</div><div class="ms">stocks</div></div>
      <div class="mc"><div class="ml">Buy signals</div><div class="mv g" id="mb">-</div><div class="ms">stocks</div></div>
      <div class="mc"><div class="ml">Sell signals</div><div class="mv r" id="ms2">-</div><div class="ms">stocks</div></div>
      <div class="mc"><div class="ml">Avg RSI</div><div class="mv" id="mrsi">-</div><div class="ms">market</div></div>
      <div class="mc"><div class="ml">Market mood</div><div class="mv" id="mmood" style="font-size:15px">-</div><div class="ms" id="mmoods"></div></div>
    </div>
    <div class="tabs">
      <div class="tab active" id="t-screen" onclick="setTab('screen')">Screener</div>
      <div class="tab" id="t-watch" onclick="setTab('watch')">Watchlist</div>
    </div>
    <div id="tab-screen">
      <div id="loading" class="loading">Fetching live data<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span><div style="font-size:12px;margin-top:8px;color:#999">First load may take 15-20s</div></div>
      <div id="err" style="display:none"></div>
      <div id="tw" class="card" style="display:none">
        <table><thead><tr>
          <th style="width:13%">Ticker</th><th style="width:9%">Price</th><th style="width:9%">1D chg</th>
          <th style="width:13%">Signal</th><th style="width:16%">Score</th><th style="width:7%">RSI</th>
          <th style="width:11%">MACD</th><th style="width:8%">Vol</th><th style="width:14%">Sector</th>
        </tr></thead><tbody id="tbody"></tbody></table>
      </div>
    </div>
    <div id="tab-watch" style="display:none">
      <div id="we" style="text-align:center;padding:3rem;color:#999;font-size:14px">Click any stock then add to watchlist.</div>
      <div id="wt"></div>
    </div>
  </div>
  <div id="detail" style="display:none">
    <button class="back" id="backbtn">← Back to screener</button>
    <div id="dl" class="loading" style="display:none">Loading<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
    <div id="de" style="display:none"></div>
    <div id="dc" style="display:none">
      <div class="card">
        <div class="ph">
          <div><span style="font-size:15px;font-weight:600" id="dticker"></span><span style="font-size:13px;color:#666;margin-left:10px" id="dname"></span></div>
          <div style="display:flex;gap:8px;align-items:center"><span id="dpill"></span><button id="wlbtn">+ Watchlist</button></div>
        </div>
        <div class="pb">
          <div class="dg">
            <div><div class="dl">Price</div><div class="dv" id="dprice"></div></div>
            <div><div class="dl">1-day change</div><div class="dv" id="dchg"></div></div>
            <div><div class="dl">Signal score</div><div class="dv" id="dscore"></div></div>
            <div><div class="dl">ATR (14)</div><div class="dv" id="datr"></div></div>
          </div>
          <div class="dl" style="margin-top:14px;margin-bottom:4px">Indicator breakdown</div>
          <div class="inds" id="dinds"></div>
          <div class="cc"><canvas id="pc"></canvas></div>
          <div class="sbox buy"><div class="sbox-t">BUY - entry conditions</div><p id="dbuy"></p></div>
          <div class="sbox sell"><div class="sbox-t">SELL / exit conditions</div><p id="dsell"></p></div>
          <div class="rbox">
            <div class="dl">Risk management</div>
            <div class="rg">
              <div><div class="dl">Stop loss (2x ATR)</div><div style="font-weight:600;color:#b03030" id="dstop"></div></div>
              <div><div class="dl">Target (3x ATR)</div><div style="font-weight:600;color:#2d7a3a" id="dtgt"></div></div>
              <div><div class="dl">Risk / reward</div><div style="font-weight:600" id="drr"></div></div>
            </div>
          </div>
          <p class="note">Live data · Yahoo Finance · 15-min delayed · Not financial advice</p>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const API='';
let stocks=[],filtered=[],watchlist=new Set(),curTicker=null,chart=null;
const $=id=>document.getElementById(id);
const fmt=(n,d=2)=>n!=null?(+n).toFixed(d):'-';
const fmtP=n=>n!=null?((+n>=0?'+':'')+((+n).toFixed(2))+'%'):'-';
function pC(s){return{'strong-buy':'sb','buy':'b','hold':'h','sell':'s','strong-sell':'ss'}[s]||'h'}
function pL(s){return{'strong-buy':'Strong buy','buy':'Buy','hold':'Hold','sell':'Sell','strong-sell':'Strong sell'}[s]||s}
function sbar(sc,sig){let h='<div class="bar">';for(let i=1;i<=10;i++)h+=`<div class="seg ${i<=sc?(sig&&sig.includes('sell')?'bear':'bull'):''}"></div>`;return h+`</div><small style="color:#999;margin-top:2px;display:block">${sc}/10</small>`}
async function load(){
  const btn=$('rbtn');btn.innerHTML='<span class="spin">&#8635;</span> Loading';btn.disabled=true;
  $('loading').style.display='';$('tw').style.display='none';$('err').style.display='none';
  try{
    const res=await fetch(API+'/screen');
    if(!res.ok)throw new Error('Server error '+res.status);
    const data=await res.json();
    stocks=data.stocks||[];
    if(!stocks.length)throw new Error('No stocks returned - market may be closed');
    applyFilters();updateMetrics();
    $('loading').style.display='none';$('tw').style.display='';
    $('ts').textContent='Live · 15-min delayed · Updated '+new Date().toLocaleTimeString()+' · '+stocks.length+' stocks';
  }catch(e){
    $('loading').style.display='none';$('err').style.display='';
    $('err').innerHTML='<div class="ebox">Failed to load: '+e.message+'</div>';
  }
  btn.innerHTML='Refresh';btn.disabled=false;
}
function applyFilters(){
  const sig=$('fsig').value,sec=$('fsec').value,sort=$('fsort').value;
  filtered=[...stocks].filter(s=>(sig==='all'||s.signal===sig)&&(sec==='all'||s.sector===sec));
  if(sort==='score')filtered.sort((a,b)=>b.score-a.score);
  else if(sort==='rsi')filtered.sort((a,b)=>a.rsi-b.rsi);
  else if(sort==='change')filtered.sort((a,b)=>b.change-a.change);
  else if(sort==='volume')filtered.sort((a,b)=>b.vol_mult-a.vol_mult);
  renderTable();
}
function renderTable(){
  $('tbody').innerHTML=filtered.map(s=>`<tr onclick="showDetail('${s.ticker}')">
    <td><strong>${s.ticker}</strong><div style="font-size:11px;color:#999">${(s.name||'').split(' ').slice(0,2).join(' ')}</div></td>
    <td>$${fmt(s.price)}</td>
    <td class="${s.change>=0?'gn':'rd'}">${fmtP(s.change)}</td>
    <td><span class="pill ${pC(s.signal)}">${pL(s.signal)}</span></td>
    <td>${sbar(s.score,s.signal)}</td>
    <td style="color:${s.rsi>70?'#b03030':s.rsi<35?'#2d7a3a':'inherit'}">${fmt(s.rsi,1)}</td>
    <td class="${s.macd_val>=0?'gn':'rd'}">${s.macd_val>=0?'+':''}${fmt(s.macd_val,2)} <span style="font-size:10px;color:#999">${s.macd_dir||''}</span></td>
    <td>${fmt(s.vol_mult,2)}x</td>
    <td style="font-size:11px;color:#999">${s.sector||'-'}</td>
  </tr>`).join('');
}
function updateMetrics(){
  let sb=0,b=0,s=0,rsi=0;
  stocks.forEach(st=>{if(st.signal==='strong-buy')sb++;if(st.signal==='buy')b++;if(st.signal==='sell'||st.signal==='strong-sell')s++;rsi+=st.rsi;});
  $('msb').textContent=sb;$('mb').textContent=b;$('ms2').textContent=s;
  $('mrsi').textContent=fmt(rsi/stocks.length,1);
  const bull=sb+b,bear=s;
  $('mmood').textContent=bull>bear*1.5?'Bullish':bear>bull*1.5?'Bearish':'Mixed';
  $('mmoods').textContent=bull+' buys vs '+bear+' sells';
}
async function showDetail(ticker){
  curTicker=ticker;
  $('main').style.display='none';$('detail').style.display='';
  $('dl').style.display='';$('dc').style.display='none';$('de').style.display='none';
  if(chart){chart.destroy();chart=null;}
  try{
    const res=await fetch(API+'/stock/'+ticker);
    if(!res.ok)throw new Error('Server error '+res.status);
    const s=await res.json();
    $('dticker').textContent=s.ticker;
    $('dname').textContent=(s.name||'')+(s.sector?' · '+s.sector:'');
    $('dpill').innerHTML='<span class="pill '+pC(s.signal)+'">'+pL(s.signal)+'</span>';
    $('dprice').textContent='$'+fmt(s.price);
    $('dchg').innerHTML='<span class="'+(s.change>=0?'gn':'rd')+'">'+fmtP(s.change)+'</span>';
    $('dscore').textContent=(s.score||0)+'/10';
    $('datr').textContent='$'+fmt(s.atr);
    $('dstop').textContent='$'+fmt(s.stop);
    $('dtgt').textContent='$'+fmt(s.target);
    $('drr').textContent='1:'+fmt(s.risk_reward,1);
    $('wlbtn').textContent=watchlist.has(ticker)?'checkmark In watchlist':'+ Watchlist';
    $('dinds').innerHTML=[
      {l:'RSI '+fmt(s.rsi,1),c:s.rsi<35?'bull':s.rsi>70?'bear':'neut'},
      {l:'MACD '+(s.macd_val>=0?'+':'')+fmt(s.macd_val,2)+' '+s.macd_dir,c:s.macd_val>0?'bull':'bear'},
      {l:'EMA20 '+(s.pct_above_ema>=0?'+':'')+fmt(s.pct_above_ema,1)+'%',c:s.pct_above_ema>0?'bull':'bear'},
      {l:'Vol '+fmt(s.vol_mult,2)+'x',c:s.vol_mult>1.5?'bull':s.vol_mult<0.8?'bear':'neut'},
      {l:'BB '+(s.bb_pos*100).toFixed(0)+'%',c:s.bb_pos<0.2?'bull':s.bb_pos>0.8?'bear':'neut'},
    ].map(i=>'<span class="itag '+i.c+'">'+i.l+'</span>').join('');
    $('dbuy').innerHTML=(s.buy_points||[]).map(p=>'· '+p).join('<br>');
    $('dsell').innerHTML=(s.sell_points||[]).map(p=>'· '+p).join('<br>');
    const hist=s.history||[];
    if(hist.length>1){
      const color=s.change>=0?'#2d7a3a':'#b03030';
      const labels=hist.map((_,i)=>i===hist.length-1?'Today':'-'+(hist.length-1-i)+'d');
      chart=new Chart($('pc').getContext('2d'),{type:'line',
        data:{labels,datasets:[{data:hist,borderColor:color,backgroundColor:color+'22',borderWidth:1.5,pointRadius:0,fill:true,tension:0.3}]},
        options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'$'+c.raw.toFixed(2)}}},
        scales:{x:{grid:{display:false},ticks:{font:{size:10},color:'#999',maxRotation:0}},y:{grid:{color:'#eee'},ticks:{font:{size:10},color:'#999',callback:v=>'$'+v.toFixed(0)}}}}});
    }
    $('dl').style.display='none';$('dc').style.display='';
  }catch(e){
    $('dl').style.display='none';$('de').style.display='';
    $('de').innerHTML='<div class="ebox">Could not load '+ticker+': '+e.message+'</div>';
  }
}
function showMain(){$('main').style.display='';$('detail').style.display='none';if(chart){chart.destroy();chart=null;}}
function toggleWatchlist(){
  if(!curTicker)return;
  watchlist.has(curTicker)?watchlist.delete(curTicker):watchlist.add(curTicker);
  $('wlbtn').textContent=watchlist.has(curTicker)?'checkmark In watchlist':'+ Watchlist';
  renderWatchlist();
}
function renderWatchlist(){
  const wl=stocks.filter(s=>watchlist.has(s.ticker));
  if(!wl.length){$('we').style.display='';$('wt').innerHTML='';return;}
  $('we').style.display='none';
  $('wt').innerHTML='<div class="card"><table><thead><tr><th style="width:15%">Ticker</th><th style="width:12%">Price</th><th style="width:12%">1D chg</th><th style="width:15%">Signal</th><th style="width:18%">Score</th><th style="width:8%">RSI</th><th style="width:10%">Stop</th><th style="width:10%">Target</th></tr></thead><tbody>'+
  wl.map(s=>'<tr onclick="showDetail(\''+s.ticker+'\')"><td><strong>'+s.ticker+'</strong></td><td>$'+fmt(s.price)+'</td><td class="'+(s.change>=0?'gn':'rd')+'">'+fmtP(s.change)+'</td><td><span class="pill '+pC(s.signal)+'">'+pL(s.signal)+'</span></td><td>'+sbar(s.score,s.signal)+'</td><td>'+fmt(s.rsi,1)+'</td><td style="color:#b03030">$'+fmt(s.stop)+'</td><td style="color:#2d7a3a">$'+fmt(s.target)+'</td></tr>').join('')+
  '</tbody></table></div>';
}
function setTab(name){
  ['screen','watch'].forEach(t=>{$('t-'+t).classList.toggle('active',t===name);$('tab-'+t).style.display=t===name?'':'none';});
  if(name==='watch')renderWatchlist();
}
$('rbtn').addEventListener('click',load);
$('backbtn').addEventListener('click',showMain);
$('wlbtn').addEventListener('click',toggleWatchlist);
$('fsig').addEventListener('change',applyFilters);
$('fsec').addEventListener('change',applyFilters);
$('fsort').addEventListener('change',applyFilters);
load();
</script>
</body>
</html>"""


def compute_signal_score(rsi, macd_val, macd_dir, pct_above_ema, vol_mult, bb_pos, price_change):
    score = 0
    if rsi < 30: score += 2
    elif rsi < 45: score += 1
    elif rsi > 72: score -= 1
    if macd_val > 0 and macd_dir == "rising": score += 2
    elif macd_val > 0: score += 1
    elif macd_val < 0 and macd_dir == "falling": score -= 1
    if pct_above_ema > 3: score += 2
    elif pct_above_ema > 0: score += 1
    if vol_mult > 2.0 and price_change > 0: score += 2
    elif vol_mult > 1.5: score += 1
    if bb_pos < 0.15: score += 2
    elif bb_pos < 0.35: score += 1
    elif bb_pos > 0.85: score -= 1
    return max(0, min(10, score + 4))


def signal_label(score):
    if score >= 8: return "strong-buy"
    elif score >= 6: return "buy"
    elif score >= 4: return "hold"
    elif score >= 2: return "sell"
    return "strong-sell"


def fetch_and_analyze(ticker: str) -> dict:
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="60d", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 30:
            return None
        intra = tk.history(period="1d", interval="15m", auto_adjust=True)
        if not intra.empty:
            current_price = float(intra["Close"].iloc[-1])
            current_vol_today = float(intra["Volume"].sum())
        else:
            current_price = float(hist["Close"].iloc[-1])
            current_vol_today = float(hist["Volume"].iloc[-1])
        close = hist["Close"].astype(float)
        high = hist["High"].astype(float)
        low = hist["Low"].astype(float)
        volume = hist["Volume"].astype(float)
        rsi_series = ta.rsi(close, length=14)
        rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.isna().all() else 50.0
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            hist_col = [c for c in macd_df.columns if "MACDh" in c]
            macd_val = float(macd_df[hist_col[0]].iloc[-1]) if hist_col else 0.0
            prev_macd_val = float(macd_df[hist_col[0]].iloc[-2]) if hist_col and len(macd_df) > 1 else macd_val
            macd_dir = "rising" if macd_val > prev_macd_val else "falling"
        else:
            macd_val, macd_dir = 0.0, "flat"
        ema20_series = ta.ema(close, length=20)
        ema20 = float(ema20_series.iloc[-1]) if ema20_series is not None and not ema20_series.isna().all() else current_price
        pct_above_ema = (current_price - ema20) / ema20 * 100
        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            lower_col = [c for c in bb_df.columns if "BBL" in c]
            upper_col = [c for c in bb_df.columns if "BBU" in c]
            bb_lower = float(bb_df[lower_col[0]].iloc[-1]) if lower_col else current_price * 0.95
            bb_upper = float(bb_df[upper_col[0]].iloc[-1]) if upper_col else current_price * 1.05
            bb_range = bb_upper - bb_lower
            bb_pos = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        else:
            bb_lower, bb_upper, bb_pos = current_price * 0.95, current_price * 1.05, 0.5
        atr_series = ta.atr(high, low, close, length=14)
        atr = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.isna().all() else current_price * 0.015
        avg_vol_20 = float(volume.iloc[-21:-1].mean()) if len(volume) > 21 else float(volume.mean())
        vol_mult = (current_vol_today / avg_vol_20) if avg_vol_20 > 0 else 1.0
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        price_change = (current_price - prev_close) / prev_close * 100
        history_20 = [round(float(p), 2) for p in close.iloc[-20:].tolist()]
        score = compute_signal_score(rsi, macd_val, macd_dir, pct_above_ema, vol_mult, bb_pos, price_change)
        signal = signal_label(score)
        buy_points, sell_points = [], []
        if signal in ("strong-buy", "buy"):
            buy_points.append(f"Entry near current price ${current_price:.2f}")
            if rsi < 45: buy_points.append(f"RSI at {rsi:.1f} — oversold conditions support entry")
            if macd_val > 0 and macd_dir == "rising": buy_points.append("MACD histogram positive and rising — bullish momentum confirmed")
            if pct_above_ema > 0: buy_points.append(f"Price {pct_above_ema:.1f}% above EMA20 (${ema20:.2f}) — uptrend intact")
            if vol_mult > 1.5: buy_points.append(f"Volume {vol_mult:.1f}x 20-day average — strong participation")
            if bb_pos < 0.3: buy_points.append("Price near lower Bollinger Band — mean-reversion setup")
        else:
            buy_points.append(f"Wait — RSI at {rsi:.1f}, look for drop below 40 before entering")
            buy_points.append(f"Watch for price reclaim above EMA20 (${ema20:.2f})")
            if macd_val < 0: buy_points.append("Wait for MACD histogram to cross back into positive territory")
        stop = current_price - atr * 2
        target = current_price + atr * 3
        rr = round(atr * 3 / (atr * 2), 1)
        sell_points.append(f"Stop loss: ${stop:.2f} (2x ATR below entry) — exit immediately if breached")
        sell_points.append(f"Primary target: ${target:.2f} (3x ATR above entry) — take full profit here")
        if rsi > 68: sell_points.append(f"RSI at {rsi:.1f} — reduce size or wait for pullback before entry")
        if macd_val > 0 and macd_dir == "falling": sell_points.append("MACD flattening — momentum weakening, tighten stop to breakeven")
        if bb_pos > 0.8: sell_points.append("Price near upper Bollinger Band — consider partial profit taking (50%)")
        sell_points.append(f"Also exit on daily close below EMA20 (${ema20:.2f}) or gap-down open >2%")
        return {
            "ticker": ticker,
            "name": tk.info.get("shortName", ticker) if hasattr(tk, "info") else ticker,
            "sector": SECTOR_MAP.get(ticker, tk.info.get("sector", "Unknown") if hasattr(tk, "info") else "Unknown"),
            "price": round(current_price, 2), "change": round(price_change, 2),
            "rsi": round(rsi, 1), "macd_val": round(macd_val, 3), "macd_dir": macd_dir,
            "ema20": round(ema20, 2), "pct_above_ema": round(pct_above_ema, 2),
            "bb_lower": round(bb_lower, 2), "bb_upper": round(bb_upper, 2), "bb_pos": round(bb_pos, 3),
            "atr": round(atr, 2), "vol_mult": round(vol_mult, 2),
            "score": score, "signal": signal,
            "stop": round(stop, 2), "target": round(target, 2), "risk_reward": rr,
            "history": history_20, "buy_points": buy_points, "sell_points": sell_points,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_note": "15-min delayed via Yahoo Finance",
        }
    except Exception:
        print(f"Error fetching {ticker}: {traceback.format_exc()}")
        return None


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=FRONTEND_HTML)

@app.get("/app", response_class=HTMLResponse)
def frontend():
    return HTMLResponse(content=FRONTEND_HTML)

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/screen")
async def screen(tickers: str = Query(default=None)):
    ticker_list = [t.strip().upper() for t in tickers.split(",")] if tickers else DEFAULT_TICKERS
    if len(ticker_list) > 50:
        raise HTTPException(status_code=400, detail="Max 50 tickers per request")
    loop = asyncio.get_event_loop()
    results = await asyncio.gather(*[loop.run_in_executor(executor, fetch_and_analyze, t) for t in ticker_list])
    stocks = sorted([r for r in results if r], key=lambda s: s["score"], reverse=True)
    return {"stocks": stocks, "count": len(stocks), "screened": len(ticker_list),
            "failed": len(ticker_list) - len(stocks),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_note": "15-min delayed intraday via Yahoo Finance (free, no API key required)"}

@app.get("/stock/{ticker}")
async def single_stock(ticker: str):
    result = await asyncio.get_event_loop().run_in_executor(executor, fetch_and_analyze, ticker.upper())
    if result is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {ticker}")
    return result
