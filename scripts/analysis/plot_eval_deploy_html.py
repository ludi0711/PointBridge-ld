#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 eval 与 deploy 日志的 HTML 曲线对比。"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

DEFAULT_LOG_DIR = Path("test_log/xarm7_pick_pose-2026-05-22_20-44-56")
DEFAULT_EVAL_CSV = DEFAULT_LOG_DIR / "eval_pick_pose_2026-05-22_20-44-56.csv"
DEFAULT_DEPLOY_CSV = DEFAULT_LOG_DIR / "sim2real_deploy_state_pose_2026-05-22_20-44-56_run04.csv"
DEFAULT_OUTPUT = DEFAULT_LOG_DIR / "eval_vs_sim2real_deploy_curves.html"


def _to_float(value: str | None, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default



def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = sum(v * v for v in q) ** 0.5
    if norm < 1.0e-12:
        raise ValueError("quaternion norm is too small")
    return tuple(v / norm for v in q)


def _quat_apply(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = _quat_normalize(q)
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _eval_tcp_source_pos(
    row: dict[str, str],
    tcp_pos: tuple[float, float, float],
    eval_tcp_frame: str,
    eval_tcp_offset_m: float,
) -> tuple[float, float, float] | None:
    if eval_tcp_frame == "source":
        return tcp_pos

    qw = _to_float(row.get("tcp_qw"))
    qx = _to_float(row.get("tcp_qx"))
    qy = _to_float(row.get("tcp_qy"))
    qz = _to_float(row.get("tcp_qz"))
    if qw is None or qx is None or qy is None or qz is None:
        return None

    offset = _quat_apply((qw, qx, qy, qz), (0.0, 0.0, eval_tcp_offset_m))
    return (tcp_pos[0] - offset[0], tcp_pos[1] - offset[1], tcp_pos[2] - offset[2])


def _distance_sq(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((av - bv) ** 2 for av, bv in zip(a, b))


def _resolve_eval_tcp_frame(
    row: dict[str, str],
    tcp_pos: tuple[float, float, float],
    eval_tcp_offset_m: float,
    deploy_ref_tcp: tuple[float, float, float] | None,
) -> str:
    if deploy_ref_tcp is None:
        return "source"
    corrected = _eval_tcp_source_pos(row, tcp_pos, "target-offset", eval_tcp_offset_m)
    if corrected is None:
        return "source"
    raw_dist = _distance_sq(tcp_pos, deploy_ref_tcp)
    corrected_dist = _distance_sq(corrected, deploy_ref_tcp)
    return "target-offset" if corrected_dist < raw_dist else "source"


def _read_eval_csv(
    path: Path,
    episode: int,
    eval_tcp_frame: str,
    eval_tcp_offset_m: float,
    deploy_ref_tcp: tuple[float, float, float] | None = None,
) -> tuple[list[dict[str, float]], str]:
    rows: list[dict[str, float]] = []
    resolved_tcp_frame: str | None = None if eval_tcp_frame == "auto" else eval_tcp_frame
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row_episode = int(_to_float(row.get("episode"), 0) or 0)
            if row_episode != episode:
                continue
            step = _to_float(row.get("step"), float(i))
            tcp_x = _to_float(row.get("tcp_x_m"))
            tcp_y = _to_float(row.get("tcp_y_m"))
            tcp_z = _to_float(row.get("tcp_z_m"))
            if tcp_x is None or tcp_y is None or tcp_z is None:
                continue
            tcp_pos = (tcp_x, tcp_y, tcp_z)
            if resolved_tcp_frame is None:
                resolved_tcp_frame = _resolve_eval_tcp_frame(row, tcp_pos, eval_tcp_offset_m, deploy_ref_tcp)
            tcp_source_pos = _eval_tcp_source_pos(row, tcp_pos, resolved_tcp_frame, eval_tcp_offset_m)
            if tcp_source_pos is None:
                continue
            values = {
                "dist_cm": _to_float(row.get("dist_cm")),
                "ori_deg": _to_float(row.get("ori_err_deg")),
                "tcp_x_m": tcp_source_pos[0],
                "tcp_y_m": tcp_source_pos[1],
                "tcp_z_m": tcp_source_pos[2],
                "obj_x_m": _to_float(row.get("object_x_m")),
                "obj_y_m": _to_float(row.get("object_y_m")),
                "obj_z_m": _to_float(row.get("object_z_m")),
            }
            if step is None or any(v is None for v in values.values()):
                continue
            rows.append({"step": step, **values})
    return rows, resolved_tcp_frame or "source"


def _read_deploy_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            step = _to_float(row.get("step"), float(i))
            dist_mm = _to_float(row.get("obj_rel_dist_mm"))
            ee_x_m = (_to_float(row.get("ee_x_mm")) or 0.0) / 1000.0
            ee_y_m = (_to_float(row.get("ee_y_mm")) or 0.0) / 1000.0
            ee_z_m = (_to_float(row.get("ee_z_mm")) or 0.0) / 1000.0
            rel_x_m = (_to_float(row.get("obj_rel_x_mm")) or 0.0) / 1000.0
            rel_y_m = (_to_float(row.get("obj_rel_y_mm")) or 0.0) / 1000.0
            rel_z_m = (_to_float(row.get("obj_rel_z_mm")) or 0.0) / 1000.0
            ee_quat = (
                _to_float(row.get("ee_qw")) or 1.0,
                _to_float(row.get("ee_qx")) or 0.0,
                _to_float(row.get("ee_qy")) or 0.0,
                _to_float(row.get("ee_qz")) or 0.0,
            )
            rel_base = _quat_apply(ee_quat, (rel_x_m, rel_y_m, rel_z_m))
            values = {
                "ori_deg": _to_float(row.get("obj_rel_ori_err_deg")),
                "tcp_x_m": ee_x_m,
                "tcp_y_m": ee_y_m,
                "tcp_z_m": ee_z_m,
                "obj_x_m": ee_x_m + rel_base[0],
                "obj_y_m": ee_y_m + rel_base[1],
                "obj_z_m": ee_z_m + rel_base[2],
            }
            if step is None or dist_mm is None or any(v is None for v in values.values()):
                continue
            rows.append({"step": step, "dist_cm": dist_mm / 10.0, **values})
    return rows


def _build_html(
    eval_csv: Path,
    deploy_csv: Path,
    eval_rows: list[dict[str, float]],
    deploy_rows: list[dict[str, float]],
    episode: int,
    eval_tcp_frame: str,
    eval_tcp_offset_m: float,
) -> str:
    payload = json.dumps(
        {
            "eval": eval_rows,
            "deploy": deploy_rows,
            "episode": episode,
            "evalTcpFrame": eval_tcp_frame,
            "evalTcpOffsetM": eval_tcp_offset_m,
        },
        ensure_ascii=False,
    )
    title = "Eval vs Sim2Real Deploy Curves"
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light; --bg:#f7f7f4; --panel:#fff; --text:#202124; --muted:#62656a; --grid:#deded8; --eval:#0f766e; --deploy:#b45309; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1180px; margin:0 auto; padding:28px 18px 36px; } h1 { margin:0 0 8px; font-size:24px; font-weight:700; letter-spacing:0; }
.meta { display:grid; gap:4px; margin-bottom:20px; color:var(--muted); word-break:break-all; } .toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px; margin-bottom:18px; color:var(--muted); }
label { display:inline-flex; align-items:center; gap:6px; cursor:pointer; } input[type="checkbox"] { width:16px; height:16px; accent-color:var(--eval); }
.legend { display:inline-flex; align-items:center; gap:8px; margin-right:12px; color:var(--text); } .swatch { width:18px; height:3px; border-radius:999px; background:currentColor; } .eval { color:var(--eval); } .deploy { color:var(--deploy); }
.chart { background:var(--panel); border:1px solid #e5e2da; border-radius:8px; padding:14px 14px 8px; margin-top:16px; box-shadow:0 1px 2px rgba(0,0,0,.04); } .chart h2 { margin:0 0 8px; font-size:16px; font-weight:650; letter-spacing:0; }
svg { display:block; width:100%; height:360px; overflow:visible; } canvas { display:block; width:100%; height:500px; cursor:grab; touch-action:none; } canvas:active { cursor:grabbing; }
.axis,.grid { stroke:var(--grid); stroke-width:1; shape-rendering:crispEdges; } .axis-label,.tick { fill:var(--muted); font-size:12px; } .line { fill:none; stroke-width:2.2; stroke-linejoin:round; stroke-linecap:round; } .empty { fill:var(--muted); font-size:14px; }
@media (max-width:700px) { main { padding:20px 12px 28px; } svg { height:300px; } canvas { height:360px; } }
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="meta"><div>eval: __EVAL_CSV__</div><div>deploy: __DEPLOY_CSV__</div><div>eval episode: __EPISODE__</div><div>eval tcp frame: __EVAL_TCP_FRAME__</div><div>3D frame: base_link origin</div></div>
<div class="toolbar"><span class="legend eval"><span class="swatch"></span>eval</span><span class="legend deploy"><span class="swatch"></span>sim2real deploy</span><label><input id="showEval" type="checkbox" checked>显示 eval</label><label><input id="showDeploy" type="checkbox" checked>显示 deploy</label></div>
<section class="chart"><h2>距离误差 dist / obj_rel_dist（cm）</h2><svg id="distChart"></svg></section>
<section class="chart"><h2>姿态误差 ori_err / obj_rel_ori_err（deg）</h2><svg id="oriChart"></svg></section>
<section class="chart"><h2>TCP / EE 与目标位置三维轨迹 XYZ（base_link, m）</h2><canvas id="tcpObject3dChart"></canvas></section>
</main><script>
const data = __PAYLOAD__;
const colors = { eval:getComputedStyle(document.documentElement).getPropertyValue('--eval').trim(), deploy:getComputedStyle(document.documentElement).getPropertyValue('--deploy').trim() };
function niceTicks(min,max,count){ if(!Number.isFinite(min)||!Number.isFinite(max)||min===max) return [min||0,(min||0)+1]; const span=max-min, raw=span/Math.max(1,count-1), pow=Math.pow(10,Math.floor(Math.log10(raw))); const step=[1,2,5,10].find(v=>v*pow>=raw)*pow; const start=Math.floor(min/step)*step, end=Math.ceil(max/step)*step, ticks=[]; for(let v=start; v<=end+step*.5; v+=step) ticks.push(v); return ticks; }
function activeSeries(){ const s=[]; if(document.getElementById('showEval').checked) s.push({name:'eval',rows:data.eval}); if(document.getElementById('showDeploy').checked) s.push({name:'deploy',rows:data.deploy}); return s; }
function linePath(rows,xField,yField,xScale,yScale){ return rows.map((d,i)=>`${i===0?'M':'L'} ${xScale(d[xField]).toFixed(2)} ${yScale(d[yField]).toFixed(2)}`).join(' '); }
function drawAxes(svg,width,height,margin,xMin,xMax,yMin,yMax,xLabel,yLabel,xDecimals,yDecimals){ const plotW=width-margin.left-margin.right, plotH=height-margin.top-margin.bottom; const xScale=x=>margin.left+((x-xMin)/Math.max(1e-9,xMax-xMin))*plotW; const yScale=y=>margin.top+(1-((y-yMin)/Math.max(1e-9,yMax-yMin)))*plotH; for(const y of niceTicks(yMin,yMax,6)){ const yy=yScale(y); svg.insertAdjacentHTML('beforeend',`<line class="grid" x1="${margin.left}" y1="${yy}" x2="${width-margin.right}" y2="${yy}"></line><text class="tick" x="${margin.left-8}" y="${yy+4}" text-anchor="end">${Number(y.toFixed(yDecimals)).toString()}</text>`); } for(const x of niceTicks(xMin,xMax,7)){ const xx=xScale(x); svg.insertAdjacentHTML('beforeend',`<line class="grid" x1="${xx}" y1="${margin.top}" x2="${xx}" y2="${height-margin.bottom}"></line><text class="tick" x="${xx}" y="${height-margin.bottom+20}" text-anchor="middle">${Number(x.toFixed(xDecimals)).toString()}</text>`); } svg.insertAdjacentHTML('beforeend',`<line class="axis" x1="${margin.left}" y1="${height-margin.bottom}" x2="${width-margin.right}" y2="${height-margin.bottom}"></line><line class="axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height-margin.bottom}"></line><text class="axis-label" x="${width/2}" y="${height-8}" text-anchor="middle">${xLabel}</text><text class="axis-label" transform="translate(16 ${height/2}) rotate(-90)" text-anchor="middle">${yLabel}</text>`); return {xScale,yScale}; }
function drawTimeChart(svgId,field,yLabel){ const svg=document.getElementById(svgId); svg.textContent=''; const width=svg.clientWidth||900, height=svg.clientHeight||360, margin={top:18,right:20,bottom:42,left:58}; svg.setAttribute('viewBox',`0 0 ${width} ${height}`); const series=activeSeries(), all=series.flatMap(s=>s.rows); if(!all.length){ svg.insertAdjacentHTML('beforeend',`<text class="empty" x="${margin.left}" y="${height/2}">没有可绘制数据</text>`); return; } const xMin=Math.min(...all.map(d=>d.step)), xMax=Math.max(...all.map(d=>d.step)); const yMinRaw=Math.min(...all.map(d=>d[field])), yMaxRaw=Math.max(...all.map(d=>d[field])); const yPad=Math.max((yMaxRaw-yMinRaw)*.08,.5), yMin=Math.max(0,yMinRaw-yPad), yMax=yMaxRaw+yPad; const scales=drawAxes(svg,width,height,margin,xMin,xMax,yMin,yMax,'step',yLabel,0,3); for(const s of series) if(s.rows.length) svg.insertAdjacentHTML('beforeend',`<path class="line" d="${linePath(s.rows,'step',field,scales.xScale,scales.yScale)}" stroke="${colors[s.name]}"></path>`); }
const viewState = { tcpObject3dChart: { yaw: -0.75, pitch: 0.55, zoom: 1.0, pointer: null } };
function setup3dCanvas(id){
  const canvas=document.getElementById(id); let dragging=false,lastX=0,lastY=0;
  function setPointer(e){ const r=canvas.getBoundingClientRect(); viewState[id].pointer={x:e.clientX-r.left,y:e.clientY-r.top}; }
  canvas.addEventListener('pointerdown',e=>{dragging=true; lastX=e.clientX; lastY=e.clientY; setPointer(e); canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{ const st=viewState[id]; setPointer(e); if(dragging){ st.yaw+=(e.clientX-lastX)*0.01; st.pitch=Math.max(-1.45,Math.min(1.45,st.pitch+(e.clientY-lastY)*0.01)); lastX=e.clientX; lastY=e.clientY; } drawAll(); });
  canvas.addEventListener('pointerup',e=>{dragging=false; canvas.releasePointerCapture(e.pointerId);});
  canvas.addEventListener('pointercancel',()=>{dragging=false;});
  canvas.addEventListener('pointerleave',()=>{dragging=false; viewState[id].pointer=null; drawAll();});
  canvas.addEventListener('wheel',e=>{ e.preventDefault(); const st=viewState[id]; st.zoom=Math.max(.35,Math.min(4,st.zoom*(e.deltaY<0?1.12:.89))); drawAll(); },{passive:false});
}
function project3d(point,center,scale,st,cx,cy){ const x=(point.x-center.x)*scale,y=(point.y-center.y)*scale,z=(point.z-center.z)*scale; const cyaw=Math.cos(st.yaw),syaw=Math.sin(st.yaw),cp=Math.cos(st.pitch),sp=Math.sin(st.pitch); const x1=x*cyaw-y*syaw, y1=x*syaw+y*cyaw, y2=y1*cp-z*sp, z2=y1*sp+z*cp; return {x:cx+x1,y:cy-z2,depth:y2}; }
function niceAxisLength(v){ if(!Number.isFinite(v)||v<=0) return .1; const p=Math.pow(10,Math.floor(Math.log10(v))); return [1,2,5,10].find(n=>n*p>=v)*p; }
function drawArrow(ctx,a,b,color,label){
  const ang=Math.atan2(b.y-a.y,b.x-a.x), len=9;
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=2.2; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(b.x,b.y); ctx.lineTo(b.x-len*Math.cos(ang-.45),b.y-len*Math.sin(ang-.45)); ctx.lineTo(b.x-len*Math.cos(ang+.45),b.y-len*Math.sin(ang+.45)); ctx.closePath(); ctx.fill();
  ctx.fillText(label,b.x+7,b.y-5);
}
function drawTooltip(ctx,hit,rect){
  const lines=[`${hit.source} ${hit.kind}`,`step: ${hit.step}`,`x: ${hit.x.toFixed(4)} m`,`y: ${hit.y.toFixed(4)} m`,`z: ${hit.z.toFixed(4)} m`];
  ctx.font='12px system-ui, sans-serif'; const w=Math.max(...lines.map(t=>ctx.measureText(t).width))+18, h=lines.length*17+12;
  let x=hit.screen.x+14,y=hit.screen.y+14; if(x+w>rect.width-8) x=hit.screen.x-w-14; if(y+h>rect.height-8) y=hit.screen.y-h-14;
  ctx.fillStyle='rgba(255,255,255,.96)'; ctx.strokeStyle='#d8d4c9'; ctx.lineWidth=1; ctx.beginPath(); ctx.roundRect(x,y,w,h,7); ctx.fill(); ctx.stroke();
  ctx.fillStyle='#202124'; lines.forEach((t,i)=>ctx.fillText(t,x+9,y+20+i*17));
}
function draw3dChart(canvasId){
  const canvas=document.getElementById(canvasId), rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.floor(rect.width*dpr)); canvas.height=Math.max(1,Math.floor(rect.height*dpr));
  const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,rect.width,rect.height);
  const paths=[{kind:'tcp',x:'tcp_x_m',y:'tcp_y_m',z:'tcp_z_m'},{kind:'object',x:'obj_x_m',y:'obj_y_m',z:'obj_z_m'}];
  const series=[];
  for(const src of activeSeries()) for(const path of paths) series.push({source:src.name,kind:path.kind,color:colors[src.name],rows:src.rows.map(r=>({step:r.step,x:r[path.x],y:r[path.y],z:r[path.z]}))});
  const nonEmptySeries=series.filter(s=>s.rows.length);
  const dataPoints=nonEmptySeries.flatMap(s=>s.rows), baseOrigin={x:0,y:0,z:0};
  ctx.font='12px system-ui, sans-serif'; if(!dataPoints.length){ctx.fillStyle='#62656a'; ctx.fillText('没有可绘制数据',24,rect.height/2); return;}
  const dataWithOrigin=dataPoints.concat([baseOrigin]);
  const dataMin={x:Math.min(...dataWithOrigin.map(p=>p.x)),y:Math.min(...dataWithOrigin.map(p=>p.y)),z:Math.min(...dataWithOrigin.map(p=>p.z))};
  const dataMax={x:Math.max(...dataWithOrigin.map(p=>p.x)),y:Math.max(...dataWithOrigin.map(p=>p.y)),z:Math.max(...dataWithOrigin.map(p=>p.z))};
  const dataSpan=Math.max(dataMax.x-dataMin.x,dataMax.y-dataMin.y,dataMax.z-dataMin.z,1e-6);
  const axisLen=niceAxisLength(Math.max(dataSpan*.45,.05));
  const fitPoints=dataWithOrigin.concat([{x:axisLen,y:0,z:0},{x:0,y:axisLen,z:0},{x:0,y:0,z:axisLen}]);
  const min={x:Math.min(...fitPoints.map(p=>p.x)),y:Math.min(...fitPoints.map(p=>p.y)),z:Math.min(...fitPoints.map(p=>p.z))};
  const max={x:Math.max(...fitPoints.map(p=>p.x)),y:Math.max(...fitPoints.map(p=>p.y)),z:Math.max(...fitPoints.map(p=>p.z))};
  const pad=dataSpan*.16, center={x:(min.x+max.x)/2,y:(min.y+max.y)/2,z:(min.z+max.z)/2};
  const span=Math.max(max.x-min.x,max.y-min.y,max.z-min.z,1e-6)+pad*2;
  const st=viewState[canvasId], scale=Math.min(rect.width,rect.height)*.62*st.zoom/span, cx=rect.width/2, cy=rect.height/2+18;
  ctx.fillStyle='#62656a'; ctx.fillText('拖动旋转视角，滚轮缩放；坐标轴原点为 base_link (0,0,0)',14,22); ctx.fillText('实线: TCP/EE，虚线: 目标位置；悬停显示采样点坐标',14,40);
  const origin=project3d(baseOrigin,center,scale,st,cx,cy);
  ctx.fillStyle='#202124'; ctx.beginPath(); ctx.arc(origin.x,origin.y,4,0,Math.PI*2); ctx.fill(); ctx.fillText('base_link',origin.x+7,origin.y+14);
  for(const axis of [{label:'+X / m',color:'#d32f2f',end:{x:axisLen,y:0,z:0}},{label:'+Y / m',color:'#2e7d32',end:{x:0,y:axisLen,z:0}},{label:'+Z / m',color:'#1565c0',end:{x:0,y:0,z:axisLen}}]) drawArrow(ctx,origin,project3d(axis.end,center,scale,st,cx,cy),axis.color,axis.label);
  const ordered=nonEmptySeries.map(s=>({...s,depth:s.rows.reduce((acc,p)=>acc+project3d(p,center,scale,st,cx,cy).depth,0)/s.rows.length})).sort((a,b)=>a.depth-b.depth);
  let hit=null, hitD2=12*12;
  for(const s of ordered){
    const pts=s.rows.map(p=>({...p,screen:project3d(p,center,scale,st,cx,cy)}));
    ctx.strokeStyle=s.color; ctx.lineWidth=s.kind==='object'?2:2.6; ctx.setLineDash(s.kind==='object'?[8,5]:[]);
    ctx.beginPath(); pts.forEach((p,i)=>i?ctx.lineTo(p.screen.x,p.screen.y):ctx.moveTo(p.screen.x,p.screen.y)); ctx.stroke(); ctx.setLineDash([]);
    const a=pts[0],b=pts[pts.length-1]; ctx.fillStyle=s.color; ctx.beginPath(); ctx.arc(a.screen.x,a.screen.y,4.5,0,Math.PI*2); ctx.fill(); ctx.fillRect(b.screen.x-4.5,b.screen.y-4.5,9,9);
    if(st.pointer) for(const p of pts){ const d2=(p.screen.x-st.pointer.x)**2+(p.screen.y-st.pointer.y)**2; if(d2<hitD2){ hitD2=d2; hit={source:s.source,kind:s.kind,step:p.step,x:p.x,y:p.y,z:p.z,screen:p.screen,color:s.color}; } }
  }
  let lx=rect.width-190,ly=22; for(const [label,color,dashed] of [['eval TCP/EE',colors.eval,false],['eval object',colors.eval,true],['deploy TCP/EE',colors.deploy,false],['deploy object',colors.deploy,true]]){ ctx.strokeStyle=color; ctx.lineWidth=2.4; ctx.setLineDash(dashed?[8,5]:[]); ctx.beginPath(); ctx.moveTo(lx,ly-4); ctx.lineTo(lx+28,ly-4); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle='#62656a'; ctx.fillText(label,lx+36,ly); ly+=18; }
  ctx.fillStyle='#62656a'; ctx.fillText(`x: ${dataMin.x.toFixed(3)} .. ${dataMax.x.toFixed(3)} m`,14,rect.height-48); ctx.fillText(`y: ${dataMin.y.toFixed(3)} .. ${dataMax.y.toFixed(3)} m`,14,rect.height-30); ctx.fillText(`z: ${dataMin.z.toFixed(3)} .. ${dataMax.z.toFixed(3)} m`,14,rect.height-12);
  if(hit){ ctx.strokeStyle=hit.color; ctx.lineWidth=2; ctx.setLineDash([]); ctx.beginPath(); ctx.arc(hit.screen.x,hit.screen.y,7,0,Math.PI*2); ctx.stroke(); drawTooltip(ctx,hit,rect); }
}
function drawAll(){ drawTimeChart('distChart','dist_cm','cm'); drawTimeChart('oriChart','ori_deg','deg'); draw3dChart('tcpObject3dChart'); }
setup3dCanvas('tcpObject3dChart'); document.getElementById('showEval').addEventListener('change',drawAll); document.getElementById('showDeploy').addEventListener('change',drawAll); window.addEventListener('resize',drawAll); drawAll();
</script></body></html>
'''
    return (
        template
        .replace("__TITLE__", html.escape(title))
        .replace("__EVAL_CSV__", html.escape(str(eval_csv)))
        .replace("__DEPLOY_CSV__", html.escape(str(deploy_csv)))
        .replace("__EPISODE__", str(episode))
        .replace("__EVAL_TCP_FRAME__", html.escape(eval_tcp_frame))
        .replace("__PAYLOAD__", payload)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--deploy_csv", type=Path, default=DEFAULT_DEPLOY_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode", type=int, default=0, help="只绘制 eval CSV 中的指定 episode")
    parser.add_argument(
        "--eval_tcp_frame",
        choices=("auto", "source", "target-offset"),
        default="auto",
        help=(
            "eval CSV 里的 tcp_* 坐标兼容模式。source 表示直接使用 CSV 原始 tcp_*；"
            "target-offset 表示减去 0.177 m target offset 来兼容旧对比；auto 会比较首帧自动判断。"
        ),
    )
    parser.add_argument(
        "--eval_tcp_offset_m",
        type=float,
        default=0.177,
        help="--eval_tcp_frame target-offset 时使用的 target 相对 source 的 z 向 offset，单位 m",
    )
    args = parser.parse_args()

    deploy_rows = _read_deploy_csv(args.deploy_csv)
    deploy_ref_tcp = None
    if deploy_rows:
        first_deploy = deploy_rows[0]
        deploy_ref_tcp = (first_deploy["tcp_x_m"], first_deploy["tcp_y_m"], first_deploy["tcp_z_m"])
    eval_rows, resolved_eval_tcp_frame = _read_eval_csv(
        args.eval_csv,
        args.episode,
        args.eval_tcp_frame,
        args.eval_tcp_offset_m,
        deploy_ref_tcp,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _build_html(
            args.eval_csv,
            args.deploy_csv,
            eval_rows,
            deploy_rows,
            args.episode,
            resolved_eval_tcp_frame,
            args.eval_tcp_offset_m,
        ),
        encoding="utf-8",
    )
    print(f"[HTML] {args.output}")
    print(f"[Rows] eval_episode_{args.episode}={len(eval_rows)} deploy={len(deploy_rows)}")
    print(f"[Eval TCP frame] {args.eval_tcp_frame} -> {resolved_eval_tcp_frame}")


if __name__ == "__main__":
    main()
