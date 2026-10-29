"""S16 geometry comparison plots; no new world elevation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mythic.tectonics.geometry import unit
from mythic.tectonics.source_clipping import LocalGnomonic
root=Path(__file__).resolve().parents[1];cache=root/'.cache/u36_s16'
q=json.loads((cache/'geometry.json').read_text(encoding='utf8'))
a=np.load(cache/'snapshots.npz');v=a['vertices'];tri=a['triangles']
p=LocalGnomonic(unit(v[tri].reshape(-1,3).mean(axis=0)))
pts=a['140'];off=a['shared_offsets'];indices=a['shared_indices'];old=a['old_indices']
bounds=[indices[off[i]:off[i+1]] for i in range(len(tri))]
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,(ax,bx)=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
for i in range(len(tri)):
    xy=p.forward(pts[bounds[i]])
    ax.plot(*np.vstack([xy,xy[0]]).T,color='#4a7e85',lw=.25,alpha=.65)
for i in [341,1659,1083]:
    xy=p.forward(pts[bounds[i]])
    ax.plot(*np.vstack([xy,xy[0]]).T,lw=1.7,label=str(i))
ax.set_aspect('equal');ax.set_xlabel('心射坐标 x / km');ax.set_ylabel('心射坐标 y / km')
ax.set_title('0 Ma 共享边材料单元\n颜色标出 S15 的两组悬挂边例子')
ax.legend(title='单元编号',fontsize=8)
rows=q['snapshots'];x=np.arange(4);ind=[r for r in rows if r['method']=='independent'];shared=[r for r in rows if r['method']=='shared']
bx.bar(x-.18,[r['pair_overlap_km2'] for r in ind],width=.36,label='各单元独立采样')
bx.bar(x+.18,[r['pair_overlap_km2'] for r in shared],width=.36,label='共享子边采样')
bx.set_xticks(x,[str(int(r['time_ma'])) for r in ind]);bx.set_xlabel('时间 / Ma');bx.set_ylabel('逐对正面积重叠之和 / km²')
bx.set_title('四个离散时点的材料重叠');bx.legend()
for i,r in enumerate(shared):
    if r['positive_pair_count']==0:bx.annotate('0', (i+.18,0),xytext=(0,4),textcoords='offset points',ha='center')
fig.suptitle('U36-S16：共同边界几何小试 · procedural_constrained / interpolated\n只读局部审计，未更新世界高程；输入物理分辨率约 60 km',fontsize=12)
out=root/'figures/design/v2/forward/u36_s16';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'133_shared_boundary_audit.png',dpi=160);plt.close(fig)
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>U36-S16</title><h1>共享边界局部审计</h1><p>本图不是新的全球地貌；标准世界参照为 S15。</p><img style="max-width:100%" src="133_shared_boundary_audit.png"><p><a href="../../../../stage1/u36_s15_reference/index.html">S15 冻结世界标准图</a></p>',encoding='utf8')
print(out/'133_shared_boundary_audit.png')
