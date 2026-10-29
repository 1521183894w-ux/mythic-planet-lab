"""B6 和 PL8 渐变阻流候选：只改候选运行，不覆盖默认引擎。

终图 B6 和 PL8 范围均为随 101 板块运动的假设（user_override），不是地质历史事实。
不修改终态高程或质量指标；以完整300-0Ma正向运行检验地图及材料后果。
"""
from pathlib import Path
import sys, json, time
import numpy as np
from scipy.spatial import cKDTree
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import design_forward as fwd
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import CONT
from mythic.tectonics.geometry import lonlat_to_xyz
from mythic.tectonics.reverse import in_sphere_polygon
from mythic.tectonics.kinematics import RotationFileKinematics

CACHE=ROOT/'.cache/u38_transition';CACHE.mkdir(parents=True,exist_ok=True)
cfg=yaml.safe_load((ROOT/'config/design/final_map_v2.yaml').read_text(encoding='utf8'))
basin=next(x for x in cfg['basins'] if x['id']=='B6')
coords=np.asarray(basin['poly'],float)
def make_region(coords):
    polygon = lonlat_to_xyz(coords[:,0],coords[:,1])
    pts = np.vstack([a + (b-a)*np.linspace(0,1,101,endpoint=False)[:,None] for a,b in zip(coords,np.roll(coords,-1,axis=0))])
    return polygon, cKDTree(lonlat_to_xyz(pts[:,0],pts[:,1]))
plateau=next(x for x in cfg['plateaus'] if x['id']=='PL8')
regions=[make_region(coords),make_region(np.asarray(plateau['poly'],float))]
def resistance(points):
    result=np.zeros(len(points))
    for polygon, tree in regions:
        inside=in_sphere_polygon(points,polygon)
        if inside.any():
            d=tree.query(points[inside])[0]
            distance=2*6371.0*np.arcsin(np.minimum(d/2,1))
            q=np.clip(distance/250.0,0,1)
            result[inside]=np.maximum(result[inside],q*q*(3-2*q))
    return result
kin=RotationFileKinematics(ROOT/'data/worlds/design_v2/history_v2.rot')
original=Engine._surface
rows=[]
def candidate_surface(self,dt,pos=None):
    if self.t>180.0:
        return original(self,dt,pos)
    c,p=self.c,self.p
    pos=self.positions() if pos is None else pos
    ids=np.flatnonzero(self.alive & (c.plate==101) & (c.ctype==CONT))
    future=pos[ids] @ kin.total_from(101,float(self.t),0.0).T
    weight=np.zeros(len(c));weight[ids]=resistance(future)
    selected=ids[weight[ids]>0]
    strong=weight>0
    cont=self.alive & (c.ctype==CONT)
    ex=np.where(cont,np.maximum(c.T-p.T_cont,0.0),0.0)
    if p.erosion_law=='relief':
        loss=np.minimum(self.erosion_rate(pos)*dt,ex)
    elif p.erosion_law=='linear':
        loss=ex*dt/p.tau_oro_myr
    else:raise ValueError('unknown erosion law')
    c.T-=loss
    self.budget['eroded']+=float((loss*self.area).sum())
    i,j=self.pedges[:,0],self.pedges[:,1]
    ok=cont[i]&cont[j]&(np.maximum(c.T[i],c.T[j])>p.flow_T_km)
    if p.flow_min_T_km is not None:
        ok &= np.minimum(c.T[i],c.T[j])>=p.flow_min_T_km
    net_into_basin=0.0
    if ok.any():
        i,j=i[ok],j[ok]
        coef=p.kappa_km2_myr*dt*2.0/(3.0*self.m.h_km**2)
        flux=coef*(c.T[j]-c.T[i])*0.5*(self.area[i]+self.area[j])
        flux*=1.0-0.99*np.maximum(weight[i],weight[j])
        dv=np.bincount(i,flux,minlength=len(c))-np.bincount(j,flux,minlength=len(c))
        net_into_basin=float(dv[strong].sum())
        c.T+=dv/self.area
    # 按权重暂时占用部分容量，仅用于原环形分配候选搜索；
    # 恢复虚拟占用量，不产生或删掉地壳；仍保留原碰撞与上限规则。
    reserve=weight*np.maximum(p.T_max_km-c.T,0.0)
    reserved=np.flatnonzero(reserve)
    volume_before=float(np.dot(c.T,self.area))
    c.T+=reserve
    self._collapse(cont)
    c.T-=reserve
    closure=float(np.dot(c.T,self.area)-volume_before)
    if abs(closure)>0.01:
        raise RuntimeError(f'B6 collapse material balance failed: {closure} km3')
    if int(self.t)%10==0:
        vals=c.T[selected]
        row=dict(t_ma=float(self.t),region_cell_count=len(selected),reserved_count=len(reserved),
                 region_median_T_km=float(np.median(vals)) if len(vals) else None,
                 region_max_T_km=float(np.max(vals)) if len(vals) else None,
                 lateral_volume_in_region_km3=net_into_basin,collapse_balance_km3=closure)
        rows.append(row);print('U38',row,flush=True)

def main():
    start=time.perf_counter()
    Engine._surface=candidate_surface
    sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml',
              '--candidate','u38_b6_pl8_transition','--no-figures']
    try:
        fwd.main()
    finally:
        Engine._surface=original
        report=dict(method_class='procedural_constrained',structural_region_class='user_override',
          method='u38_transition_sensitivity_v1',
          assumptions=['B6 and PL8 polygons follow plate 101; active since 180 Ma',
           'resistance increases smoothly inside polygons over 250 km to 0.99, not calibrated viscosity',
           'temporarily reserved capacity continuously limits overflow reception; direct collision and erosion remain active',
           'original 65km cap and ring redistribution otherwise retained'],
          limits=['real physical basin strength and inherited history unknown',
           'ring redistribution can pass through basin to other cells; no stress or strain solver',
           'rift material source unresolved; candidate not adopted'],
          units={'thickness':'km','volume':'km3','time':'Ma','elevation':'m'},
          diagnostic_rows=rows,elapsed_s=time.perf_counter()-start)
        (CACHE/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
