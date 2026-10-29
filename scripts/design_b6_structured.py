"""真实世界B6结构阻流候选：只改候选运行，不覆盖默认引擎。

终图B6多边形作为板块101上的继承强盆地（user_override），并非重建出的古地质事实。
不修改终态高程或质量指标；以完整300-0Ma正向运行检验地图及材料后果。
"""
from pathlib import Path
import sys, json, time
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import design_forward as fwd
from mythic.tectonics.engine import Engine
from mythic.tectonics.crust import CONT
from mythic.tectonics.geometry import lonlat_to_xyz
from mythic.tectonics.reverse import in_sphere_polygon
from mythic.tectonics.kinematics import RotationFileKinematics

CACHE=ROOT/'.cache/u37_b6';CACHE.mkdir(parents=True,exist_ok=True)
cfg=yaml.safe_load((ROOT/'config/design/final_map_v2.yaml').read_text(encoding='utf8'))
basin=next(x for x in cfg['basins'] if x['id']=='B6')
coords=np.asarray(basin['poly'],float)
polygon=lonlat_to_xyz(coords[:,0],coords[:,1])
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
    selected=ids[in_sphere_polygon(future,polygon)]
    strong=np.zeros(len(c),bool);strong[selected]=True
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
        flux*=np.where(strong[i]|strong[j],0.01,1.0)
        dv=np.bincount(i,flux,minlength=len(c))-np.bincount(j,flux,minlength=len(c))
        net_into_basin=float(dv[strong].sum())
        c.T+=dv/self.area
    # 已在上限以下的盆地单元临时标成无容量，仅用于原环形分配的候选搜索；
    # 恢复原厚度，不产生或删掉地壳。超厚盆地仍由原规则处理，不压低碰撞增厚。
    reserved=selected[c.T[selected]<p.T_max_km]
    previous=c.T[reserved].copy()
    volume_before=float(np.dot(c.T,self.area))
    c.T[reserved]=p.T_max_km
    self._collapse(cont)
    c.T[reserved]=previous
    closure=float(np.dot(c.T,self.area)-volume_before)
    if abs(closure)>0.01:
        raise RuntimeError(f'B6 collapse material balance failed: {closure} km3')
    if int(self.t)%10==0:
        vals=c.T[selected]
        row=dict(t_ma=float(self.t),basin_cell_count=len(selected),reserved_count=len(reserved),
                 basin_median_T_km=float(np.median(vals)) if len(vals) else None,
                 basin_max_T_km=float(np.max(vals)) if len(vals) else None,
                 lateral_volume_in_km3=net_into_basin,collapse_balance_km3=closure)
        rows.append(row);print('B6',row,flush=True)

def main():
    start=time.perf_counter()
    Engine._surface=candidate_surface
    sys.argv=['design_forward.py','--history-config','config/design/history_v2_u35_c10_progressive.yaml',
              '--candidate','u37_b6_structured','--no-figures']
    try:
        fwd.main()
    finally:
        Engine._surface=original
        report=dict(method_class='procedural_constrained',structural_region_class='user_override',
          method='u37_inherited_basin_resistance_candidate_v1',
          assumptions=['B6 final-design polygon follows plate 101; active since 180 Ma',
           'flow involving basin multiplied by 0.01, a sensitivity candidate not calibrated viscosity',
           'basin cells below cap do not receive cap overflow; direct collision and erosion remain active',
           'original 65km cap and ring redistribution otherwise retained'],
          limits=['real physical basin strength and inherited history unknown',
           'ring redistribution can pass through basin to other cells; no stress or strain solver',
           'rift material source unresolved; candidate not adopted'],
          units={'thickness':'km','volume':'km3','time':'Ma','elevation':'m'},
          diagnostic_rows=rows,elapsed_s=time.perf_counter()-start)
        (CACHE/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
