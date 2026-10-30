"""U41：新生弧壳的有限相邻出口试验；procedural_constrained。

复用既有共享供体限额；不做高程截顶，不新增体积。弧壳40km以下
视作本试验的不可输出基底，这不是上限，也不是经温度验证的流变边界。
"""
import numpy as np
from scipy.spatial import cKDTree
from .borrowing import cap_shared_borrow
from .crust import CONT,OCEAN,ORIGIN_ARC,ORO_ARC,ORO_COLL,ORO_REACT
from .geometry import chord_from_km,km_from_chord


def finite_pair_outlet(T,area,source,target,length_km,dt,kappa=10000.,floor=40.):
    """冻结导度的双单元有限交换，再按节点凸组合和共享供体库存限额。

    双单元交换使用解析后向欧拉体积；多边合并为受限成对通量。
    属程序化有限搬运，不是完整非线性扩散/应力求解器。
    """
    T=np.asarray(T,float);A=np.asarray(area,float)
    s=np.asarray(source,np.int64);r=np.asarray(target,np.int64)
    length=np.asarray(length_km,float)
    if (T.ndim!=1 or A.shape!=T.shape or s.shape!=r.shape or s.ndim!=1
        or length.shape!=s.shape or np.any(A<=0) or not np.isfinite(A).all()
        or not np.isfinite(T).all() or np.any(length<=0) or not np.isfinite(length).all()
        or np.any(s<0) or np.any(r<0) or np.any(s>=len(T)) or np.any(r>=len(T))
        or np.any(s==r) or not np.isfinite([dt,kappa,floor]).all() or dt<0 or kappa<0):
        raise ValueError('invalid outlet input')
    if not len(s) or dt==0 or kappa==0:
        return T.copy(),np.zeros(len(T)),np.zeros(len(s))
    D=kappa*(np.maximum(T[s]-floor,0)/25.)**3
    g=D*2/(3*length**2)*.5*(A[s]+A[r])
    q=dt*g/(1+dt*g*(1/A[s]+1/A[r]))
    mu=np.bincount(s,q/A[s],minlength=len(T))+np.bincount(r,q/A[r],minlength=len(T))
    q/=np.maximum(1.,np.maximum(mu[s],mu[r]))
    proposed=q*np.maximum(T[s]-T[r],0)
    capacity=np.maximum(T-floor,0)*A
    flux=cap_shared_borrow(s[:,None],proposed[:,None],capacity).ravel()
    dv=np.bincount(r,flux,minlength=len(T))-np.bincount(s,flux,minlength=len(T))
    new=T+dv/A
    if new.min()<T.min()-1e-9 or new.max()>T.max()+1e-9:
        raise RuntimeError('outlet created new extremum')
    return new,dv,flux


def outlet_edges(eng,pos,contacts=False):
    """原生弧壳—薄壳边；可选碰撞残片到当前位置邻近陆壳的假设接触边。"""
    c=eng.c;alive=eng.alive;T=c.T;e=eng.pedges
    warm=alive&(c.ctype==CONT)&(c.origin==ORIGIN_ARC)&(T>eng.p.flow_T_km)
    thin=alive&((c.ctype==OCEAN)|(T<eng.p.flow_min_T_km))
    i,j=e.T
    mask=alive[i]&alive[j]&(c.plate[i]==c.plate[j])&(
        (warm[i]&thin[j]&(T[i]>T[j]))|(warm[j]&thin[i]&(T[j]>T[i])))
    pairs=e[mask]
    kinds=np.zeros(len(pairs),np.int8)
    if contacts and warm.any():
        eligible=alive[i]&alive[j]&(c.ctype[i]==CONT)&(c.ctype[j]==CONT)
        eligible &= np.minimum(T[i],T[j])>=eng.p.flow_min_T_km
        degree=np.bincount(e[eligible].ravel(),minlength=len(c))
        sources=np.flatnonzero(warm&(degree==0)&(c.oro==ORO_COLL))
        targets=np.flatnonzero(alive&(c.ctype==CONT)&(T>=eng.p.flow_min_T_km))
        extra=[]
        if len(sources) and len(targets):
            tree=cKDTree(pos[targets])
            lists=tree.query_ball_point(pos[sources],chord_from_km(1.5*eng.m.h_km))
            for s,ks in zip(sources,lists):
                for r in targets[ks]:
                    d=float(km_from_chord(np.linalg.norm(pos[s]-pos[r])))
                    # 排除近乎重合的中心；这仍是有限距离邻近假设，不是精确共边证明。
                    if c.plate[s]!=c.plate[r] and .5*eng.m.h_km<=d<=1.5*eng.m.h_km and T[s]>T[r]:
                        extra.append((s,r))
        if extra:
            pairs=np.vstack([pairs,np.asarray(extra,np.int64)])
            kinds=np.r_[kinds,np.ones(len(extra),np.int8)]
    if not len(pairs):
        return np.empty(0,np.int64),np.empty(0,np.int64),np.empty(0),kinds
    high=T[pairs[:,0]]>=T[pairs[:,1]]
    s=np.where(high,pairs[:,0],pairs[:,1]);r=np.where(high,pairs[:,1],pairs[:,0])
    length=km_from_chord(np.linalg.norm(pos[s]-pos[r],axis=1))
    return s,r,length,kinds


def apply_arc_outlet(eng,dt,pos=None,contacts=False):
    pos=eng.positions() if pos is None else pos
    c=eng.c;s,r,length,kinds=outlet_edges(eng,pos,contacts)
    before=c.T.copy()
    new,dv,flux=finite_pair_outlet(c.T,eng.area,s,r,length,dt,
        kappa=10000.,floor=eng.p.flow_T_km)
    c.T[:]=new
    closure=float(np.dot(c.T-before,eng.area))
    if abs(closure)>.01:raise RuntimeError(f'outlet volume imbalance {closure}')
    receivers=np.unique(r[flux>0])
    converted=receivers[(c.ctype[receivers]==OCEAN)&(c.T[receivers]>=eng.p.arc_to_cont_T_km)]
    if len(converted):
        c.ctype[converted]=CONT;c.origin[converted]=ORIGIN_ARC;c.t_form[converted]=eng.t
        c.oro[converted]=np.where(np.isin(c.oro[converted],(ORO_COLL,ORO_REACT)),c.oro[converted],ORO_ARC)
        c.oro_t[converted]=eng.t
        eng.budget['arc_converted']+=float(np.dot(c.T[converted],eng.area[converted]))
    active=flux>0
    return dict(t_ma=float(eng.t),balance_km3=closure,
        transported_km3=float(flux.sum()),contact_transported_km3=float(flux[kinds==1].sum()),
        source_cells=int(np.unique(s[active]).size),receiver_cells=int(np.unique(r[active]).size),
        converted_cells=int(len(converted)),
        converted_column_km3=float(np.dot(c.T[converted],eng.area[converted])),
        edges=[dict(source_uid=int(c.uid[a]),target_uid=int(c.uid[b]),kind=int(k),
                    length_km=float(d),volume_km3=float(v))
               for a,b,k,d,v in zip(s,r,kinds,length,flux) if v>0])
