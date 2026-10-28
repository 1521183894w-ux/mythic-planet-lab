"""U36-S15：局部球面交叠面积/一阶矩重映射小试，不改变正式引擎。"""
from __future__ import annotations
import numpy as np
import shapely
from .geometry import R_KM, unit
from .source_clipping import polygons


def _positive_cubature(poly, projection, order):
    """GEOS约束剖分+正权Duffy积分；全部面积用球面Jacobian，不归一化。"""
    from scipy.special import roots_legendre
    nodes,weights=roots_legendre(order)
    u,v=np.meshgrid((nodes+1)/2,(nodes+1)/2,indexing='ij')
    base=np.outer(weights/2,weights/2)*(1-u)
    area=0.;moment=np.zeros(3)
    triangles=list(shapely.constrained_delaunay_triangles(poly).geoms)
    if not triangles and poly.area>0:
        raise ValueError('精度薄片无法约束三角剖分，拒绝补值')
    for tri in triangles:
        a,b,c=np.array(tri.exterior.coords)[:3];b=b-a;c=c-a
        xy=a+u[...,None]*b+((1-u)*v)[...,None]*c
        weight=base*(2*tri.area)/(1+(xy*xy).sum(axis=-1)/R_KM**2)**1.5
        xyz=projection.inverse(xy.reshape(-1,2))
        area+=float(weight.sum());moment+=(weight.ravel()[:,None]*xyz).sum(axis=0)
    return area,moment


def spherical_moments(geometry, projection, return_diagnostics=False):
    """球面面积和∫单位位置dA；稳定闭式矩，病态薄片用正权球面积分。

    M=R²/2 Σθn；首项改为中心差分。若M/A超出外环球冠或单位球，
    以4/8阶正权cubature核准并采用8阶，保存敏感性；不裁负体积或逐源归一化。
    """
    area=0.;moment=np.zeros(3)
    diag=dict(cubature_polygons=0,fan_minus_cubature_area_km2=0.,
              cubature_area_4_8_difference_km2=0.,cubature_moment_4_8_difference_km2=0.)
    for poly in polygons(geometry):
        poly_area=0.;poly_moment=np.zeros(3)
        exterior=projection.inverse(np.asarray(poly.exterior.coords)[:-1]);center=unit(exterior.sum(axis=0))
        for number,ring in enumerate([poly.exterior,*poly.interiors]):
            p=projection.inverse(np.asarray(ring.coords)[:-1])
            if len(p)<3:continue
            q=np.roll(p,-1,axis=0);c=unit(p.sum(axis=0));cross=np.cross(p,q)
            length=np.linalg.norm(cross,axis=1);angle=np.arctan2(length,np.sum(p*q,axis=1))
            stable_cross=np.cross(p-c,q-c)
            numerator=stable_cross@c;denominator=1+p@c+q@c+np.sum(p*q,axis=1)
            signed_area=float(np.sum(2*np.arctan2(numerator,denominator))*R_KM**2)
            factor=np.divide(angle,length,out=np.ones_like(length),where=length>0)
            square=angle*angle
            excess=np.where(angle<.01,square/6+7*square**2/360+31*square**3/15120,factor-1)
            signed_moment=.5*R_KM**2*(stable_cross.sum(axis=0)+(cross*excess[:,None]).sum(axis=0))
            direction=1. if signed_area>=0 else -1.;sign=1. if number==0 else -1.
            poly_area+=sign*abs(signed_area);poly_moment+=sign*direction*signed_moment
        # 正面积积分的质心必在局部球冠凸包及单位球内。128eps只容许舍入。
        slack=128*np.finfo(float).eps;radius=np.linalg.norm(exterior-center,axis=1).max()
        invalid=poly_area<=0 or not np.isfinite(poly_moment).all()
        if not invalid:
            centroid=poly_moment/poly_area
            invalid=np.linalg.norm(centroid)>1+slack or np.linalg.norm(centroid-center)>radius+slack
        if invalid and poly.area>0:
            low,low_m=_positive_cubature(poly,projection,4);high,high_m=_positive_cubature(poly,projection,8)
            diag['cubature_polygons']+=1
            diag['fan_minus_cubature_area_km2']+=poly_area-high
            diag['cubature_area_4_8_difference_km2']+=abs(high-low)
            diag['cubature_moment_4_8_difference_km2']+=float(np.linalg.norm(high_m-low_m))
            poly_area,poly_moment=high,high_m
        area+=poly_area;moment+=poly_moment
    return (area,moment,diag) if return_diagnostics else (area,moment)


def same_source_neighbors(triangles, source_uid, known):
    """只沿真实共享初始边、同一来源且已知厚度的单元取模板。"""
    neighbors = [set() for _ in triangles]; edge_owner = {}
    for i, tri in enumerate(triangles):
        for edge in [(tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])]:
            key = tuple(sorted(map(int, edge)))
            if key in edge_owner:
                j = edge_owner[key]
                if known[i] and known[j] and source_uid[i] == source_uid[j]:
                    neighbors[i].add(j); neighbors[j].add(i)
            else:
                edge_owner[key] = i
    # 两圈模板减少细长三角形的梯度退化，不跨来源/未知边。
    return [sorted(s.union(*(neighbors[j] for j in s))-{i}) if s else []
            for i,s in enumerate(neighbors)]


def limited_reconstruction(means, centroids, boundary_xyz, neighbors):
    """从相邻均值拟合切平面梯度，以保存边界点作有限采样限幅。

    H_i(x)=mean_i+G_i·(x-centroid_i)，球面一阶矩保证本单元零均值修正。
    限幅仅对给定边界采样点成立，不称严格连续域最大值证明。
    """
    means=np.asarray(means); gradients=np.zeros((len(means),3)); factors=np.zeros(len(means))
    ranks=np.zeros(len(means),np.int8)
    for i,js in enumerate(neighbors):
        if not np.isfinite(means[i]) or len(js)<2:
            continue
        center=unit(centroids[i]); east=unit(np.cross([0.,0.,1.],center)); north=np.cross(center,east)
        d=(centroids[js]-centroids[i])*R_KM
        distance=np.linalg.norm(d,axis=1)
        design=np.column_stack([d@east,d@north])/distance[:,None]
        coef,_,rank,_=np.linalg.lstsq(design,(means[js]-means[i])/distance,rcond=None)
        ranks[i]=rank
        if rank<2:
            continue
        g=R_KM*(coef[0]*east+coef[1]*north)
        delta=(boundary_xyz[i]-centroids[i])@g
        low=min(means[i],float(np.min(means[js])));high=max(means[i],float(np.max(means[js])))
        theta=1.
        if delta.max()>0:theta=min(theta,(high-means[i])/delta.max())
        if delta.min()<0:theta=min(theta,(low-means[i])/delta.min())
        factors[i]=max(0.,theta); gradients[i]=g*factors[i]
    return gradients,factors,ranks


def overlap_integrals(geometries, source_polygons, projection, means, centroids, gradients):
    """按(source,target)输出面积及常数/有限线性体积；不逐源归一化。"""
    tree=shapely.STRtree(geometries)
    rows=[];columns=[];areas=[];moments=[];constant=[];linear=[];diagnostics=[]
    for i,poly in enumerate(source_polygons):
        for j in tree.query(poly,predicate='intersects'):
            hit=poly.intersection(geometries[j])
            area,moment,diag=spherical_moments(hit,projection,return_diagnostics=True)
            if area==0:
                continue
            if area<0:
                raise ValueError('交叠面积为负，保留现场而不修正')
            rows.append(i);columns.append(j);areas.append(area);moments.append(moment);diagnostics.append(diag)
            constant.append(area*means[i]);linear.append(area*means[i]+(moment-area*centroids[i])@gradients[i])
    return dict(source_cell=np.array(rows,np.int32),target_cell=np.array(columns,np.int32),
                area_km2=np.array(areas),moment_xyz_km2=np.array(moments),
                constant_volume_km3=np.array(constant),linear_volume_km3=np.array(linear),
                **{'moment_'+key:np.array([d[key] for d in diagnostics]) for key in diagnostics[0]})
