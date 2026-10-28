"""U36-S14：零通量移动材料单元的体积账原型，不是应力或通用输运求解器。"""
from __future__ import annotations
import numpy as np
from .geometry import unit, R_KM


def boundary_areas(boundary, centers):
    """批量有向球面曲边面积；保持起始方向，任何非正扇形均拒绝。"""
    p=unit(np.asarray(boundary))
    c=unit(np.asarray(centers))[:,None,:]
    q=np.roll(p,-1,axis=1)
    numerator=np.sum(c*np.cross(p-c,q-c),axis=-1)
    denominator=1+np.sum(p*c,axis=-1)+np.sum(q*c,axis=-1)+np.sum(p*q,axis=-1)
    signed=2*np.arctan2(numerator,denominator)*R_KM**2
    if np.any(~np.isfinite(signed)) or np.any(signed<=0):
        raise ValueError("移动边界存在非正球面扇形")
    return signed.sum(axis=1),float(signed.min())


def update_material_mean(thickness, previous_area, next_area):
    """H_new A_new=H_old A_old；这是方案守恒性质，不能独立验证物理。"""
    previous_area=np.asarray(previous_area);next_area=np.asarray(next_area)
    if np.any(previous_area<=0) or np.any(next_area<=0):
        raise ValueError("材料面积必须为正")
    return np.asarray(thickness)*previous_area/next_area


def thickness_error_integrals(mean, initial_thickness, initial_weights, jacobian):
    """独立内部采样：参照为指定离散映射连续性解H0/J，不归一化权重。"""
    w=np.asarray(initial_weights);j=np.asarray(jacobian)
    if np.any(w<0) or np.any(j<=0):
        raise ValueError("求积权重/累计面积因子无效")
    actual_area=w*j
    reference=np.asarray(initial_thickness)[:,None]/j
    approximate=np.asarray(mean)[:,None]
    relative=approximate/reference-1
    return dict(reference_area_km2=actual_area.sum(axis=1),
                candidate_volume_km3=(actual_area*approximate).sum(axis=1),
                l1_volume_error_km3=(actual_area*abs(approximate-reference)).sum(axis=1),
                square_relative_area_km2=(actual_area*relative**2).sum(axis=1),
                minimum_reference_thickness_km=reference.min(axis=1),
                maximum_reference_thickness_km=reference.max(axis=1))
