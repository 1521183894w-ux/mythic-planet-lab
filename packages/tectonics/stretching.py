"""固定单元表上的守恒面积展开代理（U36-S2）。

method_class=procedural_constrained。用既有源柱分摊新增覆盖面积，
H'=H*A/(A+新增分配面积)，目标柱接收相应源体积；不是应力或完整应变张量求解。
面积km²，厚度km，体积km³。稀疏矩阵仅承担成熟线性代数，不构造新PDE求解器。
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix, diags


def conservative_area_expand(thickness, source_area, target_area, neighbors, weights):
    """所有目标同时分配面积，再一次性更新源柱，避免共享源被重复透支。

    weights只定义目标面积由哪些源材料展开覆盖；不指定目标厚度。
    未获得材料支持的目标拒绝执行，不用显式新增补足。
    返回源后态、目标厚度、面积/体积边和每源展开因子。
    """
    h = np.asarray(thickness, float)
    area = np.asarray(source_area, float)
    target = np.asarray(target_area, float)
    nb = np.asarray(neighbors)
    w = np.asarray(weights, float)
    if (h.ndim != 1 or area.shape != h.shape or target.ndim != 1
            or nb.ndim != 2 or nb.shape != w.shape or len(nb) != len(target)
            or nb.dtype.kind not in "iu" or (nb < 0).any() or (nb >= len(h)).any()
            or not np.isfinite(h).all() or not np.isfinite(area).all()
            or not np.isfinite(target).all() or not np.isfinite(w).all()
            or (h < 0).any() or (area <= 0).any() or (target <= 0).any() or (w < 0).any()):
        raise ValueError("展开须提供有效索引、非负厚度/权重及正面积")
    if len(target) == 0:
        return dict(source_thickness=h.copy(), target_thickness=np.zeros(0),
                    source_extra_area=np.zeros(len(h)), source_beta=np.ones(len(h)),
                    row=np.zeros(0, np.int64), col=np.zeros(0, np.int64),
                    edge_area=np.zeros(0), edge_volume=np.zeros(0),
                    row_ptr=np.zeros(1, np.int64))
    rows = np.repeat(np.arange(len(target)), nb.shape[1])
    assignment = coo_matrix((w.ravel(), (rows, nb.ravel())), shape=(len(target), len(h))).tocsr()
    assignment.eliminate_zeros()
    sums = np.asarray(assignment.sum(axis=1)).ravel()
    if (sums <= 0).any():
        raise ValueError("存在没有源材料支持的目标")
    assignment = (diags(target / sums) @ assignment).tocsr()
    assignment.sort_indices()
    extra = np.asarray(assignment.sum(axis=0)).ravel()
    beta = 1 + extra / area
    after = h.copy()
    active = extra > 0
    after[active] = h[active] / beta[active]
    row = np.repeat(np.arange(len(target)), np.diff(assignment.indptr))
    col = assignment.indices.copy()
    edge_volume = assignment.data * after[col]
    target_h = np.asarray(assignment @ after).ravel() / target
    if np.any(target_h <= 0):
        raise ValueError("目标没有正体积陆壳，不能冒充已生成陆壳")
    return dict(source_thickness=after, target_thickness=target_h,
                source_extra_area=extra, source_beta=beta,
                row=row, col=col, edge_area=assignment.data.copy(),
                edge_volume=edge_volume, row_ptr=assignment.indptr.copy())
