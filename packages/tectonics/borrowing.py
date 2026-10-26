"""同时补多个洞时，每个供体的可用体积只允许分配一次。"""
import numpy as np


def cap_shared_borrow(neighbors: np.ndarray, proposed_km3: np.ndarray, capacity_km3: np.ndarray) -> np.ndarray:
    """按供体统一缩放超额请求，保留原有各洞之间的份额比例。

    不向其他供体二次讨料，不凭空填足缺额；调用者负责明确记录补洞缺额。
    每个洞获得量不超过原请求，每个供体合计扣款不超过同一份前态余额。
    """
    neighbors = np.asarray(neighbors)
    proposed = np.asarray(proposed_km3, float)
    capacity = np.asarray(capacity_km3, float)
    if (neighbors.ndim != 2 or neighbors.shape != proposed.shape or capacity.ndim != 1
            or neighbors.dtype.kind not in "iu" or np.any(neighbors < 0) or np.any(neighbors >= len(capacity))
            or not np.isfinite(proposed).all() or not np.isfinite(capacity).all()
            or np.any(proposed < 0) or np.any(capacity < 0)):
        raise ValueError("共享借料须为有效供体索引、非负有限请求与余额")
    used = np.bincount(neighbors.ravel(), weights=proposed.ravel(), minlength=len(capacity))
    scale = np.ones(len(capacity))
    over = used > capacity
    scale[over] = capacity[over] / used[over]
    return proposed * scale[neighbors]
