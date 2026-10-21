"""格网拓扑：8 邻域、经度环绕、跨极点相邻、连通区域标记。

约定：
- 经度方向周期（第 0 列与最后一列相邻）；
- 纬度方向不周期，但第 0 行（最北）内第 j 列与第 j + nx/2 列隔着北极点相邻，最后一行同理（南极）。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

OFFSETS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def shift(a: np.ndarray, dy: int, dx: int, fill=False) -> np.ndarray:
    """out[i, j] = a[i + dy, (j + dx) mod nx]；纬度方向越界处填 fill。"""
    src = np.roll(a, -dx, axis=1)
    out = np.full_like(a, fill)
    if dy > 0:
        out[:-dy] = src[dy:]
    elif dy < 0:
        out[-dy:] = src[:dy]
    else:
        out[:] = src
    return out


def any_neighbor(mask: np.ndarray) -> np.ndarray:
    """8 邻域内（含经度环绕、跨极点）至少有一个 True 的格点。不含自身。"""
    out = np.zeros_like(mask, dtype=bool)
    for dy, dx in OFFSETS_8:
        out |= shift(mask, dy, dx, fill=False)
    nx = mask.shape[1]
    half = nx // 2
    out[0] |= np.roll(mask[0], half)
    out[-1] |= np.roll(mask[-1], half)
    return out


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    """mask 中与 mask 外相邻的格点（例如陆地中的海岸格点）。"""
    return mask & any_neighbor(~mask)


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8 连通区域标记（经度环绕 + 跨极点）。返回 (labels, n)，背景为 0，标签 1..n 连续。"""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return lab.astype(np.int32), 0
    ny, nx = mask.shape
    a_list, b_list = [], []
    left, right = lab[:, 0], lab[:, -1]
    # 左右接缝：第 0 列第 i 行与最后一列第 i-1、i、i+1 行相邻
    for a, b in ((left, right), (left[1:], right[:-1]), (left[:-1], right[1:])):
        keep = (a > 0) & (b > 0)
        a_list.append(a[keep])
        b_list.append(b[keep])
    # 极点：首行/末行中相隔 nx/2 的格点相邻
    for row in (lab[0], lab[-1]):
        a, b = row, np.roll(row, nx // 2)
        keep = (a > 0) & (b > 0)
        a_list.append(a[keep])
        b_list.append(b[keep])
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    if a.size == 0:
        return lab.astype(np.int32), n
    graph = coo_matrix((np.ones(a.size), (a, b)), shape=(n + 1, n + 1))
    _, comp = connected_components(graph, directed=False)
    # comp[0] 是背景（没有任何边），重新编号为 0，其余按首次出现顺序编号为 1..m
    uniq, inverse = np.unique(comp[1:], return_inverse=True)
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[1:] = inverse + 1
    return remap[lab], int(uniq.size)


def component_sums(labels: np.ndarray, n: int, weights: np.ndarray) -> np.ndarray:
    """每个标签的权重和（例如面积），返回长度 n+1 的数组，下标 0 为背景。"""
    return np.bincount(labels.ravel(), weights=np.broadcast_to(weights, labels.shape).ravel(), minlength=n + 1)


def roll_to_avoid_seam(mask: np.ndarray) -> int:
    """找一个列偏移，使 mask 在 np.roll(mask, shift, axis=1) 后不跨越左右接缝；
    若每一列都有 True（环绕全球），返回 0。"""
    cols = mask.any(axis=0)
    if cols.all():
        return 0
    empty = np.flatnonzero(~cols)
    # 让第一个空列移到第 0 列
    return int(-empty[0])
