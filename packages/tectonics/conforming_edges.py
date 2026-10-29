"""局部审计：将悬挂节点处材料边拆为共享边段，不改变地壳。"""
import numpy as np
import shapely
from .geometry import unit

def shared_boundary(vertices, triangles, project, segments=32, tolerance_km=1e-9):
    """心射平面识别边内既有顶点；每个无向子边仅生成一次采样。"""
    v = unit(np.asarray(vertices))
    tri = np.asarray(triangles, dtype=int)
    if segments < 1 or tolerance_km <= 0:
        raise ValueError("采样数和识别容差必须为正")
    xy = project(v)
    tree = shapely.STRtree(shapely.points(xy))
    incidence, cell_edges, splits = {}, [], []
    for cell, abc in enumerate(tri):
        chain = []
        for a, b in zip(abc, np.roll(abc, -1)):
            vec = xy[b] - xy[a]
            length2 = vec @ vec
            if length2 == 0:
                raise ValueError("零长度材料边")
            line = shapely.LineString([xy[a], xy[b]])
            ids = tree.query(line, predicate="dwithin", distance=tolerance_km)
            fraction = ((xy[ids] - xy[a]) @ vec) / length2
            inside = (fraction > 0) & (fraction < 1) & (ids != a) & (ids != b)
            ids, fraction = ids[inside], fraction[inside]
            order = np.argsort(fraction)
            ids, fraction = ids[order], fraction[order]
            if len(ids) and np.any(np.diff(fraction) <= 0):
                raise ValueError("同一材料边内出现不可区分的顶点")
            seq = [int(a), *map(int, ids), int(b)]
            for node, f in zip(ids, fraction):
                distance = float(np.linalg.norm(xy[node] - (xy[a] + f * vec)))
                splits.append((cell, int(a), int(b), int(node), float(f), distance))
            for x, y in zip(seq, seq[1:]):
                key = tuple(sorted((x, y)))
                incidence.setdefault(key, []).append(cell)
                chain.append((key, x == key[0]))
        cell_edges.append(chain)
    pool, edge_indices = [], {}
    for key in incidence:
        a, b = v[list(key)]
        f = np.arange(segments + 1, dtype=float) / segments
        sample = unit((1-f[:, None])*a + f[:, None]*b)
        edge_indices[key] = np.arange(len(pool), len(pool)+len(sample))
        pool.extend(sample)
    bounds = []
    for chain in cell_edges:
        indices = []
        for key, forward in chain:
            edge = edge_indices[key]
            indices.extend((edge if forward else edge[::-1])[:-1])
        bounds.append(np.asarray(indices, dtype=int))
    pool, inverse = np.unique(np.asarray(pool), axis=0, return_inverse=True)
    return pool, [inverse[x] for x in bounds], incidence, splits
