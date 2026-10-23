"""U25：正向演化与终图对照的辅助函数（scripts/design_forward.py）。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fwd = _load("design_forward", "design_forward.py")


def test_label_wrap_joins_regions_across_the_dateline():
    m = np.zeros((20, 40), bool)
    m[5:9, :3] = True           # 180°E 一侧
    m[6:10, -2:] = True         # 180°W 一侧，与上面隔着 180° 经线相接
    m[15:17, 18:22] = True      # 另一块，不相接
    lab, n = fwd.label_wrap(m)
    assert n == 2
    assert lab[6, 0] == lab[7, -1] != 0
    assert lab[16, 20] not in (0, lab[6, 0])


def test_coast_mask_is_the_outer_ring_of_land():
    land = np.zeros((10, 10), bool)
    land[2:8, 2:8] = True
    c = fwd.coast_mask(land)
    assert c[2, 2] and c[7, 5] and not c[4, 4]
    assert not (c & ~land).any()
    assert c.sum() == 6 * 6 - 4 * 4
