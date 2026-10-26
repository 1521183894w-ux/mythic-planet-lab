"""长方法说明须完整可读，且不遮挡科学图的坐标或缩小绘图区。"""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mythic.figures.style import stamp, setup_fonts


def test_long_source_extends_canvas_preserving_axes_size_and_separating_footer():
    setup_fonts()
    world = SimpleNamespace(display_name="candidate", world_id="b5", resolution="1024x512",
        seed="one", code_version="abc", source="source and physical limitations " * 60 + "END")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(bottom=.2)
    ax.plot([0, 1], [0, 1])
    ax.set(xlabel="distance km", ylabel="height m")
    note = fig.text(.02, .07, "existing explanatory note", fontsize=9)
    fig.canvas.draw()
    size = ax.get_window_extent().size.copy()
    old_height = fig.get_figheight()
    stamp(fig, world, "procedural_constrained")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer = fig.texts[-1]
    assert footer.get_text().endswith("END")
    assert fig.get_figheight() > old_height
    assert footer.get_window_extent(renderer).y1 < min(ax.get_tightbbox(renderer).y0, note.get_window_extent(renderer).y0)
    np.testing.assert_allclose(ax.get_window_extent().size, size, atol=1.)
    plt.close(fig)
