"""阶段 0.4 工具验证（spike）的公共辅助：路径、计时与内存记录、出图、结果保存。

每个 spike 脚本独立运行：
    powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python spikes\\stage0_tools\\01_spherical_mesh.py
图片写到 figures/stage0/tools/，数值结果写到 figures/stage0/tools/results/<spike>.json。
figures/ 不进 Git，可随时重跑生成；结论汇总在 docs/tool_evaluation.md。
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(os.environ.get("MYTHIC_ROOT", Path(__file__).resolve().parents[2])).resolve()
FIG_DIR = ROOT / "figures" / "stage0" / "tools"
RES_DIR = FIG_DIR / "results"
WORK_DIR = ROOT / ".cache" / "spikes"
LOG_DIR = ROOT / ".cache" / "logs"
for _d in (FIG_DIR, RES_DIR, WORK_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GB = 1e9
MB = 1e6


def import_gplately():
    """导入 gplately。它在导入时会在当前目录建 gplately.log/ptt.log，所以先切到 .cache/logs。"""
    with contextlib.chdir(LOG_DIR):
        import gplately  # noqa: PLC0415
    return gplately


def available_gb() -> float:
    return psutil.virtual_memory().available / GB


class Recorder:
    """记录每个测量项的耗时、进程内存（RSS）与进程峰值工作集。"""

    def __init__(self, spike: str):
        self.spike = spike
        self.items: list[dict] = []
        self.proc = psutil.Process()
        self.t_start = time.perf_counter()
        print(f"== {spike} ==  可用内存 {available_gb():.1f} GB")

    @contextlib.contextmanager
    def measure(self, name: str, **info):
        entry: dict = {"name": name, **info}
        t0 = time.perf_counter()
        try:
            yield entry
            entry.setdefault("status", "ok")
        except Exception as exc:  # noqa: BLE001  记录失败后继续其他测量
            entry["status"] = "fail"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            entry["seconds"] = round(time.perf_counter() - t0, 3)
            mi = self.proc.memory_info()
            entry["rss_mb"] = round(mi.rss / MB)
            entry["peak_mb"] = round(getattr(mi, "peak_wset", mi.rss) / MB)
            self.items.append(entry)
            extra = {k: v for k, v in entry.items() if k not in ("name", "status", "seconds", "rss_mb", "peak_mb")}
            print(f"  [{entry['status']:>4}] {name:<48} {entry['seconds']:>8.2f}s  RSS {entry['rss_mb']:>5} MB  峰值 {entry['peak_mb']:>5} MB  {extra}",
                  flush=True)

    def skip(self, name: str, reason: str, **info):
        self.items.append({"name": name, "status": "skipped", "reason": reason, **info})
        print(f"  [skip] {name:<48} {reason}")

    def save(self, **summary) -> Path:
        out = RES_DIR / f"{self.spike}.json"
        payload = {
            "spike": self.spike,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "machine": platform.node(),
            "python": sys.version.split()[0],
            "total_seconds": round(time.perf_counter() - self.t_start, 1),
            "summary": summary,
            "items": self.items,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        n_fail = sum(i["status"] == "fail" for i in self.items)
        print(f"结果：{out}（失败 {n_fail} 项，总耗时 {payload['total_seconds']} s）")
        return out


def savefig(fig, name: str, dpi: int = 110) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  图：{path}")
    return path
