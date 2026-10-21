"""按 data/external_manifests/*.yaml 下载地球原始数据并校验 SHA256（新机器上重建 data/earth_reference/raw/）。

- 已存在且 SHA256 一致的文件直接跳过；不一致或缺失才下载。
- 大文件按字节范围分段并行下载，每段断线后自动续传（NOAA 服务器单连接常在中途断开，见 etopo2022.yaml 的下载说明）。
- 网络走系统环境变量里的代理（HTTPS_PROXY），本机直连 NOAA 不通。
- 下载完成后校验 SHA256，不符则删除并报错。

用法：
  powershell -ExecutionPolicy Bypass -File scripts\\run.ps1 python -u scripts\\download_earth_data.py
  选项：--force 强制重新下载；--only <文件名> 只处理指定文件；--parallel-threshold-mb N（默认 50）
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import shutil
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "external_manifests"
TMP = ROOT / ".cache" / "downloads"
N_PARTS = 6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url: str) -> int:
    """服务器不一定在 HEAD 中给 Content-Length（NOAA 就不给），用 1 字节范围请求读 Content-Range。"""
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=(30, 60), stream=True)
    r.raise_for_status()
    cr = r.headers.get("Content-Range", "")
    r.close()
    if "/" in cr:
        return int(cr.rsplit("/", 1)[1])
    return int(r.headers.get("Content-Length", 0))


def fetch_range(url: str, start: int, end: int, part: Path, retries: int = 10) -> None:
    """下载 [start, end] 字节到 part 文件，断线后从已下载处续传。"""
    want = end - start + 1
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return
        try:
            with requests.get(url, headers={"Range": f"bytes={start + have}-{end}"}, timeout=(30, 120), stream=True) as r:
                r.raise_for_status()
                with open(part, "ab") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except (requests.RequestException, OSError) as exc:
            log(f"    {part.name} 第 {attempt} 次中断（{type(exc).__name__}），已下 {part.stat().st_size if part.exists() else 0} / {want} 字节，续传")
            time.sleep(min(30, 3 * attempt))
    if not part.exists() or part.stat().st_size < want:
        raise RuntimeError(f"{part.name} 多次重试后仍未下载完整")


def download(url: str, dest: Path, size: int, parallel_threshold: int) -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    n = N_PARTS if size >= parallel_threshold else 1
    step = -(-size // n)
    parts = []
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            s, e = i * step, min(size, (i + 1) * step) - 1
            if s > e:
                continue
            part = TMP / f"{dest.name}.part{i}"
            parts.append(part)
            jobs.append(pool.submit(fetch_range, url, s, e, part))
        for j in jobs:
            j.result()
    tmp = TMP / f"{dest.name}.tmp"
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 22)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), dest)
    for p in parts:
        p.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--parallel-threshold-mb", type=float, default=50.0)
    args = ap.parse_args()
    threshold = int(args.parallel_threshold_mb * 1e6)

    entries = []
    for m in sorted(MANIFESTS.glob("*.yaml")):
        for f in yaml.safe_load(m.read_text(encoding="utf-8")).get("files", []):
            if args.only and f["name"] not in args.only:
                continue
            entries.append(f)
    total = sum(f.get("bytes", 0) for f in entries)
    log(f"清单中 {len(entries)} 个文件，共约 {total / 1e6:.0f} MB")

    failed = 0
    for f in entries:
        dest = ROOT / f["local_path"]
        if dest.exists() and not args.force:
            if sha256(dest) == f["sha256"]:
                log(f"  已存在且校验通过：{f['name']}")
                continue
            log(f"  已存在但 SHA256 不符，重新下载：{f['name']}")
        t0 = time.time()
        size = f.get("bytes") or remote_size(f["url"])
        log(f"  下载 {f['name']}（{size / 1e6:.1f} MB）...")
        try:
            download(f["url"], dest, size, threshold)
        except Exception as exc:  # noqa: BLE001
            log(f"  失败：{f['name']}：{exc}")
            failed += 1
            continue
        got = sha256(dest)
        if got != f["sha256"]:
            dest.unlink(missing_ok=True)
            log(f"  SHA256 不符，已删除：{f['name']}（得到 {got[:12]}…，应为 {f['sha256'][:12]}…）")
            failed += 1
            continue
        dt = time.time() - t0
        log(f"  完成并校验通过：{f['name']}（{dt:.0f} s，{size / 1e6 / max(dt, 1e-9):.2f} MB/s）")
    log("全部完成" if failed == 0 else f"有 {failed} 个文件失败")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
