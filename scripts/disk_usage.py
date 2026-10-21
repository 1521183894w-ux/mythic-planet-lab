"""统计若干目录的真实磁盘占用（硬链接只算一次）。

micromamba 安装环境时，环境目录里的大部分文件是包缓存的硬链接，
直接把两个目录的大小相加会重复计算。这里按 (设备号, 文件编号) 去重。

用法：python scripts/disk_usage.py 目录1 目录2 ...
"""

from __future__ import annotations

import os
import sys


def scan(paths: list[str]) -> None:
    seen: set[tuple[int, int]] = set()
    total_unique = 0
    for root in paths:
        apparent = 0
        unique = 0
        n_files = 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                try:
                    st = os.stat(os.path.join(dirpath, name), follow_symlinks=False)
                except OSError:
                    continue
                n_files += 1
                apparent += st.st_size
                key = (st.st_dev, st.st_ino)
                if key not in seen:
                    seen.add(key)
                    unique += st.st_size
        total_unique += unique
        print(f"{root}: {n_files} 个文件，表观 {apparent / 1e9:.2f} GB，去重后新增 {unique / 1e9:.2f} GB")
    print(f"合计真实占用（硬链接只算一次）：{total_unique / 1e9:.2f} GB")


if __name__ == "__main__":
    scan(sys.argv[1:] or ["."])
