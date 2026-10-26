"""NetCDF 读写：兼容 Windows 上 C 库不能打开的 Unicode 路径。

只在 Windows 且绝对路径含非 ASCII 字符时走 netCDF4 内存接口；文件路径交给
Python 读写。保持 NETCDF4 格式、变量编码与数值。当前全球输出为十数 MB 量级；
此分支会额外占用约一个文件大小的内存，不适用于未来的超大分块数据。
"""
from __future__ import annotations

import os
from pathlib import Path

import xarray as xr


def _memory_io(path: Path) -> bool:
    return os.name == "nt" and not str(path.absolute()).isascii()


def open_netcdf(path: str | Path) -> xr.Dataset:
    """返回可关闭的 Dataset；调用方照常使用 with，避免持有内存文件。"""
    path = Path(path)
    if _memory_io(path):
        return xr.open_dataset(path.read_bytes(), engine="netcdf4")
    return xr.open_dataset(path)


def write_netcdf(ds: xr.Dataset, path: str | Path) -> None:
    path = Path(path)
    if _memory_io(path):
        path.write_bytes(ds.to_netcdf(engine="netcdf4"))
    else:
        ds.to_netcdf(path)
