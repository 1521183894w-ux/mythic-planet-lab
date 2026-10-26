"""换机后含中文的工作区必须能真实保存并读取世界，不能只让出图脚本编译通过。"""
import numpy as np
import xarray as xr

from mythic.figures.netcdf import open_netcdf, write_netcdf


def test_netcdf_roundtrip_unicode_path_preserves_science_and_encoding(tmp_path):
    path = tmp_path / "中文工作区" / "世界.nc"
    path.parent.mkdir()
    ds = xr.Dataset(
        {"elevation": (("lat", "lon"), np.array([[150., np.nan], [-4653., 2092.]], dtype="float32")),
         "land": (("lat", "lon"), np.array([[1, 0], [0, 1]], dtype="int8"))},
        coords={"lat": [37., 36.], "lon": [-15., -14.]},
        attrs={"world_id": "测试世界", "method_class_elevation": "parameterized_physics"},
    )
    ds.elevation.encoding = {"zlib": True, "complevel": 3}
    write_netcdf(ds, path)
    assert path.read_bytes().startswith(b"\x89HDF")
    with open_netcdf(path) as loaded:
        xr.testing.assert_equal(ds, loaded)
        assert loaded.elevation.dtype == np.dtype("float32")
        assert loaded.elevation.encoding["zlib"] is True
        assert loaded.elevation.encoding["complevel"] == 3
