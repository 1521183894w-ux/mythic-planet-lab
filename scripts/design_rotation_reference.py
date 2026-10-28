"""S9跨机复用S3冻结数组，仅为固定抽查窗口使用S4的审计ID。"""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
from mythic.figures.netcdf import open_netcdf,write_netcdf
ROOT=Path(__file__).resolve().parents[1]
def main():
    source=ROOT/'.cache/u36_s3/world.nc'
    dest=ROOT/'.cache/u36_s9/reference_world.nc'
    with open_netcdf(source) as data:reference=data.load()
    original_id=reference.attrs['world_id']
    reference.attrs['world_id']='design_v2_forward_v2_0Ma_u36_s4_control'
    reference.attrs['audit_id_alias_note']='S9用S4审计ID固定r8随机窗口；物理数组来自S3冻结旁观控制，未重跑全球'
    write_netcdf(reference,dest)
    with open_netcdf(source) as a,open_netcdf(dest) as b:
        same=set(a.data_vars)==set(b.data_vars) and all(
            np.array_equal(a[k].values,b[k].values,equal_nan=True) for k in a.data_vars)
    assert same
    report=dict(source='.cache/u36_s3/world.nc',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),physical_arrays_identical=True,
        original_world_id=original_id,audit_alias=reference.attrs['world_id'],
        note='仅改变用于随机区域抽查的标识并加来源说明，所有物理数组完全相同；未冒称S4的NC字节哈希')
    (ROOT/'data/worlds/design_v2/u36_s9_reference_source.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(report)
if __name__=='__main__':main()
