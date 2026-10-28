"""U36-S9投影点插值根因与两侧有限旋转插值候选，不写回地壳。"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, subprocess
from pathlib import Path
import numpy as np
import pygplates as pg
from mythic.tectonics.geometry import R_KM,lonlat_to_xyz,unit
from mythic.tectonics.material_band import build_network,reference_strip,point_array,spherical_triangle_area
from mythic.tectonics.material_audit import small_material_patches
from mythic.tectonics.rotation_interpolation import BoundaryRotationMap,rotation_matrix
from design_step_audit import control_rotation
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/u36_s9'
DATA=ROOT/'data/worlds/design_v2'


def inputs(case):
    if case=='real':
        old=json.loads((DATA/'u36_s4_material_band.json').read_text(encoding='utf8'))
        strip=reference_strip(*np.asarray(old['registered_endpoints_xyz']),200.,30.)
        rot=pg.RotationModel(str(DATA/'history_v2.rot'))
        plates=(301,401)
        centers=unit(strip['xyz'][strip['triangles']].sum(1))
    else:
        strip=reference_strip(lonlat_to_xyz(0,-3.5),lonlat_to_xyz(0,3.5),200,60)
        (ROOT/'.cache/u36_s6').mkdir(exist_ok=True)
        if case=='slip':
            from design_material_band import synthetic_rotation
            (ROOT/'.cache/u36_s4').mkdir(exist_ok=True)
            rot=synthetic_rotation('slip')
        else:
            rot=control_rotation(case)
        plates=(1,2)
        centers=unit(strip['xyz'][strip['triangles']].sum(1))[::8]
    _,features=build_network(strip,rot,35,plates=plates,tag='S9_'+case)
    template_net=pg.TopologicalModel(features,rot).topological_snapshot(35.).get_resolved_topologies()[0]
    connectivity=BoundaryRotationMap(template_net).connectivity
    return rot,features,centers,plates,connectivity


def run_case(case,anchor,dt,bundle,connectivity_mode="fixed"):
    rot,features,centers,plates,connectivity=bundle
    active_connectivity=connectivity if connectivity_mode=="fixed" else None
    model=pg.TopologicalModel(features,rot,anchor_plate_id=anchor,
        topological_snapshot_cache_size=2,
        default_resolve_topology_parameters=pg.ResolveTopologyParameters(
            enable_strain_rate_clamping=False,
            strain_rate_smoothing=pg.StrainRateSmoothing.natural_neighbour))
    points,tri=small_material_patches(centers,1.)
    a0=spherical_triangle_area(points,tri).reshape(-1,4).sum(1)
    initial=points@rotation_matrix(rot.get_rotation(35,anchor))
    legacy=initial.copy()
    candidate=initial.copy()
    timeline=[]
    raw={k:[] for k in ['times_ma','legacy_J','candidate_J','legacy_xyz','candidate_xyz']}
    parity=0.;source_match=0.;min_selected_det=float("inf")
    start=time.perf_counter()
    times=35-np.arange(round(35/dt)+1)*dt
    for t in times:
        restore=rotation_matrix(rot.get_rotation(float(t),anchor))
        old=legacy@restore.T
        new=candidate@restore.T
        jl=spherical_triangle_area(old,tri).reshape(-1,4).sum(1)/a0
        jc=spherical_triangle_area(new,tri).reshape(-1,4).sum(1)/a0
        if np.any(jl<=0) or np.any(jc<=0):
            raise ValueError('材料斑块翻转，不删除该点')
        record=dict(t_ma=float(t),legacy_J_quantiles=np.quantile(jl,[0,.5,1]).tolist(),
            candidate_J_quantiles=np.quantile(jc,[0,.5,1]).tolist(),
            max_candidate_minus_legacy_km=float(np.max(np.linalg.norm(new-old,axis=1))*R_KM))
        for key,val in [('legacy',old),('candidate',new)]:
            if case=='rigid':
                exact=points@rotation_matrix(rot.get_rotation(float(t),plates[0],35)).T
                record[key+'_max_exact_position_error_km']=float(np.max(np.linalg.norm(val-exact,axis=1))*R_KM)
                record[key+'_max_exact_area_error']=float(np.max(abs((jl if key=='legacy' else jc)-1)))
        timeline.append(record)
        for key,value in zip(raw,[t,jl,jc,old[:len(centers)],new[:len(centers)]]):
            raw[key].append(value)
        if t<=1e-10:break
        net=model.topological_snapshot(float(t)).get_resolved_topologies()[0]
        mapper=BoundaryRotationMap(net)
        rr={p:rot.get_rotation(float(t-dt),p,float(t),anchor_plate_id=anchor) for p in plates}
        mapped=point_array([net.reconstruct_point(tuple(p),float(t-dt),False) for p in legacy])
        replica=mapper.projected_map(legacy,rr)
        parity=max(parity,float(np.max(np.linalg.norm(mapped-replica,axis=1))*R_KM))
        fixed_mapper=BoundaryRotationMap(net,connectivity=active_connectivity)
        candidate=fixed_mapper.rotation_map(candidate,rr)
        min_selected_det=min(min_selected_det,fixed_mapper.minimum_selected_triangle_det)
        legacy=mapped
        source_match=max(source_match,mapper.source_match_max_chord*R_KM)
    # 末态没有下一步，单独核对0Ma所有中心/环点的网络及正向支撑。
    end_net=model.topological_snapshot(0.).get_resolved_topologies()[0]
    end_mapper=BoundaryRotationMap(end_net,connectivity=active_connectivity)
    end_mapper.weights(candidate)
    min_selected_det=min(min_selected_det,end_mapper.minimum_selected_triangle_det)
    end_mapper.weights(legacy)
    arrays={k:np.asarray(v) for k,v in raw.items()}
    name=f'{case}_{anchor}_{dt}'
    array_dir=CACHE if connectivity_mode=='fixed' else CACHE/'dynamic'
    array_dir.mkdir(exist_ok=True)
    np.savez_compressed(array_dir/(name+'.npz'),**arrays)
    return dict(case=case,anchor=anchor,dt_myr=dt,center_count=len(centers),
        support_check_point_count=len(points),final_time_support_checked=True,timeline=timeline,
        legacy_replica_max_error_km=parity,source_vertex_match_max_error_km=source_match,
        sampled_outside_count=0,sampled_inverted_count=0,
        minimum_selected_triangle_det=min_selected_det,
        elapsed_s=time.perf_counter()-start),arrays

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',default=['rigid','ramp','turning','slip','real'])
    p.add_argument('--steps',nargs='+',type=float,default=[.25,.125])
    p.add_argument('--anchors',nargs='+',type=int)
    p.add_argument('--connectivity-mode',choices=['fixed','dynamic'],default='fixed')
    args=p.parse_args()
    if any(dt<=0 or not np.isclose(round(1/dt)*dt,1) for dt in args.steps):
        raise ValueError('步长须为正且整除1Myr关键帧')
    CACHE.mkdir(parents=True,exist_ok=True)
    rows=[];started=time.perf_counter()
    for case in args.cases:
        bundle=inputs(case)
        for dt in args.steps:
            anchor_list=args.anchors or ([0,301,401] if case=='real' else [0,1,2])
            baselines=None
            for anchor in anchor_list:
                row,arr=run_case(case,anchor,dt,bundle,args.connectivity_mode)
                if baselines is None:
                    if anchor!=0:raise ValueError('每组须先运行0号参照系')
                    baselines=arr
                for method in ['legacy','candidate']:
                    delta=np.linalg.norm(arr[method+'_xyz']-baselines[method+'_xyz'],axis=-1)*R_KM
                    jdelta=arr[method+'_J']/baselines[method+'_J']-1
                    row[method+'_max_frame_position_difference_km']=float(delta.max())
                    row[method+'_max_frame_J_relative_difference']=float(np.max(abs(jdelta)))
                    row[method+'_final_frame_position_difference_km']=float(delta[-1].max())
                    row[method+'_final_frame_J_relative_difference']=float(np.max(abs(jdelta[-1])))
                rows.append(row)
                print(case,dt,anchor,'replica_m',row['legacy_replica_max_error_km']*1000,
                      'frame_m',row['legacy_max_frame_position_difference_km']*1000,
                      row['candidate_max_frame_position_difference_km']*1000,
                      'seconds',row['elapsed_s'],flush=True)
    import ctypes
    class Memory(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(q,ctypes.c_size_t) for q in
            ['peak','work','peak_page','page','peak_nonpage','nonpage','pagefile','peak_pagefile']]
    mem=Memory();mem.cb=ctypes.sizeof(mem)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(mem),mem.cb):
        raise RuntimeError('无法读取进程峰值工作集')
    report=dict(checkpoint='U36-S9',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        framework_version=pg.__version__,reference_connectivity_mode=args.connectivity_mode,method_classes={'geometry_width':'user_override',
        'legacy_and_candidate_point_maps':'interpolated','comparison':'procedural_constrained'},
        units={'position':'km','time':'Myr / Ma','J':'无量纲'},rows=rows,
        elapsed_s=time.perf_counter()-started,peak_working_set_mib=mem.peak/2**20,
        method=('35Ma官方网络连接按来源名称固定' if args.connectivity_mode=='fixed' else '每时点官方网格动态连接')+'；LAEA重心权重；pyGPlates两侧阶段旋转经SciPy SLERP后作用查询点',
        legacy='官方reconstruct_point(...,False)重心分支；并用同源顶点旋转/投影独立复现',
        support='每步检查所有中心/环点；不在网络中或翻转即终止，不动态丢弃材料',
        limitations=['仅两个源板块、无刚体内岛/裂谷细分节点的局部插值候选',
        '不是现有框架补丁，不替换框架beta或速度场，不接入全球/B5',
        '带宽200km仍是假设；无新的物理古接触/来源/外圈演化结果',
        '候选的参照系改善与原框架beta/速度两项严格失败分开记录',
        '旧S8来源UID误差平台未在本轮修复；不输出地貌高程或水位',
        '整张参考网格边缘存在投影方向转负和面积覆盖差，完整网络几何检查失败；本轮只验证查询点选中的正向、无内部重叠三角形'])
    report_path=DATA/('u36_s9_rotation_maps.json' if args.connectivity_mode=='fixed' else 'u36_s9_dynamic_rotation_maps.json')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('TOTAL',report['elapsed_s'],'peakMiB',report['peak_working_set_mib'],flush=True)

if __name__=='__main__':main()
