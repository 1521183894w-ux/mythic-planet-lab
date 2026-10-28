"""同一来源连接的换框架审计；不重新选择材料或做物理网格细化。"""
import json,hashlib,time
import numpy as np
import pygplates as pg
from mythic.tectonics.rotation_interpolation import BoundaryRotationMap
from mythic.tectonics.material_band import spherical_triangle_area
from mythic.tectonics.geometry import R_KM
from design_rotation_interpolation import inputs,ROOT,CACHE,DATA

def triangles(mapper):
    return {tuple(sorted(row)) for row in mapper.connectivity}

def main():
    started=time.perf_counter();rows=[]
    for case in ['rigid','ramp','turning','slip','real']:
        rot,features,centers,plates,connectivity=inputs(case)
        anchors=[0,*plates]
        models={a:pg.TopologicalModel(features,rot,anchor_plate_id=a,topological_snapshot_cache_size=2) for a in anchors}
        timeline=[]
        for t in np.arange(35.,-.001,-.25):
            baseline=None
            for a in anchors:
                net=models[a].topological_snapshot(float(t)).get_resolved_topologies()[0]
                dynamic=BoundaryRotationMap(net)
                fixed=BoundaryRotationMap(net,connectivity=connectivity)
                graph=triangles(dynamic)
                if baseline is None:baseline=graph
                xy=fixed.project(fixed.xyz)[fixed.triangles]
                det=np.linalg.det(np.stack([xy[:,0]-xy[:,2],xy[:,1]-xy[:,2]],axis=-1))
                polygon_area=net.get_resolved_boundary().get_area()*R_KM**2
                mesh_area=spherical_triangle_area(fixed.xyz,fixed.triangles).sum()
                # 固定连接不能靠翻转/重叠维持小框架差。
                timeline.append(dict(t_ma=float(t),anchor=a,
                    dynamic_triangle_symmetric_difference_count=len(graph.symmetric_difference(baseline)),
                    fixed_min_projected_signed_double_area=float(det.min()),
                    fixed_negative_triangle_count=int(np.sum(det<=0)),
                    fixed_negative_spherical_area_km2=float(-np.sum(spherical_triangle_area(fixed.xyz,fixed.triangles)[spherical_triangle_area(fixed.xyz,fixed.triangles)<0])),
                    fixed_mesh_polygon_relative_area_difference=float(mesh_area/polygon_area-1),
                    dynamic_mesh_polygon_relative_area_difference=float(spherical_triangle_area(dynamic.xyz,dynamic.triangles).sum()/polygon_area-1),
                    fixed_mesh_boundary_area_test_at_1e8_passed=bool(abs(mesh_area/polygon_area-1)<=1e-8),
                    source_match_max_chord=float(fixed.source_match_max_chord)))
        rows.append(dict(case=case,reference_connectivity=connectivity,timeline=timeline,
            frame_graph_disagreement_samples=sum(q['dynamic_triangle_symmetric_difference_count']>0 for q in timeline)))
        print(case,'graph discrepancies',rows[-1]['frame_graph_disagreement_samples'],flush=True)
    report=dict(checkpoint='U36-S9',method_class='procedural_constrained',rows=rows,
        full_network_geometry_test_passed=all(q['fixed_negative_triangle_count']==0 and q['fixed_mesh_boundary_area_test_at_1e8_passed'] for r in rows for q in r['timeline']),
        sample_times_ma=np.arange(35.,-.001,-.25).tolist(),elapsed_s=time.perf_counter()-started,
        meaning='三角形由排序来源名称标识；对称差计三角形，不是物理裂缝数。固定连接方向检查与网络边界面积分别报告；不以方向正确声称完整覆盖',
        limitation='141采样时点，不证明连续时间或完整网络覆盖；官方按三角形中心分区的三角网也可与连续网络边界有面积差。候选仅验证本轮中心/环点的内部支持，不能作完整域输运')
    (DATA/'u36_s9_mesh_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('mesh seconds',report['elapsed_s'],flush=True)
if __name__=='__main__':main()
