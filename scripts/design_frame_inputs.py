"""U36-S7输入等价核对：独立记录坐标往返、边界点集、源ID及旋转群关系。"""
from pathlib import Path
import numpy as np,pygplates as pg
from scipy.spatial import cKDTree
from mythic.tectonics.geometry import R_KM,lonlat_to_xyz,unit
from mythic.tectonics.kinematics import fr_to_mat
from mythic.tectonics.material_band import reference_strip,build_network,point_array
from design_step_audit import control_rotation
from design_material_audit import write_json
ROOT=Path(__file__).resolve().parents[1]
def main():
    strip=reference_strip(lonlat_to_xyz(0,-3.5),lonlat_to_xyz(0,3.5),200,60);rows=[]
    for case in ['rigid','ramp','turning','real']:
        if case=='real':
            a=np.load(ROOT/'.cache/u36_s4/material_band.npz');points=a['xyz'];rot=pg.RotationModel(str(ROOT/'data/worlds/design_v2/history_v2.rot'));features=pg.FeatureCollection(str(ROOT/'.cache/u36_s4/material_network.gpml'));plates=[301,401]
        else:
            points=strip['xyz'];rot=control_rotation(case);_,features=build_network(strip,rot,35,plates=(1,2),tag=case);plates=[1,2]
        models={i:pg.TopologicalModel(features,rot,anchor_plate_id=i) for i in [0]+plates}
        for anchor in [0]+plates:
            ra=fr_to_mat(rot.get_rotation(35.,anchor));recovered=(points@ra)@ra.T
            matrix_max=0.;relative_max=0.
            for t in range(51):
                ra=fr_to_mat(rot.get_rotation(float(t),anchor));rm=models[anchor].get_rotation_model()
                aa=[];gg=[]
                for plate in plates:
                    m=fr_to_mat(rm.get_rotation(float(t),plate));g=fr_to_mat(rot.get_rotation(float(t),plate));aa.append(m);gg.append(g)
                    matrix_max=max(matrix_max,float(np.max(abs(ra@m-g))))
                relative_max=max(relative_max,float(np.max(abs(aa[0].T@aa[1]-gg[0].T@gg[1]))))
            boundary=[]
            for t in [35.,34.,20.,0.]:
                base=models[0].topological_snapshot(t).get_resolved_topologies()[0];other=models[anchor].topological_snapshot(t).get_resolved_topologies()[0]
                x=point_array(base.get_resolved_geometry_points());y=point_array(other.get_resolved_geometry_points())@fr_to_mat(rot.get_rotation(t,anchor)).T
                dist=max(cKDTree(x).query(y)[0].max(),cKDTree(y).query(x)[0].max())*R_KM
                ids=lambda net:sorted(str(q.get_feature_id()) for q in net.get_resolved_geometry_point_features())
                boundary.append(dict(t_ma=t,base_point_count=len(x),anchored_point_count=len(y),maximum_bidirectional_chord_error_km=float(dist),source_feature_ids_identical=ids(base)==ids(other)))
            rows.append(dict(case=case,anchor=anchor,initial_point_roundtrip_max_chord_km=float(np.linalg.norm(recovered-points,axis=1).max()*R_KM),
                total_rotation_matrix_max_component_error=matrix_max,relative_rotation_matrix_max_component_error=relative_max,rotation_check_times_ma=list(range(51)),boundary_checks=boundary))
    out=dict(checkpoint='U36-S7',method_class='procedural_constrained',units={'length':'km','matrix':'无量纲'},rows=rows,
        limitations=['初始点为同一输入的坐标往返核对；不是独立地质历史','边界几何只检查列出的4时刻，旋转51时刻','不把输入等价核对当输出客观性已通过'])
    write_json(ROOT/'data/worlds/design_v2/u36_s7_input_equivalence.json',out)
    print('max initial km',max(q['initial_point_roundtrip_max_chord_km'] for q in rows),'max boundary km',max(z['maximum_bidirectional_chord_error_km'] for q in rows for z in q['boundary_checks']),
        'rotation matrix',max(q['total_rotation_matrix_max_component_error'] for q in rows),'all IDs',all(z['source_feature_ids_identical'] for q in rows for z in q['boundary_checks']),flush=True)
if __name__=='__main__':main()
