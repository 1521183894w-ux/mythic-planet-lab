"""给定独立参考材料带，用pyGPlates运动学网络追踪，不改变构造引擎。

本模块仅构造框架输入、固定审计三角形和计算球面几何。
速度/应变插值及材料点积分均由pyGPlates实现；没有自写流变求解器。
"""
from __future__ import annotations
import numpy as np
import pygplates as pg
from .geometry import R_KM, unit


def slerp_points(a, b, fraction):
    a, b = unit(a), unit(b)
    angle = np.arccos(np.clip(a @ b, -1, 1))
    if angle < 1e-10 or angle > np.pi - 1e-8:
        raise ValueError("参考线端点须不同且不对跖")
    f = np.asarray(fraction, float)[..., None]
    return unit(np.sin((1-f)*angle)/np.sin(angle)*a + np.sin(f*angle)/np.sin(angle)*b)


def reference_strip(a, b, width_km, spacing_km):
    """沿一段独立大圆定义材料带；返回固定编号顶点、三角形和两条刚体边界。

    across从C侧负值到D侧正值；端点按配置顺序取南→北，右侧为东。
    这是局部试验拓扑，不自动推断全球接触或断层分支。
    """
    if not np.isfinite([width_km, spacing_km]).all() or min(width_km, spacing_km) <= 0:
        raise ValueError("带宽与采样间距须为有限正值")
    a, b = unit(a), unit(b)
    length = np.arccos(np.clip(a @ b, -1, 1))*R_KM
    nx, ny = max(2, int(np.ceil(width_km/spacing_km))), max(2, int(np.ceil(length/spacing_km)))
    # 跨带数取偶数，保证参考中线身份不随分辨率移位。
    nx += nx % 2
    center = slerp_points(a, b, np.linspace(0, 1, ny+1))
    east = -unit(np.cross(a, b))
    across = np.linspace(-width_km/2, width_km/2, nx+1)
    xyz = unit(np.cos(across/R_KM)[None, :, None]*center[:, None, :]
               + np.sin(across/R_KM)[None, :, None]*east)
    grid = np.arange((ny+1)*(nx+1)).reshape(ny+1, nx+1)
    cells = np.column_stack([grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel(),
                             grid[1:, 1:].ravel(), grid[1:, :-1].ravel()])
    triangles = np.concatenate([cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]])
    boundary = np.r_[grid[:, 0], grid[-1, 1:], grid[-2::-1, -1], grid[0, -2:0:-1]]
    # 顶/底连接点不用虚构中间板块；网络边缘只引用两侧刚体点。
    rail_boundary = np.r_[grid[:, 0], grid[::-1, -1]]
    return dict(xyz=xyz.reshape(-1, 3), triangles=triangles, grid=grid,
                boundary=boundary, rails=rail_boundary, across_km=across,
                along_km=np.linspace(0, length, ny+1), east=east,
                actual_dx_km=width_km/nx, actual_dy_km=length/ny)


def spherical_triangle_area(xyz, triangles):
    """有符号球面三角形面积；用符号跟踪翻转，不取绝对值掩盖失效。"""
    a, b, c = np.asarray(xyz)[triangles].transpose(1, 0, 2)
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = 1 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return 2*np.arctan2(numerator, denominator)*R_KM**2


def point_array(points):
    return np.asarray([p.to_xyz() for p in points], float)


def build_network(strip, rotation_model, initial_time, plates=(301, 401), tag="CD", guard_km=400., strain_smoothing="natural_neighbour"):
    """边界点具有稳定GPML身份，分别由真实两侧刚体运动驱动。

    控制节点沿两侧边界分布。采样间距只细化已给定直边，不改变物理W0。
    不设置rift_parameters，不引入额外指数型减薄或应变限幅。
    """
    features, sections = [], []
    grid = strip["grid"]
    center = strip["xyz"][grid[:, grid.shape[1]//2]]
    length = strip["along_km"][-1]
    nside = int(np.ceil((length+2*guard_km)/strip["actual_dy_km"]))+1
    spine = slerp_points(center[0], center[-1], np.linspace(-guard_km/length, 1+guard_km/length, nside))
    half = strip["across_km"][-1]/R_KM
    left = np.cos(half)*spine-np.sin(half)*strip["east"]
    right = np.cos(half)*spine+np.sin(half)*strip["east"]
    rails = np.r_[left, right[::-1]]
    for j, xyz in enumerate(rails):
        plate = plates[0] if j < nside else plates[1]
        initial = pg.PointOnSphere(tuple(xyz))
        present = rotation_model.get_rotation(0., plate, initial_time)*initial
        feature = pg.Feature.create_reconstructable_feature(
            pg.FeatureType.gpml_unclassified_feature, present,
            name=f"{tag}_rail_{j:05d}",
            reconstruction_plate_id=plate, valid_time=(initial_time+1., 0.))
        features.append(feature)
        sections.append(pg.GpmlTopologicalSection.create(feature, topological_geometry_type=pg.GpmlTopologicalNetwork))
    network = pg.Feature.create_topological_network_feature(
        pg.GpmlTopologicalNetwork(sections), name=tag, valid_time=(initial_time+1., 0.))
    model = pg.TopologicalModel(features+[network], rotation_model,
        default_resolve_topology_parameters=pg.ResolveTopologyParameters(enable_strain_rate_clamping=False,
            strain_rate_smoothing=getattr(pg.StrainRateSmoothing,strain_smoothing)))
    return model, features+[network]


def reconstruct_band(strip, model, initial_time, dt, times, natural_neighbour=True):
    """追踪固定材料顶点/三角形中心并保存框架beta，返回独立面积审计。

    H0/J是面积守恒定义，不能单独作为物理解正确的证据；框架beta与
    三角形面积J、网络边界面积分别比较。点在网络外时不悄悄按刚体继续。
    """
    vertices, tri = strip["xyz"], strip["triangles"]
    centers = unit(vertices[tri].sum(axis=1))
    points = np.r_[vertices, centers]
    span = model.reconstruct_geometry([tuple(p) for p in points], initial_time,
        time_increment=dt, deactivate_points=None,
        deformation_uses_natural_neighbour_interpolation=natural_neighbour)
    area0 = spherical_triangle_area(vertices, tri)
    records = []
    snapshots = {}
    for time in times:
        all_xyz = point_array(span.get_geometry_points(float(time), return_inactive_points=True))
        v = all_xyz[:len(vertices)]
        area = spherical_triangle_area(v, tri)
        j = area/area0
        beta = np.asarray(span.get_scalar_values(float(time), pg.ScalarType.gpml_crustal_stretching_factor), float)[len(vertices):]
        snapshot = model.topological_snapshot(float(time))
        locations = snapshot.get_point_locations([tuple(p) for p in all_xyz])
        outside = np.array([p.not_located_in_resolved_topology() for p in locations])
        outside_centers = outside[len(vertices):]
        net = snapshot.get_resolved_topologies()[0]
        net_area = net.get_resolved_geometry().get_area()*R_KM**2
        geometric_area = np.abs(area).sum()
        polygon_area = pg.PolygonOnSphere([tuple(p) for p in v[strip["boundary"]]]).get_area()*R_KM**2
        records.append(dict(t_ma=float(time), area_km2=float(geometric_area),
            network_area_km2=float(net_area), area_ratio=float(geometric_area/abs(area0).sum()),
            material_polygon_area_km2=float(polygon_area),
            material_polygon_area_relative_error=float((geometric_area-polygon_area)/polygon_area),
            inverted_cells=int(np.sum(j<=0)), outside_center_count=int(outside_centers.sum()),
            outside_vertex_count=int(outside[:len(vertices)].sum()),
            outside_center_area_fraction=float(np.sum(abs(area0)[outside_centers])/abs(area0).sum()),
            J_quantiles=np.quantile(j,[0,.1,.5,.9,1]).tolist(),
            framework_beta_vs_J_relative_p90=float(np.quantile(abs(beta/j-1),.9)),
            framework_volume_relative_residual=float(np.sum(abs(area)/beta)/abs(area0).sum()-1)))
        snapshots[float(time)] = dict(xyz=v, center_xyz=all_xyz[len(vertices):], area_km2=abs(area), J=j, framework_beta=beta, outside_centers=outside_centers)
    return records, snapshots
