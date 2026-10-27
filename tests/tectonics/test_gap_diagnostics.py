"""只读候选重建与已知连续边界的独立解析检查。"""
import numpy as np
import pytest
from mythic.tectonics.crust import CONT, Crust, Params
from mythic.tectonics.engine import Engine
from mythic.tectonics.geometry import R_KM, omega_rotmat
from mythic.tectonics.mesh import load_mesh
from mythic.tectonics.gap_diagnostics import rift_candidates
from mythic.tectonics.deformation_kinematics import boundary_jump


@pytest.mark.parametrize("slip", [False, True])
def test_readonly_candidates_match_real_commit(slip):
    m=load_mesh(5)
    c=Crust(m.n)
    a=np.array([1.,.4,.2]); a/=np.linalg.norm(a)
    z=np.cross(a,[0.,0.,1.]); z/=np.linalg.norm(z)
    c.plate[:]=np.where(m.xyz@a>0,1,2)
    c.ctype[:]=CONT
    c.T[:]=35.
    c.origin[:]=1
    axis=a if slip else z
    w={1:-.1*axis,2:.1*axis}
    eng=Engine(m,c,Params(rift_area_conservative=True),100.,w)
    eng.R={q:omega_rotmat(omega,1.) for q,omega in w.items()}
    eng.cont_gap_pairs={1002:dict(knots=[(0.,35.),(2000.,20.)],holes=False)}
    eng.gap_prefer={1002:2}
    before={key:getattr(c,key).copy() for key in ("uid","T","node","plate")}
    rows=rift_candidates(eng)
    assert len(rows["gap"])>0
    for key,value in before.items():
        np.testing.assert_array_equal(getattr(c,key),value)
    assert eng.budget.get("rift_transferred",0.)==0.
    eng.commit()
    audits=eng.rift_area_audit
    assert len(audits)==1
    np.testing.assert_array_equal(audits[0]["target_pair"],rows["pair"])
    np.testing.assert_array_equal(audits[0]["target_area_km2"],rows["target_area_km2"])


def test_normal_flux_integral_agrees_with_spherical_lune_area():
    angle=np.linspace(-np.pi/2,np.pi/2,10001)
    x=np.column_stack([np.zeros(len(angle)),np.cos(angle),np.sin(angle)])
    n=np.broadcast_to([-1.,0.,0.],x.shape)
    w=5./R_KM
    normal,slip=boundary_jump(x,n,[0.,0.,-w],[0.,0.,w])
    flux=np.trapezoid(normal,angle)*R_KM
    exact_area_rate=4.*R_KM**2*w
    assert flux==pytest.approx(exact_area_rate,rel=1e-8)
    np.testing.assert_allclose(slip,0.,atol=1e-12)
