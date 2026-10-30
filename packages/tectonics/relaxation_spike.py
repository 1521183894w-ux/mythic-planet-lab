"""U40 有限速率厚度松弛小试；procedural_constrained，非应力或热流变求解。

复用 SciPy 稀疏求解器计算冻结系数后向欧拉步；面积质量矩阵、成对边流。
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def relax_thickness(thickness, area, edges, mobility, dt_myr, h_km,
                    kappa_km2_myr=3000.0, onset_km=40.0, scale_km=25.0):
    """返回厚度、逐单元净体积和诊断；没有截顶、裁零或归一化。

    D=kappa*[max(max(T_i,T_j)-onset,0)/scale]^3*mobility。
    三次方为受通道流动启发的经验曲线，不把T-onset当真实通道厚度。
    调用者筛选存活陆壳原生边；零mobility为不透过边界。
    """
    T=np.asarray(thickness,float); A=np.asarray(area,float)
    e=np.asarray(edges,np.int64).reshape(-1,2); mob=np.asarray(mobility,float)
    if (T.ndim!=1 or A.shape!=T.shape or mob.shape!=(len(e),)
        or not np.isfinite(T).all() or not np.isfinite(A).all() or np.any(A<=0)
        or not np.isfinite(mob).all() or np.any((mob<0)|(mob>1))
        or np.any(e<0) or np.any(e>=len(T))
        or not np.isfinite([dt_myr,h_km,kappa_km2_myr,onset_km,scale_km]).all()
        or dt_myr<0 or h_km<=0 or kappa_km2_myr<0 or scale_km<=0):
        raise ValueError('invalid relaxation input')
    empty=dict(active_edges=0,active_cells=0,balance_km3=0.,
        max_local_residual_km3=0.,max_diffusivity_km2_myr=0.,gross_transport_km3=0.)
    if not len(e) or not dt_myr or not kappa_km2_myr:
        return T.copy(),np.zeros(len(T)),empty
    i,j=e.T
    D=kappa_km2_myr*(np.maximum(np.maximum(T[i],T[j])-onset_km,0)/scale_km)**3*mob
    active=D>0
    if not active.any():return T.copy(),np.zeros(len(T)),empty
    i,j,D=i[active],j[active],D[active]
    ids,local=np.unique(np.r_[i,j],return_inverse=True)
    n=len(i); ii,jj=local[:n],local[n:]
    g=D*2/(3*h_km**2)*.5*(A[i]+A[j])
    L=coo_matrix((np.r_[g,g,-g,-g],(np.r_[ii,jj,ii,jj],np.r_[ii,jj,jj,ii])),
                 shape=(len(ids),len(ids))).tocsc()
    matrix=diags(A[ids],format='csc')+dt_myr*L
    reference=float(np.mean(T[ids]))
    shifted=spsolve(matrix,A[ids]*(T[ids]-reference))
    new=reference+shifted
    if not np.isfinite(new).all():raise RuntimeError('nonfinite sparse solve')
    out=T.copy();out[ids]=new
    # 重算成对边流，独立核对质量矩阵变化。
    flux=dt_myr*g*(shifted[jj]-shifted[ii])
    dv=np.bincount(i,flux,minlength=len(T))-np.bincount(j,flux,minlength=len(T))
    residual=(out-T)*A-dv
    info=dict(active_edges=len(i),active_cells=len(ids),
        balance_km3=float(np.sum((out-T)*A)),
        max_local_residual_km3=float(np.max(np.abs(residual))),
        max_diffusivity_km2_myr=float(D.max()),
        gross_transport_km3=float(np.abs(flux).sum()))
    return out,dv,info
