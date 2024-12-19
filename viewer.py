import pyvista as pv
import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D
from sympy.vector.operators import gradient
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import inspect
import numbers
import sys
import scipy
import scipy.ndimage.filters as filters

sp.init_printing()
pv.global_theme.allow_empty_mesh = True

def shape_operator(f, x, y, R):
    fx = R.i + R.k * sp.diff(f, x)
    fy = R.j + R.k * sp.diff(f, y)
    n = fx.cross(fy)
    n = n/sp.sqrt(n.dot(n))
    L = (R.k * sp.diff(f, x, x)).dot(n)
    M = (R.k * sp.diff(f, x, y)).dot(n)
    N = (R.k * sp.diff(f, y, y)).dot(n)
    E = fx.dot(fx)
    F = fx.dot(fy)
    G = fy.dot(fy)
    I = sp.Matrix([[E, F],[F, G]])
    I_inv = sp.Matrix([[G, -F], [-F, E]])*1/(E*G-F*F)
    II = sp.Matrix([[L, M], [M, N]])
    S = I_inv * II
    return S, I, II

def eval_on_grid(f, xs, ys):
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    f_flat = f(X.ravel(), Y.ravel())
    return f_flat.reshape(X.shape)

def _shape_operator(f, x, y):
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    n = 1/sp.sqrt(1 + fx**2 + fy**2)
    L = n * (sp.diff(f, x, x))
    M = n * sp.diff(f, x, y)
    N = n * sp.diff(f, y, y)
    E = 1 + fx**2
    F = fx*fy
    G = 1 + fy**2
    I = sp.Matrix([[E, F],[F, G]])
    I_inv = sp.Matrix([[G, -F], [-F, E]])*1/(E*G-F*F)
    II = sp.Matrix([[L, M], [M, N]])    # this is in (u,v) coordinates
    S = I_inv * II      # have to transform to r_u, r_v coordinates
    return S, I, II

def principal_dirs(S, x, y, fx, fy, R):
    evecs = S.subs(dict(x=x,y=y)).eigenvects()
    curvatures = []
    vecs = []
    fx = fx.evalf(subs=dict(x=x,y=y))
    fy = fy.evalf(subs=dict(x=x,y=y))
    e1 = R.i + R.k * fx
    e2 = R.j + R.k * fy
    if len(evecs)==1:
        evecs = [evecs[0], evecs[0]]
    for val in evecs:
        curvatures.append(val[0].evalf())
        evec_nat = val[2][0].evalf()
        vecs.append(evec_nat[0,0]*e1 + evec_nat[1,0]*e2)
        # vecs.append(evec_nat)
    return curvatures, vecs

def numeric_adirs(kappas, pdirs):
    hyperbolic_mask = np.prod(kappas,1)<=0
    # w = cos(x) e1 + sin(x) e2
    # k1 cos^2(x) + k2 sin^2(x) = 0
    # k1 cos^2(x) = -k2 sin^2(x)
    # sqrt(k1) cos(x) = sqrt(-k2) sin(x)
    # make sure k1 is the positive one (because sign(k1) != sign(k2))
    # pi-x is a solution if x is (0<=x<=pi)
    pdirs = pdirs/np.sqrt((pdirs**2).sum(1, keepdims=True))
    pos_k1_mask = (kappas[:,0] > 0) * hyperbolic_mask
    k1pos = kappas[pos_k1_mask]
    k1neg = kappas[~pos_k1_mask]
    
    # same size
    theta1_pos = np.arctan(np.sqrt(k1pos[:,0])/np.sqrt(-k1pos[:,1]))
    theta2_pos = np.pi - theta1_pos
    theta_pos = np.vstack([
                            np.maximum(theta1_pos, theta2_pos),
                            np.minimum(theta1_pos, theta2_pos)
                           ]).T
    
    # same size
    theta1_neg = np.arctan(np.sqrt(-k1neg[:,0])/np.sqrt(k1neg[:,1]))
    theta2_neg = np.pi - theta1_neg
    # theta_neg = np.vstack([theta1_neg, theta2_neg]).T
    theta_neg = np.vstack([
                            np.maximum(theta1_neg, theta2_neg),
                            np.minimum(theta1_neg, theta2_neg)
                           ]).T
    
    thetas = np.zeros_like(kappas)
    thetas[pos_k1_mask] = theta_pos
    thetas[~pos_k1_mask] = theta_neg
    components = np.stack([np.cos(thetas), np.sin(thetas)], 1)
    adirs = np.matmul(pdirs, components.data)
    return adirs

def _vectorize_matrix(vars, M):
    """Returns a function that evaluates a 2x2 Sympy matrix on a list of numpy 1-D arrays.

    Parameters
    ----------
    *vars : sympy variables
    M : 2x2 Sympy matrix

    Returns
    -------
    function
        Returns a function that evaluates the result of applying sympy.lambdify followed by 
        np.vectorize on each entry of M, then stacking the results so the output shape is (2,2,-1)
    """
    m00 = np.vectorize(sp.lambdify(vars, M[0,0]))
    m01 = np.vectorize(sp.lambdify(vars, M[0,1]))
    m10 = np.vectorize(sp.lambdify(vars, M[1,0]))
    m11 = np.vectorize(sp.lambdify(vars, M[1,1]))
    def _vectorizedfunc(*args):
        return np.array([[m00(*args), m01(*args)], [m10(*args), m11(*args)]])
    return _vectorizedfunc

def numeric_pdirs(S, f, x, y, X, Y):
    S_num = _vectorize_matrix([x, y], S)
    fx = sp.lambdify([x, y], sp.diff(f, x))
    fy = sp.lambdify([x, y], sp.diff(f, y))
    Ss = S_num(X.ravel(), Y.ravel())
    if Ss.ndim==2:
        Ss = np.repeat(Ss[...,None], X.size, 2)
    Ss = np.moveaxis(Ss, -1, 0)
    fxs = fx(X.ravel(), Y.ravel())
    fys = fy(X.ravel(), Y.ravel())
    pcurvs, evecs = np.linalg.eig(Ss)

    e1 = np.vstack([np.ones_like(fxs),
                    np.zeros_like(fxs),
                    fxs])
    e2 = np.vstack([np.zeros_like(fys),
                    np.ones_like(fys),
                    fys])
    d1 = evecs[:,0,0]*e1 + evecs[:,1,0]*e2
    d2 = evecs[:,0,1]*e1 + evecs[:,1,1]*e2
    pdirs = np.stack([d1.T, d2.T], 2)
    return pcurvs, pdirs

def symbolic_pdirs(S, f, x, y, X, Y):
    R = CoordSys3D('R')
    ret = S.eigenvects()
    if len(ret)==1:
        ret = (ret[0], ret[0])
    k1 = ret[0][0]
    k2 = ret[1][0]
    v1 = ret[0][2][0]
    v2 = ret[1][2][0]
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    e1 = sp.Matrix([[1.0],[0.0],[fx]])
    e2 = sp.Matrix([[0.0],[1.0],[fy]])
    d1_s = v1[0,0]*e1 + v1[1,0]*e2
    d2_s = v2[0,0]*e1 + v2[1,0]*e2
    d1_f = np.vectorize(sp.lambdify([x, y], d1_s), signature='(),()->(2,1)')
    d2_f = np.vectorize(sp.lambdify([x, y], d2_s), signature='(),()->(2,1)')
    d1 = d1_f(X.ravel(), Y.ravel()).squeeze(2)
    d2 = d2_f(X.ravel(), Y.ravel()).squeeze(2)
    return d1, d2

def normal(f, x, y, R):
    fx = -sp.diff(f, x)
    fy = -sp.diff(f, y)
    fz = 1
    norm = sp.sqrt(fx**2 + fy**2 + fz**2)
    n = 1 / norm * (R.i*fx + R.j*fy + R.k*fz)
    return n

def make_glyphs(data, mask=None, **kwargs):
    
    masked_data = data.glyph(**kwargs)
    if mask is None or mask.sum()==0:
        return masked_data
    masked_data = masked_data.remove_cells(mask)
    return masked_data

if __name__=='__main__':
    x, y = sp.symbols('x y')
    f_input = input("Equation: ").strip() if len(sys.argv)==1 else sys.argv[1].strip()
    f = sp.parse_expr(f_input)
    f_num = np.vectorize(sp.lambdify([x, y], f))
    S, I, II = _shape_operator(f, x, y)
    
    xs, ys = np.linspace(-2, 2, 61), np.linspace(-2, 2, 61)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_small, Y_small = np.meshgrid(np.linspace(xs.min(), xs.max(), 15),
                                   np.linspace(ys.min(), ys.max(), 15))
    # Y = np.flipud(Y)
    f_vals = f_num(X, Y)
    points = np.stack([X, Y, f_vals], 2).reshape(-1,3)
    pcurvs, pdirs = numeric_pdirs(S, f, x, y, X, Y)
    poly = pv.PolyData(points)
    surf = pv.StructuredGrid(X, Y, f_vals)
    
    eps = 1e-7
    poly.point_data['d1'] = pdirs[:,:,0]
    poly.point_data['d2'] = pdirs[:,:,1]
    surf.point_data['d1'] = pdirs[:,:,0]
    surf.point_data['d2'] = pdirs[:,:,1]
    surf.point_data['gaussian_k'] = np.prod(pcurvs, -1).reshape(X.shape).T.ravel()
    surf.point_data['kg_under'] = surf.point_data['gaussian_k'] - eps
    surf.point_data['kg_over'] = surf.point_data['gaussian_k'] + eps
    surf.point_data['clasticity'] = np.sum(np.sign(pcurvs),-1).reshape(X.shape).T.ravel()
    surf.point_data['parabolics'] = (np.abs(np.prod(pcurvs, -1))<0.01).reshape(X.shape).T.ravel()
    
    adirs = numeric_adirs(pcurvs, pdirs)
    print(np.sum(np.prod(pdirs,-1),-1).max())
    poly.point_data['a1'] = adirs[:,:,0]
    poly.point_data['a2'] = adirs[:,:,1]
    surf.point_data['a1'] = adirs[:,:,0]
    surf.point_data['a2'] = adirs[:,:,1]
    
    line = pv.Line()
    synclastic_mask = (np.prod(pcurvs, -1) > 0)
    d1_arrows = make_glyphs(poly, mask=synclastic_mask,
                            geom=line, scale=False, factor=0.04, orient='d1')
    d2_arrows = make_glyphs(poly, mask=synclastic_mask,
                            geom=line, scale=False, factor=0.04, orient='d2')
    a1_arrows = make_glyphs(poly, mask=synclastic_mask,
                            geom=line, scale=False, factor=0.04, orient='a1')
    a2_arrows = make_glyphs(poly, mask=synclastic_mask,
                            geom=line, scale=False, factor=0.04, orient='a2')

    # compute ridges
    # ridges are local extrema of principal curvatures when traveling in a principal direction
    
    
    plotter = pv.Plotter()
    # plotter.add_mesh(d1_arrows, color='black', line_width=3)
    # plotter.add_mesh(d2_arrows, color='black', line_width=3)
    plotter.add_mesh(a1_arrows, color='red', line_width=3)
    plotter.add_mesh(a2_arrows, color='red', line_width=3)
    plotter.add_axes()
    # plotter.add_mesh(poly, scalars='min_curv')
    
    plotter.add_mesh(surf, scalars='gaussian_k')
    pos_parabolic = surf.contour([0.0], scalars='kg_over')
    neg_parabolic = surf.contour([0.0], scalars='kg_under')
    parabolic = pos_parabolic.merge(neg_parabolic)
    # plotter.add_mesh(parabolic, color="white", line_width=5)
    # source_pts = pv.PolyData(points[synclastic_mask][::11])
    # streamlines1 = surf.copy().streamlines_from_source(
    #     source_pts,
    #     vectors='a1',
    #     integration_direction='both',
    #     surface_streamlines=True,
    #     initial_step_length=0.01,
    #     step_unit='l',
    #     compute_vorticity=False,
    #     interpolator_type='point'
    # )
    # plotter.add_mesh(streamlines1.tube(radius=0.02), color='red')
    
    plotter.show()