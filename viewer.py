import pyvista as pv
import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D
from sympy.vector.operators import gradient
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sp.init_printing()

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
    II = sp.Matrix([[L, M], [M, N]])
    S = I_inv * II
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

def numeric_pdirs(S, f, x, y, X, Y):
    S_num = sp.lambdify([x, y], S)
    fx = sp.lambdify([x, y], sp.diff(f, x))
    fy = sp.lambdify([x, y], sp.diff(f, y))
    Ss = S_num(X.ravel(), Y.ravel())
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

def normal(f, x, y, R):
    fx = -sp.diff(f, x)
    fy = -sp.diff(f, y)
    fz = 1
    norm = sp.sqrt(fx**2 + fy**2 + fz**2)
    n = 1 / norm * (R.i*fx + R.j*fy + R.k*fz)
    return n

if __name__=='__main__':
    x, y = sp.symbols('x y')
    f = x**2*y + y**2
    f_num = np.vectorize(sp.lambdify([x, y], f))
    S, I, II = _shape_operator(f, x, y)
    
    xs, ys = np.linspace(-1, 1, 51), np.linspace(-1,1,51)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_small, Y_small = np.meshgrid(np.linspace(xs.min(), xs.max(), 15),
                                   np.linspace(ys.min(), ys.max(), 15))
    Y = np.flipud(Y)
    f_vals = f_num(X, Y)
    points = np.stack([X, Y, f_vals], 2).reshape(-1,3)
    pcurvs, pdirs = numeric_pdirs(S, f, x, y, X, Y)
    poly = pv.PolyData(points)
    surf = pv.StructuredGrid(X, Y, f_vals)
    poly['d1'] = pdirs[:,:,0]
    poly['d2'] = pdirs[:,:,1]
    surf['gaussian_k'] = np.prod(pcurvs, -1).reshape(X.shape).T.ravel()
    line = pv.Line()
    print(pv.examples.download_crater_topo())
    d1_arrows = poly.glyph(geom=line, scale=False, factor=0.07, orient='d1', tolerance=0.02)
    d2_arrows = poly.glyph(geom=line, scale=False, factor=0.07, orient='d2', tolerance=0.02)
    plotter = pv.Plotter()
    plotter.add_mesh(d1_arrows, color='black', line_width=3)
    plotter.add_mesh(d2_arrows, color='black', line_width=3)
    # plotter.add_mesh(poly, scalars='min_curv')
    
    plotter.add_mesh(surf, scalars='gaussian_k')
    grid_point_data = surf.cell_data_to_point_data()
    plotter.add_mesh(grid_point_data.contour(), color="white", line_width=5)
    plotter.show()