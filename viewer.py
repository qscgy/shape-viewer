import pyvista as pv
import numpy as np
import scipy.interpolate
import scipy.signal
import sympy as sp
from sympy.vector import CoordSys3D
from sympy.vector.operators import gradient
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import scipy
import scipy.ndimage.filters as filters
import yaml
from utils import _set_config_defaults
import einops as eo

sp.init_printing()
pv.global_theme.allow_empty_mesh = True

def _remove_from_grid(grid: pv.StructuredGrid, mask: np.ndarray):
    h, w, _ = grid.dimensions
    inds = np.arange(h * w).astype(float)
    inds[mask] = np.nan     # make masked inds nan to check later
    inds = inds.reshape(h, w)
    four = np.ones((h-1, w-1)) * 4
    cells = np.stack([
        four,
        inds[:-1,:-1],
        inds[1:,:-1],
        inds[1:,1:],
        inds[:-1,1:]
    ], 2).reshape(-1, 5)
    cell_mask = np.any(np.isnan(cells), axis=1)
    poly = pv.PolyData(grid.points, cells[~cell_mask].astype(int))
    for k, v in grid.point_data.items():
        poly.point_data[k] = v
    return poly

def shape_operator(f, x, y, R):
    fx = R.i + R.k * sp.diff(f, x)
    fy = R.j + R.k * sp.diff(f, y)
    n = fx.cross(fy)
    n = n / sp.sqrt(n.dot(n))
    L = (R.k * sp.diff(f, x, x)).dot(n)
    M = (R.k * sp.diff(f, x, y)).dot(n)
    N = (R.k * sp.diff(f, y, y)).dot(n)
    E = fx.dot(fx)
    F = fx.dot(fy)
    G = fy.dot(fy)
    I = sp.Matrix([[E, F], [F, G]])
    I_inv = sp.Matrix([[G, -F], [-F, E]]) * 1 / (E * G - F * F)
    II = sp.Matrix([[L, M], [M, N]])
    S = I_inv * II
    return S, I, II

def eval_on_grid(f, xs, ys):
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    f_flat = f(X.ravel(), Y.ravel())
    return f_flat.reshape(X.shape)

def _shape_operator(f, x, y):
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    n = 1 / sp.sqrt(1 + fx**2 + fy**2)
    L = n * (sp.diff(f, x, x))
    M = n * sp.diff(f, x, y)
    N = n * sp.diff(f, y, y)
    E = 1 + fx**2
    F = fx * fy
    G = 1 + fy**2
    I = sp.Matrix([[E, F], [F, G]])
    I_inv = sp.Matrix([[G, -F], [-F, E]]) * 1 / (E * G - F * F)
    II = sp.Matrix([[L, M], [M, N]])  # this is in (u,v) coordinates
    S = I_inv * II  # have to transform to r_u, r_v coordinates
    return S, I, II


def principal_dirs(S, x, y, fx, fy, R):
    evecs = S.subs(dict(x=x, y=y)).eigenvects()
    curvatures = []
    vecs = []
    fx = fx.evalf(subs=dict(x=x, y=y))
    fy = fy.evalf(subs=dict(x=x, y=y))
    e1 = R.i + R.k * fx
    e2 = R.j + R.k * fy
    if len(evecs) == 1:
        evecs = [evecs[0], evecs[0]]
    for val in evecs:
        curvatures.append(val[0].evalf())
        evec_nat = val[2][0].evalf()
        vecs.append(evec_nat[0, 0] * e1 + evec_nat[1, 0] * e2)
        # vecs.append(evec_nat)
    return curvatures, vecs


def numeric_adirs(kappas, pdirs):
    # w = cos(x) e1 + sin(x) e2
    # k1 cos^2(x) + k2 sin^2(x) = 0
    # k1 cos^2(x) = -k2 sin^2(x)
    # sqrt(k1) cos(x) = sqrt(-k2) sin(x)
    # make sure k1 is the positive one (because sign(k1) != sign(k2))
    # pi-x is a solution if x is (0<=x<=pi)
    pdirs = pdirs / np.sqrt((pdirs**2).sum(1, keepdims=True))
    hyperbolic_mask = (np.prod(kappas, 1) < 0)[:, None].repeat(2, 1)
    hyper_kappas = np.ma.array(kappas, mask=~hyperbolic_mask)
    thetas = np.arctan(np.sqrt(np.abs(hyper_kappas[:, 0] / (1e-6 + hyper_kappas[:, 1]))))
    thetas = np.vstack([thetas, -thetas]).T
    thetas[~hyperbolic_mask] = np.nan
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
    m00 = np.vectorize(sp.lambdify(vars, M[0, 0]))
    m01 = np.vectorize(sp.lambdify(vars, M[0, 1]))
    m10 = np.vectorize(sp.lambdify(vars, M[1, 0]))
    m11 = np.vectorize(sp.lambdify(vars, M[1, 1]))

    def _vectorizedfunc(*args):
        return np.array([[m00(*args), m01(*args)], [m10(*args), m11(*args)]])

    return _vectorizedfunc


def numeric_pdirs(S, f, x, y, X, Y):
    S_num = np.vectorize(sp.lambdify([x, y], S), signature="(),()->(2,2)")
    fx = sp.lambdify([x, y], sp.diff(f, x))
    fy = sp.lambdify([x, y], sp.diff(f, y))
    f_num = sp.lambdify([x, y], f)
    fs = np.vstack([X.ravel(), Y.ravel(), f_num(X.ravel(), Y.ravel())]).T
    Ss = S_num(X.ravel(), Y.ravel())
    if Ss.ndim == 2:
        Ss = np.repeat(Ss[..., None], X.size, 2)
    fxs = fx(X.ravel(), Y.ravel())
    fys = fy(X.ravel(), Y.ravel())
    pcurvs, evecs = np.linalg.eig(Ss)
    idx = pcurvs.argsort(axis=1)
    pcurvs = np.take_along_axis(pcurvs, idx, 1)
    evecs = np.take_along_axis(evecs.swapaxes(1, 2), idx[..., None], 1).swapaxes(1, 2)

    e1 = np.vstack([np.ones_like(fxs), np.zeros_like(fxs), fxs])
    e2 = np.vstack([np.zeros_like(fys), np.ones_like(fys), fys])
    d1 = evecs[:, 0, 0] * e1 + evecs[:, 1, 0] * e2
    d2 = evecs[:, 0, 1] * e1 + evecs[:, 1, 1] * e2
    pdirs = np.stack([d1.T, d2.T], 2)
    normals = np.cross(pdirs[..., 0], pdirs[..., 1])
    pdirs[normals[..., 2] < 0, :, 1] *= -1
    normals[normals[..., 2] < 0] *= -1
    xys = (fs - np.min(fs, 0, keepdims=True)) * np.array([[1.0, 1.0, 0.0]])
    pdirs[
        np.cross(xys, (pdirs[..., 0]))[..., 2] < 0
    ] *= -1  # enforce chirality condition so flows can be computed
    return pcurvs, pdirs, normals


def symbolic_pdirs(S, f, x, y, X, Y):
    R = CoordSys3D("R")
    ret = S.eigenvects()
    if len(ret) == 1:
        ret = (ret[0], ret[0])
    k1 = ret[0][0]
    k2 = ret[1][0]
    v1 = ret[0][2][0]
    v2 = ret[1][2][0]
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    e1 = sp.Matrix([[1.0], [0.0], [fx]])
    e2 = sp.Matrix([[0.0], [1.0], [fy]])
    d1_s = v1[0, 0] * e1 + v1[1, 0] * e2
    d2_s = v2[0, 0] * e1 + v2[1, 0] * e2
    d1_f = np.vectorize(sp.lambdify([x, y], d1_s), signature="(),()->(2,1)")
    d2_f = np.vectorize(sp.lambdify([x, y], d2_s), signature="(),()->(2,1)")
    d1 = d1_f(X.ravel(), Y.ravel()).squeeze(2)
    d2 = d2_f(X.ravel(), Y.ravel()).squeeze(2)
    return d1, d2


def normal(f, x, y, R):
    fx = -sp.diff(f, x)
    fy = -sp.diff(f, y)
    fz = 1
    norm = sp.sqrt(fx**2 + fy**2 + fz**2)
    n = 1 / norm * (R.i * fx + R.j * fy + R.k * fz)
    return n

def make_glyphs(data, mask=None, **kwargs):
    partial_data = data.remove_cells(mask)
    masked_data = partial_data.glyph(**kwargs)
    if mask is None or mask.sum() == 0:
        return masked_data
    print(masked_data)
    return masked_data

def find_ridges(X, Y, zs, pdirs, pcurvs):
    pdirs = pdirs/np.sqrt((pdirs**2).sum(1, keepdims=True))
    pdirs = pdirs.reshape(zs.shape[0], zs.shape[1], 3, 2)
    pcurvs = pcurvs.reshape(zs.shape[0], zs.shape[1], 2)
    ridges = np.zeros((2, zs.shape[0], zs.shape[1]), dtype=bool)
    step = np.abs(X[0,0]-X[0,1])
    ridge_pts = []
    
    for i in range(2):
        k = pcurvs[...,i]
        dsteps = pdirs[...,i] * step
        interp = scipy.interpolate.RegularGridInterpolator((Y[:,0], X[0]),
                                                        values=k,
                                                        method='linear',
                                                        fill_value=None,
                                                        bounds_error=False)
        XY = np.stack([X, Y], 2)
        d1pos = interp(np.flip(XY + dsteps[...,:2], axis=2))
        d1neg = interp(np.flip(XY - dsteps[...,:2], axis=2))
        ridges[i][np.sign(k-d1pos) * np.sign(k-d1neg) == 1] = 1
        ridge_inds = np.where(ridges[i])
        ridge_pts.append(np.stack([X, Y, zs], 2)[ridge_inds[0], ridge_inds[1]])
    return ridge_pts

def asym_map(adirs, poly: pv.StructuredGrid = None):
    asymp_mask = ~np.isnan(adirs[:,0,0])
    adirs = adirs[asymp_mask] / np.linalg.norm(adirs[asymp_mask], axis=1, keepdims=True)
    adirs_double = np.vstack([adirs, -adirs])
    if poly is None:
        map1 = pv.PolyData(adirs_double[:,:,0])
        map2 = pv.PolyData(adirs_double[:,:,1])
        return map1, map2
    else:
        poly1 = poly.copy()
        poly2 = poly.copy()
        poly1.points[asymp_mask] = adirs[:,:,0]
        poly2.points[asymp_mask] = adirs[:,:,1]
        poly1 = _remove_from_grid(poly1, ~asymp_mask)
        poly2 = _remove_from_grid(poly2, ~asymp_mask)
        poly1_ = poly1.copy()
        poly2_ = poly2.copy()
        poly1_.points = -poly1.points
        poly2_.points = -poly2.points
        poly1 = poly1.merge(poly1_)
        poly2 = poly2.merge(poly2_)
        return poly1, poly2

def find_flecnodes(zs, pdirs, pcurvs, streamlines1, streamlines2):
    pdirs = pdirs/np.sqrt((pdirs**2).sum(1, keepdims=True))
    pdirs = pdirs.reshape(zs.shape[0], zs.shape[1], 3, 2)
    pcurvs = pcurvs.reshape(zs.shape[0], zs.shape[1], 2)
    flecnodes = []
    
    for i in range(2):
        k = pcurvs[...,i]
        if i==0:
            streamline = streamlines1
        else:
            streamline = streamlines2

        for j in range(streamline.n_cells):
            line_pts = streamline.get_cell(j).points
            tangents = np.diff(line_pts, axis=1)
            tangents = tangents/np.sqrt((tangents**2).sum(1, keepdims=True))
            angles = 1 - (tangents[1:]*tangents[:-1]).sum(1)
            ext_idx = np.where(np.diff(np.sign(angles)))[0] + 2
            flecnodes.append(line_pts[ext_idx])
    flecnodes = np.vstack(flecnodes)
    print(flecnodes)
    return flecnodes

def run_viewer(cfg):
    x, y = sp.symbols("x y")
    f_input = cfg["surface"]["equation"].strip()
    f = sp.parse_expr(f_input)
    f_num = np.vectorize(sp.lambdify([x, y], f))
    S, I, II = _shape_operator(f, x, y)

    xmin, xmax, ymin, ymax = (
        cfg["lims"]["xmin"],
        cfg["lims"]["xmax"],
        cfg["lims"]["ymin"],
        cfg["lims"]["ymax"],
    )
    xs, ys = np.linspace(xmin, xmax, cfg["lims"]["steps"]), np.linspace(
        ymin, ymax, cfg["lims"]["steps"]
    )
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    f_vals = f_num(X, Y)
    pcurvs, pdirs, normals = numeric_pdirs(S, f, x, y, X, Y)
    points = np.stack([X, Y, f_vals], 2).reshape(-1, 3)
    # points = np.stack([X, Y, pcurvs[:,0].reshape(X.shape)], 2).reshape(-1, 3)
    poly = pv.PolyData(points)
    surf = pv.StructuredGrid(Y, X, f_vals.T)
    # surf = pv.StructuredGrid(Y, X, (pcurvs[:,0].reshape(X.shape)).T)

    eps = 1e-7
    poly.point_data["normals"] = normals/np.sqrt((normals**2).sum(1, keepdims=True))
    poly.point_data["d1"] = pdirs[:, :, 0]
    poly.point_data["d2"] = pdirs[:, :, 1]
    surf.point_data["d1"] = pdirs[:, :, 0]
    surf.point_data["d2"] = pdirs[:, :, 1]
    surf.point_data["gaussian_k"] = np.prod(pcurvs, -1)
    surf.point_data["kg_under"] = surf.point_data["gaussian_k"] - eps   # used to avoid numerical instability when computing parabolic curves
    surf.point_data["kg_over"] = surf.point_data["gaussian_k"] + eps
    surf.point_data["clasticity"] = np.sum(np.sign(pcurvs), -1)
    surf.point_data["mean_k"] = 0.5 * np.sum(pcurvs, 1)
    surf.point_data["k1"] = pcurvs[:,0]
    surf.point_data["k2"] = pcurvs[:,1]
    
    k1 = np.max(pcurvs, axis=1)
    k2 = np.min(pcurvs, axis=1)
    surf.point_data['shape_index'] = 2/np.pi * np.arctan((k2+k1)/(k2-k1+1e-8))

    adirs = numeric_adirs(pcurvs, pdirs)
    poly.point_data["a1"] = adirs[:, :, 0]
    poly.point_data["a2"] = adirs[:, :, 1]
    surf.point_data["a1"] = adirs[:, :, 0]
    surf.point_data["a2"] = adirs[:, :, 1]
    
    if cfg['other_maps']['gaussmap']:
        gaussmap = pv.Plotter()
        gauss_poly = pv.PolyData(poly['normals'])
        gauss_poly['shape_index'] = surf.point_data['shape_index']
        gaussmap.add_axes()
        gaussmap.add_mesh(
            gauss_poly,
            point_size=10,
            scalars='shape_index'
        )
        gaussmap.show()
    
    if cfg['other_maps']['asymptotic_spherical_map']:
        asmap1, asmap2 = asym_map(adirs, poly=surf)
        asplot = pv.Plotter()
        asplot.add_axes()
        # asplot.add_mesh(asmap1, color=cfg['asymptotic_dirs']['colors'][0])
        # asplot.add_mesh(asmap2, color=cfg['asymptotic_dirs']['colors'][1])
        asplot.add_mesh(asmap1)
        asplot.add_mesh(asmap2)
        asplot.show()
    if cfg['surface']['plot']:
        plotter = pv.Plotter()
        plotter.add_axes()
        if cfg["surface"]["scalars"]=='shape_index':
            clims = [-1, 1]
        else:
            clims = None
            
        plotter.add_mesh(surf, scalars=cfg["surface"]["scalars"], cmap='jet', clim=clims)
        # plotter.add_mesh(pv.StructuredGrid(Y, X, pcurvs[:,0].reshape(X.shape).T))


        # line = pv.Line().tube(radius=0.03)
        line=None
        synclastic_mask = np.prod(pcurvs, -1) > 0
        if cfg["normals"]["plot"]:
            normal_arrows = poly.glyph(orient="normals", scale=False, factor=0.04)
            plotter.add_mesh(normal_arrows, color=cfg["normals"]["color"])
        if cfg["principal_dirs"]["plot"]:
            d1_arrows = poly.glyph(geom=line, scale=False, factor=0.04, orient="d1")
            d2_arrows = poly.glyph(geom=line, scale=False, factor=0.04, orient="d2")
            plotter.add_mesh(
                d1_arrows, color=cfg["principal_dirs"]["colors"][0], line_width=3
            )
            plotter.add_mesh(
                d2_arrows, color=cfg["principal_dirs"]["colors"][1], line_width=3
            )
        if cfg["asymptotic_dirs"]["plot"]:
            a1_arrows = make_glyphs(
                poly, mask=synclastic_mask, geom=line, scale=False, factor=0.04, orient="a1"
            )
            a2_arrows = make_glyphs(
                poly, mask=synclastic_mask, geom=line, scale=False, factor=0.04, orient="a2"
            )
            plotter.add_mesh(
                a1_arrows, color=cfg["asymptotic_dirs"]["colors"][0], line_width=3
            )
            plotter.add_mesh(
                a2_arrows, color=cfg["asymptotic_dirs"]["colors"][1], line_width=3
            )

        if cfg['other_maps']["parabolic_curves"]:
            pos_parabolic = surf.contour([0.0], scalars="kg_over")
            neg_parabolic = surf.contour([0.0], scalars="kg_under")
            parabolic = pos_parabolic.merge(neg_parabolic)
            plotter.add_mesh(parabolic, color="white", line_width=5)

        if cfg["principal_dirs"]["draw_curves"]:
            source_mask = np.zeros(X.shape).astype(bool)
            # print(source_mask.shape, poly.point_data['d1'].shape)
            source_mask[::7, ::7] = True
            source_mask_flat = source_mask.ravel()
            source_pts = pv.PolyData(points[source_mask_flat])
            
            # plt.streamplot(
            #     X[::7,::7],
            #     Y[::7,::7],
            #     poly.point_data['d1'].reshape(*X.shape, 3)[::7,::7,0],
            #     poly.point_data['d1'].reshape(*X.shape, 3)[::7,::7,1],
            # )
            # plt.show()
            
            streamlines1 = surf.copy().streamlines_from_source(
                source_pts,
                vectors="d1",
                integration_direction="both",
                surface_streamlines=True,
                initial_step_length=0.1,
                step_unit="cl",
                compute_vorticity=False,
                interpolator_type="point",
                max_steps=3000,
            )
            streamlines2 = surf.copy().streamlines_from_source(
                source_pts,
                vectors="d2",
                integration_direction="both",
                surface_streamlines=True,
                initial_step_length=0.1,
                step_unit="cl",
                compute_vorticity=False,
                interpolator_type="point",
                max_steps=3000,
            )
            plotter.add_mesh(
                streamlines1.tube(radius=(xmax - xmin) / (len(xs) * 10)),
                color=cfg["principal_dirs"]["colors"][0],
            )
            plotter.add_mesh(
                streamlines2.tube(radius=(xmax - xmin) / (len(xs) * 10)),
                color=cfg["principal_dirs"]["colors"][1],
            )
            
        if cfg['other_maps']['ridges']:
            # ridges are local extrema of principal curvatures when traveling in a principal direction
            ridge_pts1, ridge_pts2 = find_ridges(X, Y, f_vals, pdirs, pcurvs)
            ridge_poly1 = pv.PolyData(ridge_pts1)
            plotter.add_mesh(ridge_poly1, point_size=15, color=cfg['principal_dirs']['colors'][0])
            ridge_poly2 = pv.PolyData(ridge_pts2)
            plotter.add_mesh(ridge_poly2, point_size=15, color=cfg['principal_dirs']['colors'][1])
        
        if cfg["asymptotic_dirs"]["draw_curves"]:
            source_mask = np.zeros(X.shape).astype(bool)
            source_mask[::3, ::3] = True
            source_mask = source_mask.ravel()
            source_pts = pv.PolyData(points[source_mask][~synclastic_mask[source_mask]])
            streamlines1 = surf.copy().streamlines_from_source(
                source_pts,
                vectors="a1",
                integration_direction="both",
                surface_streamlines=True,
                initial_step_length=0.03,
                step_unit="cl",
                compute_vorticity=False,
                interpolator_type="point",
                max_steps=2000,
            )
            streamlines2 = surf.copy().streamlines_from_source(
                source_pts,
                vectors="a2",
                integration_direction="both",
                surface_streamlines=True,
                initial_step_length=0.03,
                step_unit="cl",
                compute_vorticity=False,
                interpolator_type="point",
                max_steps=2000,
            )
            plotter.add_mesh(
                streamlines1.tube(radius=(xmax - xmin) / (len(xs) * 10)),
                color=cfg["asymptotic_dirs"]["colors"][0],
            )
            plotter.add_mesh(
                streamlines2.tube(radius=(xmax - xmin) / (len(xs) * 10)),
                color=cfg["asymptotic_dirs"]["colors"][1],
            )
        
            if cfg['other_maps']['flecnodes']:
                flecnodes = find_flecnodes(f_vals, pdirs, pcurvs, streamlines1, streamlines2)
                flec_pts = pv.PolyData(flecnodes)
                plotter.add_mesh(flec_pts, scalars=None, point_size=15, color='red')

        plotter.show()

if __name__=="__main__":
    with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)
    _set_config_defaults(config)
    run_viewer(config)