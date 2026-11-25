"""Georeferencing of torch-ngp outputs.

Coordinate chain (forward, see tools/opensfm2nerf.py and nerf/provider.py):

    ENU (OpenSfM topocentric, metres, origin = reference_lla)
      -> json  = (R_up @ F @ enu - totp) * 4 / avglen                    (transforms.json world)
      -> ngp   = Q @ json * scale + offset                               (nerf_matrix_to_ngp)

so the exact inverse is

    enu = F @ inv(R_up) @ (Q.T @ (ngp - offset) / scale * avglen / 4 + totp)

ENU -> lat/lon/alt uses WGS84 like OpenSfM; lat/lon -> projected CRS uses pyproj.

CLI (for meshes / point clouds already exported by torch-ngp):

    python -m nerf.georef mesh.ply --transforms data/x/transforms.json \
        --crs EPSG:32632 --walls walls_v2.npy --walls_crs EPSG:3857
"""
import json
import os

import numpy as np

# world-axis swap/flip applied by colmap2nerf-style converters (swap x/y, flip z). Self-inverse.
F_AXES = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
# row permutation applied by nerf_matrix_to_ngp: (x, y, z) -> (y, z, x)
Q_NGP = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)

WGS84_A = 6378137.0
WGS84_B = 6356752.314245


def rotmat(a, b):
    """Rotation matrix taking direction a onto direction b (as in colmap2nerf)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -1 + 1e-10:  # opposite directions: perturb deterministically
        return rotmat(a + np.array([1e-2, -1e-2, 1e-2]), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


# --------------------------------------------------------------------------- ngp <-> ENU

def load_georef(transforms_path):
    """Read the georef parameters from a transforms.json.

    New files (tools/opensfm2nerf.py) carry a 'georef' block. Older ones written by
    sfm_to_colmap.py only have top-level 'up', 'totp', 'avglen': R_up is rebuilt from 'up'
    and reference_lla must then be given separately.
    """
    with open(transforms_path) as f:
        t = json.load(f)
    if 'georef' in t:
        g = t['georef']
        return {'R_up': np.array(g['R_up'], float), 'totp': np.array(g['totp'], float),
                'avglen': float(g['avglen']), 'reference_lla': g.get('reference_lla')}
    if all(k in t for k in ('up', 'totp', 'avglen')):
        return {'R_up': rotmat(t['up'], [0, 0, 1]), 'totp': np.array(t['totp'], float),
                'avglen': float(t['avglen']), 'reference_lla': None}
    raise KeyError(f'{transforms_path} has no georef information (run tools/opensfm2nerf.py)')


def ngp_to_enu(pts, georef, scale=0.33, offset=(0, 0, 0)):
    """[N, 3] torch-ngp coordinates -> [N, 3] ENU metres. `scale`/`offset` must be the
    --scale/--offset used for training."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    p = (pts - np.asarray(offset, float)) / scale @ Q_NGP         # Q^T x, row-vector form
    w = p * georef['avglen'] / 4.0 + georef['totp']
    # inv() rather than .T: rotmat() is only orthonormal up to ~1e-8 when `up` is close to -z
    return w @ np.linalg.inv(georef['R_up']).T @ F_AXES          # F R^-1 w, row-vector form


def enu_to_ngp(enu, georef, scale=0.33, offset=(0, 0, 0)):
    """Inverse of ngp_to_enu (e.g. to put map data such as walls into NeRF space)."""
    enu = np.asarray(enu, dtype=np.float64).reshape(-1, 3)
    w = enu @ F_AXES @ georef['R_up'].T                          # R F enu, row-vector form
    p = (w - georef['totp']) * 4.0 / georef['avglen']
    return p @ Q_NGP.T * scale + np.asarray(offset, float)


# --------------------------------------------------------------------------- ENU <-> geodetic

def _ecef_from_lla(lat, lon, alt):
    a2, b2 = WGS84_A ** 2, WGS84_B ** 2
    lat, lon = np.radians(lat), np.radians(lon)
    L = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    x = (a2 * L + alt) * np.cos(lat) * np.cos(lon)
    y = (a2 * L + alt) * np.cos(lat) * np.sin(lon)
    z = (b2 * L + alt) * np.sin(lat)
    return np.stack([x, y, z], -1)


def _lla_from_ecef(x, y, z):
    a, b = WGS84_A, WGS84_B
    ea = np.sqrt((a ** 2 - b ** 2) / a ** 2)
    eb = np.sqrt((a ** 2 - b ** 2) / b ** 2)
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + eb ** 2 * b * np.sin(theta) ** 3, p - ea ** 2 * a * np.cos(theta) ** 3)
    N = a / np.sqrt(1 - ea ** 2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt


def enu_to_lla(enu, reference_lla):
    """[N, 3] ENU metres -> [N, 3] (lat, lon, alt) on WGS84, same convention as OpenSfM."""
    enu = np.asarray(enu, dtype=np.float64).reshape(-1, 3)
    lat0, lon0 = reference_lla['latitude'], reference_lla['longitude']
    alt0 = reference_lla.get('altitude', 0.0)
    sa, ca = np.sin(np.radians(lat0)), np.cos(np.radians(lat0))
    so, co = np.sin(np.radians(lon0)), np.cos(np.radians(lon0))
    # columns: east, north, up expressed in ECEF
    R = np.array([[-so, -sa * co, ca * co],
                  [co, -sa * so, ca * so],
                  [0.0, ca, sa]])
    ecef = enu @ R.T + _ecef_from_lla(lat0, lon0, alt0)
    lat, lon, alt = _lla_from_ecef(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return np.stack([lat, lon, alt], -1)


def lla_to_crs(lla, crs):
    """[N, 3] (lat, lon, alt) -> [N, 3] (x, y, alt) in a projected CRS, e.g. 'EPSG:32632'."""
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    x, y = tr.transform(lla[:, 1], lla[:, 0])
    return np.stack([x, y, lla[:, 2]], -1)


def crs_to_crs(xy, src, dst):
    """[N, 2+] points from CRS `src` to CRS `dst` (extra columns are kept)."""
    from pyproj import Transformer
    xy = np.asarray(xy, dtype=np.float64)
    tr = Transformer.from_crs(src, dst, always_xy=True)
    x, y = tr.transform(xy[:, 0], xy[:, 1])
    return np.concatenate([np.stack([x, y], -1), xy[:, 2:]], -1)


# --------------------------------------------------------------------------- alignment to map walls

def _kabsch_2d(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])  # proper rotation, no mirroring
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def align_to_walls(xy, walls_xy, max_angle_deg=10.0, n_init=7, iters=60, trim=0.6, max_pts=20000, seed=0):
    """Rigid 2D alignment (rotation + translation) of a point set onto wall points with
    trimmed ICP. Only the `trim` fraction of closest correspondences is used each iteration,
    so facade points pull and trees/ground/noise are ignored.

    Returns (R [2,2], t [2], score) such that aligned = xy @ R.T + t; score is the trimmed
    mean distance (CRS units) of the best start.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, np.float64)
    walls_xy = np.asarray(walls_xy, np.float64)
    sub = xy[rng.choice(len(xy), min(max_pts, len(xy)), replace=False)]
    tree = cKDTree(walls_xy)
    c = sub.mean(0)
    best = (np.eye(2), np.zeros(2), np.inf)
    for ang in np.radians(np.linspace(-max_angle_deg, max_angle_deg, n_init)):
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        t = c - R @ c  # rotate around the centroid
        for _ in range(iters):
            cur = sub @ R.T + t
            d, idx = tree.query(cur)
            keep = d <= np.quantile(d, trim)
            dR, dt = _kabsch_2d(cur[keep], walls_xy[idx[keep]])
            R, t = dR @ R, dR @ t + dt
            if np.abs(dt).max() < 1e-4 and abs(dR[1, 0]) < 1e-7:
                break
        d, _ = tree.query(sub @ R.T + t)
        score = d[d <= np.quantile(d, trim)].mean()
        # reject solutions that drifted outside the allowed angle range
        if abs(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) <= max_angle_deg + 1e-6 and score < best[2]:
            best = (R, t, score)
    return best


# --------------------------------------------------------------------------- export

def _save_ply(geom, vertices, path):
    import trimesh
    if isinstance(geom, trimesh.Trimesh):
        out = trimesh.Trimesh(vertices=vertices, faces=geom.faces, process=False,
                              vertex_colors=geom.visual.vertex_colors if geom.visual.kind == 'vertex' else None)
    else:
        out = trimesh.PointCloud(vertices, colors=getattr(geom, 'colors', None))
    out.export(path)


def georef_export(ply_path, transforms_path, scale=0.33, offset=(0, 0, 0), crs='EPSG:32632',
                  reference_lla=None, walls=None, walls_crs='EPSG:3857', max_angle_deg=10.0,
                  z_band=None, log=print):
    """Write ENU / CRS / wall-aligned versions of a torch-ngp mesh or point cloud.

    Outputs, next to `ply_path`:
      *_enu.ply                 ENU metres around reference_lla (float32 is fine here)
      *_<crs>.ply + _origin.json CRS coordinates minus a rounded origin (PLY stores float32,
                                which would lose ~0.5 m on UTM coordinates)
      *_<crs>.npz               full-precision float64 CRS vertices
      *_<crs>_aligned.*         same, after rigid 2D alignment to `walls` (if given)
    """
    import trimesh
    g = load_georef(transforms_path)
    if reference_lla is not None:
        g['reference_lla'] = reference_lla
    geom = trimesh.load(ply_path, process=False)
    v = np.asarray(geom.vertices, np.float64)
    base = os.path.splitext(ply_path)[0]
    out = {}

    enu = ngp_to_enu(v, g, scale, offset)
    out['enu'] = base + '_enu.ply'
    _save_ply(geom, enu, out['enu'])
    log(f'[georef] ENU -> {out["enu"]}')

    if g['reference_lla'] is None:
        log('[georef] no reference_lla: stopping at ENU.')
        return out

    xyz = lla_to_crs(enu_to_lla(enu, g['reference_lla']), crs)
    tag = crs.replace(':', '').lower()

    def save_crs(pts, suffix, extra=None):
        origin = np.round(pts.mean(0) / 1000.0) * 1000.0
        origin[2] = 0.0
        p = f'{base}_{tag}{suffix}'
        _save_ply(geom, pts - origin, p + '.ply')
        np.savez_compressed(p + '.npz', xyz=pts, crs=crs)
        meta = {'crs': crs, 'origin': origin.tolist(), 'note': 'ply vertices = true coordinates - origin'}
        meta.update(extra or {})
        with open(p + '_origin.json', 'w') as f:
            json.dump(meta, f, indent=2)
        log(f'[georef] {crs} -> {p}.ply (origin {origin[:2].tolist()})')
        return p + '.ply'

    out['crs'] = save_crs(xyz, '')

    if walls is not None:
        w = np.load(walls) if isinstance(walls, str) else np.asarray(walls)
        w = crs_to_crs(w[:, :2], walls_crs, crs) if walls_crs != crs else w[:, :2]
        sel = np.ones(len(xyz), bool)
        if z_band is not None:  # e.g. (1, 10): only facade points between 1 and 10 m above the reference
            sel = (xyz[:, 2] > z_band[0]) & (xyz[:, 2] < z_band[1])
        R, t, score = align_to_walls(xyz[sel, :2], w, max_angle_deg)
        aligned = xyz.copy()
        aligned[:, :2] = xyz[:, :2] @ R.T + t
        ang = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        c = xyz[sel, :2].mean(0)
        shift = float(np.linalg.norm(R @ c + t - c))
        log(f'[georef] wall alignment: rotation {ang:+.2f} deg, centroid shift {shift:.2f}, '
            f'trimmed mean distance {score:.2f} ({crs} units)')
        out['aligned'] = save_crs(aligned, '_aligned',
                                  {'R': R.tolist(), 't': t.tolist(), 'rotation_deg': ang, 'score': float(score)})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ply', help='mesh / point cloud exported by torch-ngp (NeRF coordinates)')
    ap.add_argument('--transforms', required=True, help='transforms.json used for training')
    ap.add_argument('--scale', type=float, default=0.33, help='--scale used for training')
    ap.add_argument('--offset', type=float, nargs=3, default=[0, 0, 0], help='--offset used for training')
    ap.add_argument('--crs', default='EPSG:32632')
    ap.add_argument('--reference_lla', default=None,
                    help='reconstruction.json or reference_lla.json, if transforms.json lacks it')
    ap.add_argument('--walls', default=None, help='.npy of wall points [N, 2+]')
    ap.add_argument('--walls_crs', default='EPSG:3857')
    ap.add_argument('--max_angle', type=float, default=10.0)
    ap.add_argument('--z_band', type=float, nargs=2, default=None)
    a = ap.parse_args()

    ref = None
    if a.reference_lla:
        with open(a.reference_lla) as f:
            j = json.load(f)
        ref = j[0]['reference_lla'] if isinstance(j, list) else j.get('reference_lla', j)
    georef_export(a.ply, a.transforms, a.scale, a.offset, a.crs, ref, a.walls, a.walls_crs, a.max_angle, a.z_band)


if __name__ == '__main__':
    main()
