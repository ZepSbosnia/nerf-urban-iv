#!/usr/bin/env python3
"""Convert an OpenSfM reconstruction.json into a torch-ngp transforms.json.

Same math as the former my_scripts/sfm_to_colmap.py (so datasets stay compatible),
plus a "georef" block that stores everything needed to map torch-ngp coordinates
back to OpenSfM's topocentric ENU frame (metres) — see nerf/georef.py.

    python tools/opensfm2nerf.py data/sempione --ext png
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from nerf.georef import F_AXES, rotmat  # noqa: E402


def closest_point_2_lines(oa, da, ob, db):
    # point closest to both rays o+t*d, and a weight that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    ta = min(ta, 0)
    tb = min(tb, 0)
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def shot_to_c2w(shot):
    """OpenSfM shot (world->camera, OpenCV axes, ENU world) -> c2w in the transforms.json world frame
    *before* the up-rotation / recentering / rescaling."""
    R = Rotation.from_rotvec(shot['rotation']).as_matrix()
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = shot['translation']
    c2w = np.linalg.inv(w2c)
    c2w[0:3, 2] *= -1  # OpenCV -> OpenGL camera axes
    c2w[0:3, 1] *= -1
    c2w[:3, :] = F_AXES @ c2w[:3, :]  # swap world x/y and flip z (as in colmap2nerf)
    return c2w


def convert(reconstruction, rec_index=0, ext=None, image_prefix='./images/', aabb_scale=2):
    rec = reconstruction[rec_index]
    if len(reconstruction) > 1:
        print(f'[WARN] {len(reconstruction)} partial reconstructions found, using #{rec_index} '
              f'({len(rec["shots"])} shots). Other reconstructions live in different frames and are skipped.')

    cam = list(rec['cameras'].values())
    if len(cam) > 1:
        print(f'[WARN] {len(cam)} cameras found, intrinsics are taken from the first one.')
    cam = cam[0]
    w, h = cam['width'], cam['height']
    # OpenSfM normalises the focal length by max(width, height)
    fl = cam['focal'] * max(w, h)

    frames = []
    for name, shot in sorted(rec['shots'].items()):
        if ext is not None:
            name = os.path.splitext(name)[0] + '.' + ext.lstrip('.')
        frames.append({'file_path': image_prefix + name, 'transform_matrix': shot_to_c2w(shot)})

    # rotate the average camera up vector to +z
    up = np.sum([f['transform_matrix'][0:3, 1] for f in frames], axis=0)
    up = up / np.linalg.norm(up)
    R_up = rotmat(up, [0, 0, 1])
    for f in frames:
        f['transform_matrix'][:3, :] = R_up @ f['transform_matrix'][:3, :]

    # recenter on the point the cameras look at
    totw, totp = 0.0, np.zeros(3)
    for f in frames:
        mf = f['transform_matrix'][0:3, :]
        for g in frames:
            mg = g['transform_matrix'][0:3, :]
            p, wgt = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if wgt > 1e-5:
                totp += p * wgt
                totw += wgt
    if totw > 0:
        totp /= totw
    for f in frames:
        f['transform_matrix'][0:3, 3] -= totp

    # rescale so the average camera distance is 4 ("nerf sized")
    avglen = np.mean([np.linalg.norm(f['transform_matrix'][0:3, 3]) for f in frames])
    for f in frames:
        f['transform_matrix'][0:3, 3] *= 4.0 / avglen
        f['transform_matrix'] = f['transform_matrix'].tolist()

    out = {
        'camera_angle_x': 2 * np.arctan(w / (2 * fl)),
        'camera_angle_y': 2 * np.arctan(h / (2 * fl)),
        'fl_x': fl, 'fl_y': fl,
        'k1': cam.get('k1', 0.0), 'k2': cam.get('k2', 0.0),
        'cx': w / 2, 'cy': h / 2, 'w': w, 'h': h,
        'aabb_scale': aabb_scale,
        # legacy keys (kept for older scripts)
        'up': up.tolist(), 'totp': totp.tolist(), 'avglen': float(avglen),
        'georef': {
            'R_up': R_up.tolist(),
            'totp': totp.tolist(),
            'avglen': float(avglen),
            'reference_lla': rec.get('reference_lla'),
            'frame': 'OpenSfM topocentric ENU (metres)',
        },
        'frames': frames,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='dataset folder (OpenSfM project)')
    ap.add_argument('--reconstruction', default=None, help='default: <dataset>/reconstruction.json')
    ap.add_argument('--out', default=None, help='default: <dataset>/transforms.json')
    ap.add_argument('--index', type=int, default=0, help='which partial reconstruction to use')
    ap.add_argument('--ext', default=None, help="replace image extension in file_path (e.g. 'png')")
    ap.add_argument('--image_prefix', default='./images/')
    ap.add_argument('--aabb_scale', type=int, default=2)
    args = ap.parse_args()

    rec_path = args.reconstruction or os.path.join(args.dataset, 'reconstruction.json')
    out_path = args.out or os.path.join(args.dataset, 'transforms.json')
    with open(rec_path) as f:
        reconstruction = json.load(f)

    out = convert(reconstruction, args.index, args.ext, args.image_prefix, args.aabb_scale)
    if out['georef']['reference_lla'] is None:
        print('[WARN] no reference_lla in reconstruction: ENU output works, CRS export will not.')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] {len(out['frames'])} frames, avglen={out['georef']['avglen']:.3f} m -> {out_path}")


if __name__ == '__main__':
    main()
