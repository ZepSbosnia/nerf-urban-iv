# nerf-urban-ivr

NeRF for **urban geometry reconstruction**, built on top of [torch-ngp](https://github.com/ashawkey/torch-ngp) (a PyTorch implementation of instant-ngp).

The goal is not novel-view synthesis but **clean, georeferenced geometry** of streets and building facades from casually captured photos (smartphone, GPS-tagged EXIF), reconstructed with OpenSfM and then refined with a NeRF.

Main differences from upstream torch-ngp:

* **View-independent colour.** The view direction is removed from the colour network, so appearance cannot explain away geometry (no view-dependent effects baked into wrong densities).
* **Density regularisers.** Minimum-support, total-variation, L1, "extreme" and mip-NeRF 360 distortion losses on the density, plus an optional Sobel-weighted photometric loss.
* **Density activations.** Alternatives to `trunc_exp` (thresholded "binary" density, sigmoids) to push the density towards empty/occupied.
* **Geometry extraction.** Besides marching cubes: *first-hit* point clouds along the training rays, largest-connected-component meshes, meshes in a road-aligned box, horizontal slices.
* **Analysis tools.** Density histograms, density profiles along rays, depth ("distance") images, density evolution across epochs.
* **Georeferencing.** Exact mapping from NeRF coordinates back to OpenSfM's metric ENU frame, then to a projected CRS (default UTM 32N), and rigid 2D alignment to building footprints from a map.

The other torch-ngp models (TensoRF, CCNeRF, D-NeRF, SDF) are still in the tree but untouched; see the [upstream README](https://github.com/ashawkey/torch-ngp#readme) for them.

## Install

```bash
git clone --recursive https://github.com/ZepSbosnia/nerf-urban-ivr.git
cd nerf-urban-ivr
# already cloned without --recursive? fetch the cutlass submodule (needed by --ff):
git submodule update --init

pip install -r requirements.txt          # or: conda env create -f environment.yml
# optional, prebuild the CUDA extensions (otherwise they are JIT-compiled on first run)
bash scripts/install_ext.sh
```

Requirements are those of upstream torch-ngp (PyTorch with CUDA). `pyproj` is only needed for the georeferencing step.

## Pipeline

```
photos (GPS EXIF) ─► OpenSfM ─► tools/opensfm2nerf.py ─► main_nerf.py (train) ─► meshes / point clouds
                                        │                                              │
                                        └── transforms.json (+ "georef" block) ────────┴─► ENU / UTM / aligned to map walls
```

### 1. Structure from motion (OpenSfM)

Run OpenSfM on `data/<scene>/images` (`extract_metadata`, `detect_features`, `match_features`, `create_tracks`, `reconstruct`). With GPS in the EXIF, OpenSfM works in a topocentric ENU frame (metres) around `reference_lla`.

### 2. OpenSfM → torch-ngp

```bash
python tools/opensfm2nerf.py data/<scene> [--ext png] [--index 0]
```

Writes `data/<scene>/transforms.json`. Besides the usual intrinsics and per-frame `transform_matrix`, it stores a `georef` block (`R_up`, `totp`, `avglen`, `reference_lla`) that makes the NeRF coordinates invertible back to metres.
`--ext` changes the image extension (e.g. when images were converted to PNG with a transparent sky).

### 3. Train

```bash
python main_nerf.py data/<scene> -O --workspace results/<scene> --max_epoch 101 --MS_lambda 1e-2
```

`-O` = `--fp16 --cuda_ray --preload`. Training saves checkpoints and validation images only; geometry is exported with the flags below (they run right after training, or on the latest checkpoint with `--plot` / `--test`).

> **Workspace naming.** The workspace is extended automatically with the density activation and the active losses, e.g. `--workspace results/x --MS_lambda 1e-2` writes to `results/x/original/_MS_1e-2`, and with no extra loss to `results/x/original/default`. Pass the same flags again to resume or analyse a run.

### 4. Export geometry

```bash
# first point along each training ray whose density is above --mesh_thresh (coloured point cloud)
python main_nerf.py data/<scene> -O --workspace results/<scene> --MS_lambda 1e-2 --plot --first_point --mesh_thresh 9

# marching-cubes mesh (+ test video)
python main_nerf.py data/<scene> -O --workspace results/<scene> --MS_lambda 1e-2 --test --mesh_res 512 --mesh_thresh 9
```

### 5. Georeference

Add `--georef` to any export command:

```bash
python main_nerf.py data/<scene> -O --workspace results/<scene> --MS_lambda 1e-2 --plot --first_point \
    --georef --walls data/<scene>/walls_v2.npy --walls_crs EPSG:3857 --align_z_band 1 10
```

or run it on a `.ply` that was already exported:

```bash
python -m nerf.georef results/<scene>/.../meshes/xxx.ply --transforms data/<scene>/transforms.json \
    [--reference_lla data/<scene>/reconstruction.json] [--walls walls_v2.npy --walls_crs EPSG:3857]
```

Outputs, next to each exported `.ply`:

| file | content |
|---|---|
| `*_enu.ply` | OpenSfM topocentric ENU, metres around `reference_lla` |
| `*_epsg32632.ply` + `*_origin.json` | UTM 32N. PLY stores float32 (≈0.5 m steps at UTM magnitudes), so vertices are saved **minus a rounded origin**: add `origin` back. |
| `*_epsg32632.npz` | full-precision float64 coordinates |
| `*_epsg32632_aligned.*` | after rigid 2D alignment (trimmed ICP, at most `--align_max_angle` degrees) to the map walls. The rotation, shift and residual are in `*_aligned_origin.json`. |

Notes:
* `--scale` / `--offset` must be the ones used for training.
* `transforms.json` files produced by the old `sfm_to_colmap.py` (only `up` / `totp` / `avglen`) still work; pass `--georef_ref_lla` with the `reconstruction.json` or `reference_lla.json`.
* Use a metric CRS (`--georef_crs`, default `EPSG:32632`). EPSG:3857 inflates distances by ≈1.43 at Milan's latitude; walls given in 3857 are reprojected automatically.
* `--align_z_band 1 10` uses only points between 1 and 10 m above the reference to fit the alignment (facades rather than ground and trees).

## Options added to `main_nerf.py`

### Losses (all off by default)

| flag | what it does |
|---|---|
| `--MS_lambda λ` | minimum-support loss on the sampled densities, `Σ δ·σ²/(σ²+ε²)`: favours empty space |
| `--TV_lambda λ` | Huber total variation of the density along the sampled points (`--Huber_eps` = Huber delta) |
| `--dist_loss λ` | mip-NeRF 360 distortion loss on 10% of the rays (512 samples each; memory/time heavy) |
| `--l1_loss λ` | L1 on `σ·δ` |
| `--extreme_loss λ` | L1 pull of the density towards 0 |
| `--sobel_mse` | adds a photometric term weighted by the Sobel magnitude of the GT (see known issues) |
| `--Huber_eps ε` | ε of minimum-support and delta of the TV Huber loss (default `1e-2`) |
| `--max_epoch N` | train for N epochs instead of deriving them from `--iters` |

### Density activation

| flag | network |
|---|---|
| *(none)* | `nerf/network.py`, `trunc_exp`, view-independent colour |
| `--binary_sigma` | `nerf/network_binary.py`: `σ = 0.001` if `exp(h) < 1`, else `10 + exp(h)` |

### Geometry export and analysis

| flag | output (under the workspace) |
|---|---|
| `--first_point` | `meshes/*boundary_ep####.ply`: first sample along each training ray with `--mesh_thresh < σ < --mesh_thresh_up` (60k random rays per image), coloured |
| `--test` | test images/video + `meshes/mesh_TV_*.ply` (marching cubes, `--mesh_res`, `--mesh_thresh`) |
| `--mesh_principal` | with `--test`: re-extract the mesh in the bounding box of the largest connected component |
| `--mesh_partial` | with `--test`: mesh in a box aligned with the road direction estimated from the cameras |
| `--plane` | `meshes/*plane_ep####.{ply,npz}`: occupied points on the horizontal slice `z = 0` |
| `--get_histo` | `graph/`: density histograms on a grid |
| `--rays_plot` | `graph/`: density profiles along training rays (`--dens_clip` for the clipped plots) |
| `--dist_img` | `dist_imgs/`: per-view depth of the first occupied sample, and its colour |
| `--density_prog` | `density_prog.npy`: density on a `--TV_density_dim`³ grid after every epoch |
| `--random_video` | video interpolating between consecutive test poses |
| `--plot` | run the analysis flags above on the latest checkpoint, without training |

## Repository layout

```
main_nerf.py          entry point (training, export, analysis, georef)
activation.py         trunc_exp and the custom sigmoid activations
nerf/network*.py      NeRF networks (default, binary density, fully-fused, tcnn)
nerf/renderer.py      ray marching / volume rendering (+ density-grid queries)
nerf/provider.py      dataset loading (transforms.json), road-direction estimate
nerf/utils.py         Trainer, losses, mesh extraction and analysis tools
nerf/georef.py        NeRF -> ENU -> CRS, wall alignment, georeferenced export
tools/opensfm2nerf.py OpenSfM reconstruction.json -> transforms.json
```

## Known issues

* The view direction is removed by editing the networks, not with a flag: the original torch-ngp behaviour and checkpoints are not reproducible without code changes.
* `--binary_learn` and `--sigmoid_sigma` import modules that are not in the repository.
* `--ff` (fully-fused MLP) does not run: wrong import and input width after removing the view direction.
* `--sobel_mse` applies the Sobel filter to the batch of randomly sampled rays, not to the image, so the weights are not image gradients.
* `--mesh_partial`: the road frame built in `DP_road_versors` is not orthogonal unless the road is level.
* `--random_video`, and `--test` on COLMAP-style data, crash in the dataset loader (`self.images` is `None`).
* GUI test mode calls an undefined `test_gui` instead of `get_rays`.
* `bg_radius > 0` (background model) still uses the removed direction encoder.
* `DP_change_permission` `chown`s the workspace to uid/gid 2001 and is called unconditionally: it only works when running as root (e.g. in Docker).
* Loss flags have no `type=float`: they are parsed as strings (`float()` is applied where they are used).

## Acknowledgements

Built on [torch-ngp](https://github.com/ashawkey/torch-ngp) by Jiaxiang Tang (MIT licence, see `LICENSE`), which implements [instant-ngp](https://github.com/NVlabs/instant-ngp) (Müller et al., 2022). Structure from motion by [OpenSfM](https://github.com/mapillary/OpenSfM). The distortion loss follows [mip-NeRF 360](https://jonbarron.info/mipnerf360/) (Barron et al., 2022).
