# AMBF runner for FoundationPose
# Data layout expected:
#   AMBF_data/
#     cam_K.txt
#     rgb/
#     depth/
#     masks/
#     mesh/

from estimater import *
from datareader import *
import argparse
import glob
import os
import cv2
import imageio
import numpy as np
import trimesh


class AMBFReader:
  def __init__(self, data_dir, mesh_file=None, depth_scale=1.0, depth_positive="auto"):
    self.data_dir = data_dir
    self.rgb_dir = os.path.join(data_dir, "rgb")
    self.depth_dir = os.path.join(data_dir, "depth")
    self.mask_dir = os.path.join(data_dir, "masks")
    self.K_file = os.path.join(data_dir, "cam_K.txt")

    self.depth_scale = depth_scale
    self.depth_positive = depth_positive

    if not os.path.exists(self.K_file):
      raise FileNotFoundError(f"Cannot find camera intrinsic file: {self.K_file}")

    self.K = np.loadtxt(self.K_file).reshape(3, 3).astype(np.float32)

    self.color_files = sorted(
      glob.glob(os.path.join(self.rgb_dir, "*.png")) +
      glob.glob(os.path.join(self.rgb_dir, "*.jpg")) +
      glob.glob(os.path.join(self.rgb_dir, "*.jpeg"))
    )

    if len(self.color_files) == 0:
      raise FileNotFoundError(f"No RGB images found in: {self.rgb_dir}")

    self.depth_files = sorted(glob.glob(os.path.join(self.depth_dir, "*.npy")))
    if len(self.depth_files) == 0:
      raise FileNotFoundError(f"No depth .npy files found in: {self.depth_dir}")

    self.mask_files = sorted(
      glob.glob(os.path.join(self.mask_dir, "*.png")) +
      glob.glob(os.path.join(self.mask_dir, "*.jpg")) +
      glob.glob(os.path.join(self.mask_dir, "*.jpeg"))
    )

    if len(self.mask_files) == 0:
      raise FileNotFoundError(f"No mask images found in: {self.mask_dir}")

    n = min(len(self.color_files), len(self.depth_files), len(self.mask_files))
    self.color_files = self.color_files[:n]
    self.depth_files = self.depth_files[:n]
    self.mask_files = self.mask_files[:n]

    self.id_strs = [
      os.path.splitext(os.path.basename(f))[0]
      for f in self.color_files
    ]

    print(f"[AMBFReader] Loaded {n} frames")
    print(f"[AMBFReader] K:\n{self.K}")
    print(f"[AMBFReader] RGB dir:   {self.rgb_dir}")
    print(f"[AMBFReader] Depth dir: {self.depth_dir}")
    print(f"[AMBFReader] Mask dir:  {self.mask_dir}")

  def __len__(self):
    return len(self.color_files)

  def get_color(self, i):
    img_bgr = cv2.imread(self.color_files[i], cv2.IMREAD_COLOR)
    if img_bgr is None:
      raise FileNotFoundError(f"Cannot read RGB image: {self.color_files[i]}")

    # FoundationPose demo uses RGB images.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

  def get_depth(self, i):
    depth = np.load(self.depth_files[i]).astype(np.float32)

    # If the AMBF point cloud z values are negative because the camera looks along -Z,
    # convert them to positive depth.
    if self.depth_positive == "negative_z":
      depth = -depth
    elif self.depth_positive == "abs":
      depth = np.abs(depth)
    elif self.depth_positive == "auto":
      valid = np.isfinite(depth)
      if valid.any():
        med = np.nanmedian(depth[valid])
        if med < 0:
          depth = -depth

    depth = depth * self.depth_scale

    # Remove invalid values.
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0

    return depth

  def get_mask(self, i):
    mask = cv2.imread(self.mask_files[i], cv2.IMREAD_GRAYSCALE)
    if mask is None:
      raise FileNotFoundError(f"Cannot read mask image: {self.mask_files[i]}")

    return mask > 0


def find_default_mesh(data_dir):
  mesh_dir = os.path.join(data_dir, "mesh")
  candidates = []
  for ext in ["*.obj", "*.ply", "*.stl"]:
    candidates.extend(glob.glob(os.path.join(mesh_dir, ext)))
  candidates = sorted(candidates)

  if len(candidates) == 0:
    raise FileNotFoundError(f"No mesh file found in: {mesh_dir}")

  return candidates[0]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))

  parser.add_argument(
    '--data_dir',
    type=str,
    default=f'{code_dir}/AMBF_data',
    help='AMBF data folder containing rgb, depth, masks, mesh, and cam_K.txt'
  )
  parser.add_argument(
    '--mesh_file',
    type=str,
    default=None,
    help='Path to object mesh. If not provided, first mesh file in AMBF_data/mesh is used.'
  )
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=2)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_AMBF')

  # If your saved depth is already positive metric depth, keep auto.
  # If AMBF z is negative, auto usually converts it correctly.
  parser.add_argument(
    '--depth_positive',
    type=str,
    default='auto',
    choices=['auto', 'negative_z', 'abs', 'none'],
    help='How to convert AMBF depth values to positive depth.'
  )
  parser.add_argument(
    '--depth_scale',
    type=float,
    default=1.0,
    help='Scale factor for depth. Use 1.0 if depth is already in meters.'
  )

  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  if args.mesh_file is None:
    args.mesh_file = find_default_mesh(args.data_dir)

  print(f"[run_AMBF] Using mesh: {args.mesh_file}")
  print(f"[run_AMBF] Using data_dir: {args.data_dir}")

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()

  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=debug_dir,
    debug=debug,
    glctx=glctx
  )

  logging.info("AMBF estimator initialization done")

  reader = AMBFReader(
    data_dir=args.data_dir,
    depth_scale=args.depth_scale,
    depth_positive=args.depth_positive
  )

  for i in range(len(reader)):
    logging.info(f'i:{i}')

    color = reader.get_color(i)
    depth = reader.get_depth(i)

    if i == 0:
      mask = reader.get_mask(0).astype(bool)

      pose = est.register(
        K=reader.K,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=args.est_refine_iter
      )

      if debug >= 3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')

        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth >= 0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

    else:
      pose = est.track_one(
        rgb=color,
        depth=depth,
        K=reader.K,
        iteration=args.track_refine_iter
      )

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

    if debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=0.1,
        K=reader.K,
        thickness=3,
        transparency=0,
        is_input_rgb=True
      )

    if debug >= 2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

  print(f"[run_AMBF] Done. Results saved to: {debug_dir}")
  print(f"[run_AMBF] Pose matrices: {debug_dir}/ob_in_cam")
  print(f"[run_AMBF] Visualization images: {debug_dir}/track_vis")
