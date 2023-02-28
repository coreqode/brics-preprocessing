import os
import params as param_utils
import numpy as np
import pyngp as ngp
import cv2
import time
import json
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

parser = ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--network", type=str, required=True)
parser.add_argument("--action", type=str, default="")
parser.add_argument("--separate_calib", action="store_true")
parser.add_argument("--n_steps", type=int, default=2000)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--aabb_scale", type=int, default=1)
parser.add_argument("--use_3d_keypoints", action="store_true")

args = parser.parse_args()

if args.gui:
    import ipdb

if args.action:
    base_path = os.path.join(args.root_dir, "actions", args.action)
else:
    base_path = args.root_dir

image_dir = os.path.join(base_path, "image", "segmented_ngp")

if args.separate_calib:
    params_path = os.path.join(base_path, "calib", "optim_params.txt")
else:
    params_path = os.path.join(args.root_dir, "calib", "optim_params.txt")

params = param_utils.read_params(params_path)

extrs = []
intrs = []
dists = []
for param in params:
    extr = param_utils.get_extr(param)
    intr, dist = param_utils.get_intr(param)
    extr = np.vstack((extr, np.asarray([[0, 0, 0, 1]])))
    extr = np.linalg.inv(extr)

    extrs.append(extr)
    intrs.append(intr)
    dists.append(dist)

extrs = np.stack(extrs)

pos = extrs[:,:3,3]
if args.use_3d_keypoints:
    keypoints_dir = os.path.join(base_path, "keypoints_3d")
    keypoints_file = glob(f"{keypoints_dir}/*.json")[0]
    with open(keypoints_file, "r") as f:
        keypoints = np.asarray(json.load(f))
        valid_keypoints = ~np.isclose(keypoints[:,3], 0)
        keypoints = keypoints[valid_keypoints,:3]
    
    center = keypoints.mean(axis=0)
else:
    center = pos.mean(axis=0)

max_dist = cdist(pos, pos).max()

# Move center of scene to [0, 0, 0]
extrs[:,:3,3] -= center

# By observetion multiplying is better
# scale_factor = 3 / max_dist
# extrs[:,:3,3] *= scale_factor

# Move center of scene to [0.5, 0.5, 0.5]
extrs[:,:3,3] += 0.5

testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
testbed.reload_network_from_file(args.network)
testbed.create_empty_nerf_dataset(n_images=len(params), aabb_scale=args.aabb_scale)

for idx, param in enumerate(params):
    img_list = natsorted(glob(f"{image_dir}/{param['cam_name']}/*.[jp][pn]g"))
    img = cv2.cvtColor(cv2.imread(img_list[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    img = img.astype(np.float32)
    depth_img = np.zeros((img.shape[0], img.shape[1]))
    img /= 255

    extr = extrs[idx]
    intr = intrs[idx]
    dist = dists[idx]

    testbed.nerf.training.set_image(idx, img, depth_img)
    testbed.nerf.training.set_camera_extrinsics(idx, extr[:3], convert_to_ngp=False)
    testbed.nerf.training.set_camera_intrinsics(
        idx,
        fx=param["fx"], fy=param["fy"],
        cx=param["cx"], cy=param["cy"],
        k1=param["k1"], k2=param["k2"],
        p1=param["p1"], p2=param["p2"]
    )


# Taken from i-ngp:scripts/run.py
testbed.color_space = ngp.ColorSpace.SRGB
testbed.nerf.cone_angle_constant =0
testbed.nerf.visualize_cameras = True
testbed.nerf.training.random_bg_color = True

testbed.nerf.training.n_images_for_training = len(params)

testbed.shall_train = True

n_steps = args.n_steps
old_training_step = 0
tqdm_last_update = 0
if n_steps > 0:
    with tqdm(desc="Training", total=n_steps, unit="step") as t:
        while testbed.frame():
            # What will happen when training is done?
            if testbed.training_step >= n_steps:
                break

            # Update progress bar
            now = time.monotonic()
            if now - tqdm_last_update > 0.1:
                t.update(testbed.training_step - old_training_step)
                t.set_postfix(loss=testbed.loss)
                old_training_step = testbed.training_step
                tqdm_last_update = now

testbed.shall_train = False

if args.gui:
    testbed.init_window(1920, 1080)
    while testbed.frame():
        if testbed.want_repl():
            ipdb.set_trace()