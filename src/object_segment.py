import os
import numpy as np
import pyngp as ngp
import cv2
import time
import json
import ipdb
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
import sys
import utils.params as param_utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--roi", nargs=3, default=[0.5, 0.5, 0.5], type=float)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--separate_calib", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--aabb_scale", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--optimize_params", action="store_true")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--save_segmented_images", action="store_true")
    parser.add_argument("--cam_faces_path", type=str, default="./metadata/faces.json")
    
    parser.add_argument("--n_steps", nargs="+", default=[3000], type=int)
    ## Iterations are controlled by the downsampling factor list
    parser.add_argument("--downscale_factor", nargs="+", default=[0.65, 0.65, 0.65, 0.65], type=float)
    
    ## All about postprocessing of masks ##
    ## The list should be equal to the number of downscale factors
    parser.add_argument("--mask_thresh", nargs="+", default=[0.8, 0.6, 0.4, 0.2], type=float)
    parser.add_argument("--con_comp", nargs="+", default=[False, True, True, False], type=bool)
    parser.add_argument("--use_alpha", nargs="+", default=[False, False, True, True], type=bool)
    parser.add_argument("--dilate_mask", nargs="+", default=[False, False, False, True], type=bool)

    args = parser.parse_args()
    return args

def process_mask(alpha, opening_kernel, thresh, con_comp = False, use_alpha = False, dilate_mask = False):
    mask = alpha.copy()
    mask = mask > thresh
    mask = cv2.morphologyEx((mask*255).astype(np.uint8), cv2.MORPH_OPEN, opening_kernel)
    if con_comp: 
        mask = connected_components((mask* 255).astype(np.uint8))

    if dilate_mask:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.astype(np.uint8), opening_kernel, iterations=1)
        
    mask = mask.astype(np.float32) / 255
    
    if use_alpha:
        mask = mask * alpha
    return mask

def connected_components(th):
    num_labels, labels_im = cv2.connectedComponents(th)
    s = 0
    ind = 1
    if num_labels > 2:
        for nl in range(1, num_labels):
            temp = np.sum((labels_im == nl).astype(np.uint8))
            if temp > s:
                s = temp
                ind = nl
    mask = (labels_im == ind) * 255
    return mask

def segment_frame(testbed, roi, params, c2ws, frame_num, base_path, args, output_path):
    testbed.reload_network_from_file(args.network)
    testbed.create_empty_nerf_dataset(n_images=len(params), aabb_scale=1)

    # Upload images to I-NGP
    imgs = []
    image_dir = os.path.join(base_path, "image")
    for idx, param in enumerate(params):
        img_list = natsorted(glob(f"{image_dir}/{param['cam_name']}/*.[jp][pn]g"))
        img = cv2.cvtColor(cv2.imread(img_list[frame_num], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        img = img.astype(np.float32)
        depth_img = np.zeros((img.shape[0], img.shape[1]))
        img /= 255
        height, width = img.shape[:2]

        c2w = c2ws[idx]

        testbed.nerf.training.set_image(idx, img, depth_img)
        testbed.nerf.training.set_camera_extrinsics(idx, c2w[:3], convert_to_ngp=False)
        testbed.nerf.training.set_camera_intrinsics(
            idx,
            fx=param["fx"], fy=param["fy"],
            cx=param["cx"], cy=param["cy"],
            k1=param["k1"], k2=param["k2"],
            p1=param["p1"], p2=param["p2"]
        )
        imgs.append(img)
        idx +=1
        
    # if args.gui:
    #     testbed.init_window(1920, 1080)
    #     while testbed.frame():
    #         if testbed.want_repl():
    #             ipdb.set_trace()
        
    # Taken from i-ngp:scripts/run.py
    testbed.color_space = ngp.ColorSpace.SRGB
    testbed.nerf.cone_angle_constant = 0
    testbed.nerf.visualize_cameras = True
    testbed.nerf.training.random_bg_color = True
    # Set alpha to 0 to get accumulation value
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]

    testbed.nerf.training.n_images_for_training = len(params)

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Segment
    total_n_steps = args.n_steps[0]
    downscale_factor = args.downscale_factor
    for i in range(len(downscale_factor)):
        if args.optimize_params:
            print(f"Step: {i} | Optimizing Camera Parameters !!")
            print("--------------------------------------------")
            testbed.nerf.training.optimize_extrinsics = True
            testbed.nerf.training.optimize_focal_length = True
            testbed.nerf.training.optimize_distortion = True

        size = 1
        testbed.shall_train = True
        old_training_step = 0
        tqdm_last_update = 0
            
        if total_n_steps > 0:
            with tqdm(desc="Training", total=total_n_steps, unit="step") as t:
                while testbed.frame():
                    # What will happen when training is done?
                    if testbed.training_step >= total_n_steps:
                        break

                    # Update progress bar
                    now = time.monotonic()
                    if now - tqdm_last_update > 0.1:
                        t.update(testbed.training_step - old_training_step)
                        t.set_postfix(loss=testbed.loss)
                        old_training_step = testbed.training_step
                        tqdm_last_update = now
                        
        if len(args.n_steps) > 1:
            if i != len(args.n_steps) - 1:
                n_steps = args.n_steps[i+1]
            total_n_steps += n_steps
            
        
            
        testbed.shall_train = False

        size -= downscale_factor[i]
        mx = np.minimum(0.9*np.ones(3), ((np.ones(3) * roi) + size / 2))
        mn = np.maximum(0.1*np.ones(3), ((np.ones(3) * roi) - size / 2))
        testbed.render_aabb.min = mn
        testbed.render_aabb.max = mx

        masks = []
        for idx in tqdm(range(len(params))):
            testbed.set_camera_to_training_view(idx)
            frame = testbed.render(width, height, 1, True)
            alpha = frame[:,:,3]
                
            mask = process_mask(alpha, opening_kernel, args.mask_thresh[i], args.con_comp[i], args.use_alpha[i], args.dilate_mask[i])

            masks.append(mask)
            img = imgs[idx] * mask[...,None] # Multiply mask to get sharper image to retrain
            depth_img = np.zeros_like(img)[:,:,0]
            testbed.nerf.training.set_image(idx, img, depth_img)
            
            if i == len(downscale_factor) - 1: 
                if args.save_segmented_images:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                    cv2.imwrite(os.path.join(output_path, params[idx]['cam_name'], f"{frame_num:08d}.png"), 255*img)

        if args.gui:
            testbed.init_window(1920, 1080)
            while testbed.frame():
                if testbed.want_repl():
                    ipdb.set_trace()
        



def main():
    args = get_parser()

    with open(args.cam_faces_path, "r") as f:
        faces = json.load(f)

    base_path = os.path.join(args.root_dir, "objects", args.name)

    params_name = "optim_params.txt"
    if args.optimize_params:
        params_name = "params.txt"

    if args.separate_calib:
        params_path = os.path.join(base_path, "calib", params_name)
        to_skip_path = os.path.join(base_path, "calib", "to_skip.txt")
    else:
        params_path = os.path.join(args.root_dir, "calib", params_name)
        to_skip_path = os.path.join(args.root_dir, "calib", "to_skip.txt")

    params = param_utils.read_params(params_path)
    fields = params.dtype.fields
    to_skip = param_utils.read_to_skip(to_skip_path)
    
    if args.optimize_params:
        if len(to_skip) > 0:
            new_params = np.zeros(len(params) - len(to_skip), dtype=params.dtype) 
            count = 0
            for i, param in enumerate(params):
                if param['cam_name'] not in to_skip:
                    new_params[count] = param
                    count+=1
            params = new_params
    
    # Create output directory
    output_path = os.path.join(base_path, "segmented_ngp")
    os.makedirs(output_path, exist_ok=True)

    # Canocalize the capture system
    cam2idx = {}
    pos = []
    rot = []
    
    for idx, param in enumerate(params):
        w2c = param_utils.get_extr(param)
        w2c = np.vstack((w2c, np.asarray([[0, 0, 0, 1]])))
        c2w = np.linalg.inv(w2c)
        cam2idx[param["cam_name"]] = idx

        pos.append(c2w[:3,3])
        rot.append(c2w[:3,:3])

        os.makedirs(os.path.join(output_path, param["cam_name"]), exist_ok=True)

    pos = np.stack(pos)
    rot = np.stack(rot)

    roi = args.roi

    center = pos.mean(axis=0)
    max_dist = cdist(pos, pos).max()

    # Move center of scene to [0, 0, 0]
    pos -= center

    axs = np.zeros((3, 3))

    # Rotate to align bounding box
    for idx, dir_ in enumerate([
        ["1 0 0", "-1 0 0"],
        ["0 1 0", "0 -1 0"],
        ["0 0 1", "0 0 -1"],
    ]):
        avg1 = []
        for camera in faces[dir_[0]]["cameras"]:
            if not camera in to_skip:
                avg1.append(pos[cam2idx[camera]])
        avg2 = []
        for camera in faces[dir_[1]]["cameras"]:
            if not camera in to_skip:
                avg2.append(pos[cam2idx[camera]])
        axs[idx] = np.asarray(avg1).mean(axis=0) - np.asarray(avg2).mean(axis=0)
        axs[idx] /= np.linalg.norm(axs[idx])

    # Get closest orthormal basis
    u, _, v = np.linalg.svd(axs)
    orth_axs = u @ v

    new_pos = (orth_axs @ pos.T).T
    new_rot = orth_axs @ rot

    # Scale to fit diagonal in unity cube
    scale_factor = np.sqrt(2) / max_dist * 0.9
    new_pos *= scale_factor

    # Move center of scene to [0.5, 0.5, 0.5]
    new_pos += 0.5

    c2ws = np.zeros((new_pos.shape[0], 4, 4))
    c2ws[:,:3,:3] = new_rot
    c2ws[:,:3,3] = new_pos
    c2ws[:,3,3] = 1

    
    if args.static:
        frame_nums = [ 0 ]
    else:
        frame_nums = range(args.start, args.end, args.step)

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    for frame_num in tqdm(frame_nums):
        segment_frame(testbed, roi, params, c2ws, frame_num, base_path, args, output_path)

    # if args.optimize_params:
    optim_params = params.copy()
    for idx, param in enumerate(params):
            c2w = testbed.nerf.training.get_camera_extrinsics(idx, convert_to_ngp = False)
            c2w = np.vstack((c2w, np.asarray([[0, 0, 0, 1]])))
            w2c = np.linalg.inv(c2w)
            qvec = Rotation.from_matrix(w2c[:3,:3]).as_quat()
            tvec = w2c[:3,3]
            optim_params[idx]["qvecx"] = qvec[0]
            optim_params[idx]["qvecy"] = qvec[1]
            optim_params[idx]["qvecz"] = qvec[2]
            optim_params[idx]["qvecw"] = qvec[3]
            optim_params[idx]["tvecx"] = tvec[0]
            optim_params[idx]["tvecy"] = tvec[1]
            optim_params[idx]["tvecz"] = tvec[2]
            
    np.savetxt(os.path.join(os.path.dirname(params_path), 'optim_params.txt'), optim_params, fmt="%s", header=" ".join(fields))

if __name__ == '__main__':
    main()