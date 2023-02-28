# BRICS Preprocessing
Repository maintained for BRICS data preprocessing and camera calibration.

---
## Dependencies
---
### Instant-NGP
- Clone the modified version (patch is already applied) of Instant-NGP from https://github.com/coreqode/instant-ngp
- Build the Instant-NGP version by following the official instructions from the original repository. 

### Python Packages 
- Install python packages using `pip install -r requirements.txt`
- export PYTHONPATH=":"


---
## Segmentation
---
- Please check the `assets` folder for the data directory structure. 
- Right now assumes `params.txt` (unoptimized camera parameters) in the `calib` folder. 
  
### Optimizing the Camera Parameters:
- It is not possible right now to optimize extrinsics as well as do segmentation. The network doesn't converge very well. 
- So we need to have alpha mask of one object (53 images) to refine the extrinsics first. 
  - To achieve this, you can run `sh scripts/optimize_extrinsics.sh`
  - This will train Instant-NGP on the segmented images, optimize extrinsics and dump the `optim_params.txt` in the `calib` folder.

### Segmentation
- Once we get the refined params `optim_params.txt`, we can use the same parameters for all the object for that capture session (assuming that we didn't disturb the cameras). 
- To do so, you can run `sh scripts/segment.sh`.
---
## HyperParameters
---

### Generic Parameters
- `root_dir`: Path of the root directory, e.g. `./assets/`
- `network`: Path of the Instant-NGP network config file
- `name`: Name of the object sequence
- `separate_calib`: Flag whether we need to provide separate calib
- `gui`: Do we need gui or not
- `aabb_scale`: aabb scale for Instant-NGP
- `start`: Start Frame
- `end`:  End Frame

### Tunable Parameters
- `roi`: Region of interest for the bounding box. Can vary if object moves away from the center. 
- `step`: Step size for choosing the frames in between the start and end.
- `optimize_params`: Flag whethere to optimize parameters 
- `static`: Flag for static scene i.e. only one timestamp
- `save_segmented_images`: Flag for exporting the segmented images
- `cam_faces_path`: Sequence name for cameras
- `n_steps`: List of number of training steps in each iteration.
- `downscale_factor`: List of downscaling factor for the bounding box in each iteration.
- `mask_thresh`: List of mask threshold in each iteration
- `con_comp`:  Use connected-componnets or not in each iteration
- `use_alpha`: Use alpha or not in each iteration
- `dilate_mask`: Dilate mask or not in each iteration


##### NOTE:
- What if the segmentation has cuts and looks eroded?
  - Try dilation and decreasing the mask_thresh

- What if the segmentation mask is larger thatn the actual image? 
  - Try to remove dilation and increase the mask_thresh

- For the objects which are not in the center, change the ROI so that the object does not go outside the bounding box. 

