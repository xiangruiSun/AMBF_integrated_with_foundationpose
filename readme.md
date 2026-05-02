# FoundationPose-Based Needle 6D Pose Estimation on AMBF Simulation Data

This README describes how to collect synchronized AMBF simulation data and run FoundationPose to generate 6D needle pose estimations from RGB-D images.

FoundationPose is a unified framework for 6D pose estimation and tracking of novel objects from RGB-D input. In this project, FoundationPose is used to estimate the 6D pose of a surgical needle in AMBF simulation using the recorded RGB image, depth image, camera intrinsic matrix, and needle mesh.

## 1. Pipeline Overview

The pipeline consists of two main stages:

1. **Data collection from AMBF simulation**
   - Slowly move the simulated camera.
   - Record RGB images, depth images, needle poses, and camera poses at the same time.
   - Save the synchronized data into the `FoundationPose/AMBF_data_100` folder.

2. **FoundationPose inference**
   - Use the recorded RGB-D data and camera intrinsics as input.
   - Use the surgical needle mesh as the object model.
   - Run FoundationPose to estimate the 6D pose of the needle.
   - Save the estimated object-in-camera poses into `FoundationPose_AMBF_results_100/ob_in_cam`.

## 2. AMBF Topics Used for Data Collection

During data collection, the camera is moved slowly using:

```bash
/ambf/env/cameras/cameraR/Command
```

At the same time, the following topics are recorded:

| Data | AMBF topic | Saved folder |
|---|---|---|
| RGB image | `/ambf/env/cameras/cameraR/ImageData` | `FoundationPose/AMBF_data_100/rgb` |
| Depth image | `/ambf/env/cameras/cameraR/DepthData` | `FoundationPose/AMBF_data_100/depth` |
| Needle pose | `/ambf/env/phantom/Needle/State` | `FoundationPose/AMBF_data_100/needle` |
| Camera pose | `/ambf/env/cameras/cameraR/State` | `FoundationPose/AMBF_data_100/cameraR` |

The camera intrinsic matrix is saved as:

```bash
FoundationPose/AMBF_data_100/cam_K.txt
```

All data streams should be synchronized by frame index. For example:

```bash
rgb/000012.png
depth/000012.png
needle/000012.txt
cameraR/000012.txt
```

should correspond to the same AMBF simulation time step.

## 3. Expected Input Data Structure

After collecting 100 synchronized frames, the input data folder should look like:

```bash
FoundationPose/
└── AMBF_data_100/
    ├── rgb/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── depth/
    │   ├── 000000.npy
    │   ├── 000000.png
    │   ├── 000001.npy
    │   ├── 000001.png
    │   └── ...
    ├── needle/
    │   ├── 000000.txt
    │   ├── 000001.txt
    │   └── ...
    ├── cameraR/
    │   ├── 000000.txt
    │   ├── 000001.txt
    │   └── ...
    └── cam_K.txt
```

The RGB images are stored in:

```bash
FoundationPose/AMBF_data_100/rgb
```

The depth data are stored in:

```bash
FoundationPose/AMBF_data_100/depth
```

The depth images are initially recorded as `.npy` arrays and are also converted into `.png` images for FoundationPose input.

The camera intrinsic matrix is stored in:

```bash
FoundationPose/AMBF_data_100/cam_K.txt
```

The needle pose and camera pose are recorded for synchronization and reproducibility.

## 4. FoundationPose Input

FoundationPose uses the following inputs:

| Input | Source |
|---|---|
| RGB image | `FoundationPose/AMBF_data_100/rgb` |
| Depth image | `FoundationPose/AMBF_data_100/depth` |
| Camera intrinsic matrix | `FoundationPose/AMBF_data_100/cam_K.txt` |
| Object mesh | Needle mesh from the AMBF scene |
| Object mask | Binary needle mask generated from AMBF simulation |

Because this pipeline uses AMBF simulation, the segmentation mask can be generated directly from the simulated object ID or mask output. No external detector such as RT-DETR is required for the simulated data.

## 5. Running FoundationPose

After the synchronized AMBF dataset is collected, run FoundationPose using the RGB-D data and the surgical needle mesh.

A typical command is:

```bash
cd /home/xsun97/FoundationPose

python run_AMBF_100.py \
  --data_dir /home/xsun97/FoundationPose/AMBF_data_100 \
  --debug_dir /home/xsun97/FoundationPose_AMBF_results_100 \
  --debug 2
```

where:

```bash
--data_dir
```

points to the synchronized AMBF input dataset, and:

```bash
--debug_dir
```

specifies the output folder for FoundationPose results.

The expected input folder is:

```bash
/home/xsun97/FoundationPose/AMBF_data_100
```

The expected output folder is:

```bash
/home/xsun97/FoundationPose_AMBF_results_100
```

## 6. FoundationPose Output

After inference, FoundationPose stores the estimated 6D needle poses in:

```bash
FoundationPose_AMBF_results_100/ob_in_cam
```

Each `.txt` file in this folder is a 4-by-4 homogeneous transformation matrix:

```text
r11 r12 r13 tx
r21 r22 r23 ty
r31 r32 r33 tz
0   0   0   1
```

This matrix represents the estimated needle pose in the camera coordinate frame.

For example:

```bash
FoundationPose_AMBF_results_100/
└── ob_in_cam/
    ├── 000000.txt
    ├── 000001.txt
    ├── 000002.txt
    └── ...
```

Each file corresponds to one RGB-D frame from the AMBF dataset.

## 7. Notes on Coordinate Frames

The 6D pose generated by FoundationPose is expressed in the camera frame, since the RGB-D observation is captured from `cameraR`.

Therefore, the output pose in:

```bash
FoundationPose_AMBF_results_100/ob_in_cam
```

should be interpreted as the object pose relative to the camera.

The camera pose and needle pose recorded from AMBF are stored separately in:

```bash
FoundationPose/AMBF_data_100/cameraR
FoundationPose/AMBF_data_100/needle
```

These files are useful for later analysis, visualization, and evaluation.

## 8. Troubleshooting

### No RGB or depth images found

Check that the data folder contains the expected image files:

```bash
find /home/xsun97/FoundationPose/AMBF_data_100/rgb -type f | head
find /home/xsun97/FoundationPose/AMBF_data_100/depth -type f | head
```

### No pose estimation output found

Check whether FoundationPose generated the `ob_in_cam` folder:

```bash
find /home/xsun97/FoundationPose_AMBF_results_100/ob_in_cam -type f | head
```

### Depth image format issue

If FoundationPose cannot read the depth images, confirm that the `.npy` depth arrays were converted into `.png` images:

```bash
ls /home/xsun97/FoundationPose/AMBF_data_100/depth | head
```

### Missing camera intrinsic matrix

FoundationPose requires the intrinsic matrix file:

```bash
/home/xsun97/FoundationPose/AMBF_data_100/cam_K.txt
```

Make sure this file exists before running inference.

## 9. Citation

If this pipeline is used in a report or paper, cite FoundationPose:

```bibtex
@inproceedings{wen2024foundationpose,
  title     = {FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects},
  author    = {Wen, Bowen and Yang, Wei and Kautz, Jan and Birchfield, Stan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {17868--17879},
  year      = {2024}
}
```

## 10. Reference

Wen, B., Yang, W., Kautz, J., and Birchfield, S. **FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
