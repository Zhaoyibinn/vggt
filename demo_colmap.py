# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import roma
import open3d as o3d

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=True, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=5, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=2048, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

def vis_o3d_pcd_1(cloud,color = [1,1,1]):
    
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd])

def vis_o3d_pcd_2(cloud1,cloud2,color1 = [1,1,1],color2 = [1,1,1]):
    
    pcd1=o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud1)
    pcd1.paint_uniform_color(color1)
    pcd2=o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2)
    pcd2.paint_uniform_color(color2)

    o3d.visualization.draw_geometries([pcd1,pcd2])
    

def visualize_camera_poses(poses_tensor1, poses_tensor2, size=0.1, show_coordinate_frame=True):
    
    # 创建Open3D可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 转换为NumPy数组
    poses1 = poses_tensor1.numpy()
    poses2 = poses_tensor2.numpy()
    
    # 添加第一个相机的位姿
    for i, pose in enumerate(poses1):
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        camera.transform(pose)
        camera.paint_uniform_color([1, 0, 0])  # 红色表示第一个相机
        vis.add_geometry(camera)
    
    # 添加第二个相机的位姿
    for i, pose in enumerate(poses2):
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        camera.transform(pose)
        camera.paint_uniform_color([0, 0, 1])  # 蓝色表示第二个相机
        vis.add_geometry(camera)
    
    # 可选：添加全局坐标系
    # if show_coordinate_frame:
    #     world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    #     vis.add_geometry(world_frame)
    
    vis.run()
    vis.destroy_window()

def visualize_camera_poses1(poses_tensor1, size=0.1, show_coordinate_frame=True):
    
    # 创建Open3D可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 转换为NumPy数组
    poses1 = poses_tensor1.numpy()
    
    
    # 添加第一个相机的位姿
    for i, pose in enumerate(poses1):
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        camera.transform(pose)
        camera.paint_uniform_color([1, i/3, 0])  # 红色表示第一个相机
        vis.add_geometry(camera)
    

    
    # 可选：添加全局坐标系
    # if show_coordinate_frame:
    #     world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    #     vis.add_geometry(world_frame)
    
    vis.run()
    vis.destroy_window()

def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    return np.median(pdist([p[:3, 3].numpy() for p in poses]))

def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 10
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps*poses[:, :3, 2]))
    R, T, s = roma.rigid_points_registration(center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True)
    # import open3d as o3d

    return s, R, T

def signed_log1p(x):
    sign = torch.sign(x)
    return sign * torch.log1p(torch.abs(x))

def rotate_points_with_srt(source_points, s, R, t):

    # 1. 处理缩放（先应用缩放）
    if isinstance(s, (float, int)):
        # 标量缩放：所有维度等比例缩放
        scaled_points = source_points * s
    else:
        # 各维度独立缩放（假设s是3维向量）
        scaled_points = source_points * s.reshape(1, -1)  # (N,3) * (1,3)
    
    # 2. 应用旋转变换（矩阵乘法）
    rotated_points = torch.matmul(scaled_points, R.T)  # (N,3) @ (3,3) = (N,3)
    
    # 3. 应用平移变换
    rotated_points = rotated_points + t.reshape(1, -1)  # (N,3) + (1,3)
    
    return rotated_points

def rotate_cameras_with_srt(poses, s,R,t):


    
    # 创建4×4的SRT变换矩阵
    srt_matrix = torch.eye(4, device=poses.device)
    srt_matrix[:3, :3] = s * R  # 缩放和旋转
    srt_matrix[:3, 3] = t       # 平移
    
    # 对每个相机位姿应用SRT变换
    # 注意：相机位姿通常是从世界坐标系到相机坐标系的变换
    # 因此，我们需要右乘SRT变换矩阵
    transformed_poses = srt_matrix @ poses
    
    return transformed_poses


def read_colmap_camera(colmap_camera_path):
    with open(colmap_camera_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        line = lines[3:][0]
        image_W,image_H,focal_x,focal_y,cx,cy = [float(i) for i in line.split()[2:]]

    return image_W,image_H,focal_x,focal_y,cx,cy

def read_colmap_gt(colmap_images_path):
    # colmap_images_path = "sparse_DTU/set_23_24_33/scan40/sparse/0/images.txt"
    # colmap_images_path = "sparse_DTU/wo_pose/scan24/sparse/0/images.txt"
    with open(colmap_images_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = lines[4:]
        poses = torch.zeros(int(len(lines)/2),7)
        for idx,line in enumerate(lines):
            if idx % 2 == 0:
                line_splited = line.split()
                image_idx = int(line_splited[-1][:4])
                pose = torch.tensor([float(line_splited[2]),float(line_splited[3]),float(line_splited[4]),float(line_splited[1]),float(line_splited[5]),float(line_splited[6]),float(line_splited[7])])
                poses[image_idx] = pose

    poses_R = []
    for pose in poses:
        q_x, q_y, q_z,q_w,t_x,t_y,t_z = pose

        R = torch.eye(4)
        R_3 = torch.tensor([
            [1 - 2 * q_y ** 2 - 2 * q_z ** 2, 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)],
            [2 * (q_x * q_y + q_w * q_z), 1 - 2 * q_x ** 2 - 2 * q_z ** 2, 2 * (q_y * q_z - q_w * q_x)],
            [2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * q_x ** 2 - 2 * q_y ** 2]
            ])
        t = torch.tensor([t_x,t_y,t_z])

        R[:3, :3] = R_3
        R[:3, 3] = t

        poses_R.append(R.inverse())
    return torch.stack(poses_R,dim = 0)[:,:3,:]

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    pretrained_dict = torch.load('weight/model.pt')
    # model_dict = model.state_dict()
    model.load_state_dict(pretrained_dict)

    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    
    gt_pose = read_colmap_gt("sparse_DTU/set_23_24_33/scan24/sparse/0/images.txt")
    last_row = torch.tensor([0, 0, 0, 1]).expand(gt_pose.shape[0], 1, 4)
    gt_pose44 = torch.cat([torch.tensor(gt_pose), last_row], dim=1)

    image_W,image_H,focal_x,focal_y,cx,cy = read_colmap_camera("sparse_DTU/set_23_24_33/scan24/sparse/0/cameras.txt")
    intrinsic_matrix = np.array([
        [focal_x, 0, cx],
        [0, focal_y, cy],
        [0, 0, 1]
    ])
    intrinsic_gt = np.tile(intrinsic_matrix, (len(image_path_list), 1, 1))
    image_size_gt = np.array([int(image_W),int(image_H)])
    # gt_pose_wo = read_colmap_gt("sparse_DTU/wo_pose/scan24/sparse/0/images.txt")
    # gt_pose_wo44 = torch.cat([torch.tensor(gt_pose_wo), last_row], dim=1)

    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    
    extrinsic44 = torch.cat([torch.tensor(extrinsic), last_row], dim=1)
    extrinsic44 = torch.inverse(extrinsic44)

    # flip_x_transform = torch.tensor([
    # [1.0, 0, 0, 0],
    # [0, -1, 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1]
    # ])
    # extrinsic44_trans = []
    # for extrinsic44_1 in extrinsic44:
    #     extrinsic44_1_trans = flip_x_transform@ extrinsic44_1
    #     extrinsic44_trans.append(extrinsic44_1_trans)
    # extrinsic44 = torch.stack(extrinsic44_trans)
    # gt_pose_wo44[:,:3,3] = gt_pose_wo44[:,:3,3] * 10
    s, R, T = align_multiple_poses(extrinsic44,gt_pose44)
    
    # extrinsic44_trans_inverse = rotate_cameras_with_srt(extrinsic44,s,R,T)
    

    # visualize_camera_poses(rotate_cameras_with_srt(extrinsic44,s,R,T),gt_pose44)

    
    
    
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    # gt_cloud_path = "/home/zhaoyibin/3DRE/3DGS/FatesGS/DTU/set_23_24_33/scan24/sparse/0/points3D.ply"
    # pcd = o3d.io.read_point_cloud(gt_cloud_path)
    # points_trans = rotate_points_with_srt(torch.tensor(points_3d[0].reshape(-1, 3)).float(),s,R,T).numpy()

    # vis_o3d_pcd_2(np.array(pcd.points),points_trans,color1=[1,0,0],color2=[0,1,0])

    
    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]

        points_3d_trans2gt = rotate_points_with_srt(torch.tensor(points_3d).float(),s,R,T).numpy()

        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_gt = batch_np_matrix_to_pycolmap_wo_track(
            points_3d_trans2gt,
            points_xyf,
            points_rgb,
            torch.inverse(gt_pose44)[:, :3, :],
            intrinsic_gt,
            image_size_gt,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    reconstruction_gt = rename_colmap_recons_and_rescale_camera(
        reconstruction_gt,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=max(original_coords[0,-2:].cpu().numpy()),
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera
    )

    print(f"Saving reconstruction to {args.scene_dir}/sparse/0")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/0/points.ply"))


    print(f"Saving reconstruction to {args.scene_dir}/sparse/gt")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse/gt")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction_gt.write(sparse_reconstruction_dir)


    trimesh.PointCloud(points_3d_trans2gt, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/gt/points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False,rescale_camera = True
):
    

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
