import demo_colmap
import os
import cv2
import open3d as o3d
import numpy as np

def save_ply(points_3d,points_rgb,output_path = "test.ply"):
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points_rgb[:, :3] / 255.0)  # Normalize RGB values to [0, 1]
    
    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"PLY saved to {output_path}")

def save_ply_with_gt(points_3d, points_3d_gt, output_path="test_with_gt.ply"):
    # Load the ground truth point cloud
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_3d[:, :3])
    colors1 = np.tile( [1.0, 0.0, 0.0], (len(pcd1.points), 1))
    pcd1.colors = o3d.utility.Vector3dVector(colors1)


    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_3d_gt[:, :3])
    colors2 = np.tile( [0.0, 1.0, 0.0], (len(pcd2.points), 1))
    pcd2.colors = o3d.utility.Vector3dVector(colors2)


    # Save the combined point cloud to a PLY file
    o3d.io.write_point_cloud(output_path, pcd1+pcd2)
    print(f"PLY with GT saved to {output_path}")

if __name__ == "__main__":
    path_root = "demo_colmap/scan37"
    img_path_root = os.path.join(path_root, "images")
    sparse_colmap_path_root = os.path.join(path_root, "sparse","0")
    gt_extrinsic_path = os.path.join(sparse_colmap_path_root, "images.txt")
    gt_intrinsic_path = os.path.join(sparse_colmap_path_root, "cameras.txt")
    
    if os.path.exists(os.path.join(sparse_colmap_path_root, "points3D_colmap.ply")):
        gt_ply_path =  os.path.join(sparse_colmap_path_root, "points3D_colmap.ply")
        gt_ply = o3d.io.read_point_cloud(gt_ply_path)
    else:
        gt_ply_path =  os.path.join(sparse_colmap_path_root, "points3D.ply")
        gt_ply = o3d.io.read_point_cloud(gt_ply_path)
    
    img_path_list = sorted(os.listdir(img_path_root))
    img_path_list = [os.path.join(img_path_root, img_path) for img_path in img_path_list]
    
    # img_path_list = ["sparse_DTU/set_23_24_33/scan24/images/0000.png","sparse_DTU/set_23_24_33/scan24/images/0001.png","sparse_DTU/set_23_24_33/scan24/images/0002.png"]
    img_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img_list.append(img)
    points_3d,points_rgb,extrinsic,intrinsic = demo_colmap.run_demo(img_list,gt_extrinsic = gt_extrinsic_path,gt_intrinsic = gt_intrinsic_path)
    # points_3d,points_rgb,extrinsic,intrinsic = demo_colmap.run_demo(img_list)
    save_ply(points_3d,points_rgb)
    save_ply_with_gt(points_3d,np.array(gt_ply.points))

