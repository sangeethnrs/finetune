import os
import numpy as np
import open3d as o3d

def convert_bin_to_pcd(bin_path, pcd_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(pcd_path, pcd)

def process_kitti_dataset(kitti_dir, output_dir):
    velodyne_dir = os.path.join(kitti_dir, 'velodyne')
    output_pcd_dir = os.path.join(output_dir, 'pcd')
    os.makedirs(output_pcd_dir, exist_ok=True)

    for file in os.listdir(velodyne_dir):
        if file.endswith('.bin'):
            bin_path = os.path.join(velodyne_dir, file)
            pcd_path = os.path.join(output_pcd_dir, file.replace('.bin', '.pcd'))
            convert_bin_to_pcd(bin_path, pcd_path)

if __name__ == "__main__":
    kitti_dir = 'data/kitti'
    output_dir = 'data/kitti_converted'
    process_kitti_dataset(kitti_dir, output_dir)

