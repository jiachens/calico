import open3d as o3d
import numpy as np
import math
from time import time
from calico_tools.visualize.general import draw_matplotlib

def get_ground_plane_grf(pcd_data, max_range=54, segment_width=10, sensor_height=1.841, iter_num=5, lpr_num=50, seed_thres=0.5, dist_thres=0.2):
    '''
    method introduced in Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for Autonomous Vehicle Applications
    '''
    plane_model = {}
    inliers = []
    
    # Splits segments along x-axis.
    segments = {}
    for segment_start in range(-max_range, max_range, segment_width):
        segments[(segment_start, segment_start + segment_width)] = np.argwhere(np.logical_and(
            pcd_data[:,0] >= segment_start,
            pcd_data[:,0] < segment_start + segment_width)).reshape(-1)

    for segment_range, segment_point_indices in segments.items():
        if segment_point_indices.shape[0] < lpr_num:
            continue
        segment_points = pcd_data[segment_point_indices,:3]

        # Extracts initial seeds.
        points = np.copy(segment_points)
        points = points[points[:,2] >= 0 - sensor_height - 0.5]
        lowest_height = np.mean(points[points[:,2].argsort(),2][:lpr_num])
        seeds = points[points[:,2] < lowest_height + seed_thres,:]
        
        # Main loop.
        model = None
        for _ in range(iter_num):
            # Estimates the plane using the seed points.
            open3d_seeds = o3d.geometry.PointCloud()
            open3d_seeds.points = o3d.utility.Vector3dVector(seeds)
            mean, covariance = open3d_seeds.compute_mean_and_covariance()
            U, Sigma, V = o3d.core.svd(covariance)
            normal = U.numpy()[:,2]
            seeds_mean = np.mean(seeds, axis=0)
            d = -(np.dot(normal, seeds_mean.T))
            model = np.array([*normal, d])

            seeds = seeds[np.dot(normal, seeds.T) < dist_thres - d]

        ground_point_indices = np.argwhere(np.dot(model[:3], segment_points.T) < dist_thres - model[3]).reshape(-1)
        inliers += segment_point_indices[ground_point_indices].tolist()
        plane_model[segment_range] = model

    return plane_model, inliers

if __name__ == '__main__':
    from nuscenes.utils.data_classes import LidarPointCloud
    pointcloud = LidarPointCloud.from_file("./data/nuscenes/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800543047452.pcd.bin")
    time_s = time()
    _, inlier = get_ground_plane_grf(pointcloud.points.T)
    outlier = np.setdiff1d(np.arange(pointcloud.points.shape[1]), inlier)
    time_e = time()
    print(f"Time: {time_e - time_s}")
    # print(pointcloud.points.T[inlier,:])
    draw_matplotlib(pointcloud.points.T[outlier,:], save="./data/test.png")