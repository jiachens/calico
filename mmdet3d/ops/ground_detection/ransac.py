import open3d as o3d
import numpy as np
import math

def numpy_to_open3d(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])
    pcd.paint_uniform_color([0, 0, 1])
    return 

def get_inliers(pcd_data, plane_model, point_err_thres=0.2):
    [a, b, c, d] = plane_model
    x = np.sum(pcd_data[:,:3] * np.array([a, b, c]), axis=1) + d
    inliers = np.argwhere(np.absolute(x) < point_err_thres).reshape(-1)
    return inliers

def get_ground_plane_ransac(pcd_data):
    pcd = numpy_to_open3d(pcd_data)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                            ransac_n=3,
                                            num_iterations=1000)
    inliers = get_inliers(pcd_data, plane_model, point_err_thres=0.2)
    return plane_model, np.array(inliers).reshape(-1)


if __name__ == '__main__':
    pass