import numpy as np
import open3d as o3d
from time import time
import cv2
from shapely.geometry import MultiPoint


def points_to_bbox(points): 
    bbox = MultiPoint(points[:,:2]).bounds
    return bbox

def generate_bbox(pointcloud_segments):
    bboxes = np.array([points_to_bbox(segment) for segment in pointcloud_segments])
    return bboxes

def filter_segmentation(pointcloud_segments, in_lane_mask=None, point_height=None, lidar_height=1.84):
    object_segments = []
    for pointcloud_segment in pointcloud_segments:
        polygon = points_to_polygon(pointcloud_segment[:,:2])
        cnt = pointcloud_segment[:,:2].reshape((-1,1,2)).astype(np.float32)
        _,ret,_ = cv2.minAreaRect(cnt)
        if max(np.abs(pointcloud_segment[:,1])) < 1.5 or max(np.abs(pointcloud_segment[:,0])) < 1.5: ## filter out ego vehicle
            continue
        if max(pointcloud_segment[:,2]) < -lidar_height + 0.25 or max(pointcloud_segment[:,2]) > 3.: ## too close to the ground
            continue
        if max(ret) == 0 or min(ret)/max(ret) < 0.1: ## too thin to have rich context
            continue
        if polygon.area > 20: ## too large area usually means the segment is not a object
            continue
        if polygon.area < 0.25: ## too small area usually means the segment is not a object
            continue
        if in_lane_mask is not None:
            #TODO on-road objects filtering
            pass
        object_segments.append(pointcloud_segment)
    return object_segments

def numpy_to_open3d(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])
    pcd.paint_uniform_color([0, 0, 1])
    return pcd


'''
open3d version of dbscan
'''
def lidar_segmentation_dbscan(full_pcd, ground_indices, cluster_thres=0.75, min_point_num=5):
    non_ground_mask = np.ones(full_pcd.shape[0]).astype(bool)
    non_ground_mask[ground_indices] = False
    non_ground_indices = np.argwhere(non_ground_mask > 0).reshape(-1)
    pcd = full_pcd[non_ground_mask]

    open3d_pcd = numpy_to_open3d(pcd)
    labels = np.array(open3d_pcd.cluster_dbscan(eps=cluster_thres, min_points=min_point_num, print_progress=False))
    info = []
    for label in np.unique(labels):
        if label == -1:
            continue
        indices = np.argwhere(labels == label).reshape(-1)
        info.append({
            "indices": non_ground_indices[indices]
        })

    return {"info": info}


'''
sklearn version of dbscan
'''
def lidar_segmentation_dbscan_sklearn(full_pcd, ground_indices, cluster_thres=0.75, min_point_num=5):
    from sklearn.cluster import DBSCAN
    non_ground_mask = np.ones(full_pcd.shape[0]).astype(bool)
    non_ground_mask[ground_indices] = False
    non_ground_indices = np.argwhere(non_ground_mask > 0).reshape(-1)
    pcd = full_pcd[non_ground_mask,:2]
    # filter_indices = np.argwhere((pcd[:,2] > -5.) & (pcd[:,2] < 3.)).reshape(-1)
    # pcd = pcd[filter_indices,:2]
    clustering = DBSCAN(eps=cluster_thres, min_samples=min_point_num).fit(pcd)
    labels = clustering.labels_
    info = []
    for label in np.unique(labels):
        if label == -1:
            continue
        indices = np.argwhere(labels == label).reshape(-1)
        info.append({
            "indices": non_ground_indices[indices]
        })

    return {"info": info}


##TODO: remove the following code in the end
'''
preprocessing for slr code
'''
def sort_points(pcd):
    rings = pcd[:, 4].astype(np.int32)
    distance = np.sqrt(np.sum(pcd[:,:2] ** 2, axis=1))
    # Vertical angle
    v_angle = np.arctan2(distance, -pcd[:,2])
    # Horizontal angle
    h_angle = np.arctan2(pcd[:, 0], pcd[:, 1])

    points_stack = []
    for ring_id in np.sort(np.unique(rings)):
        ring_indices = np.argwhere(rings == ring_id).reshape(-1)
        ring_h_angle = h_angle[ring_indices]
        ring_shuffle_indices = np.argsort(ring_h_angle)
        points_stack.append(pcd[ring_indices][ring_shuffle_indices])
    return np.vstack(points_stack)

'''
slr code, not used in this project due to poor performance
'''
def lidar_segmentation_slr(full_pcd, ground_indices, cluster_thres=0.5, min_num_points=10):
    non_ground_mask = np.ones(full_pcd.shape[0]).astype(bool)
    non_ground_mask[ground_indices] = False
    non_ground_indices = np.argwhere(non_ground_mask > 0).reshape(-1)
    pcd = full_pcd[non_ground_mask]
    distance = np.sqrt(np.sum(pcd[:,:2] ** 2, axis=1))
    angle = np.arctan2(distance, -pcd[:,2])
    angle_delta = angle - np.concatenate((np.array([angle[0]]), angle[:-1]), axis=None)
    rings = pcd[:, 4].astype(np.int32)
    prev_pcd = np.vstack((np.array([pcd[0,:]]), pcd[:-1,:]))
    dist_delta = np.sqrt(np.sum((pcd - prev_pcd) ** 2, axis=1))
    breaks = dist_delta > cluster_thres
    instance_label = np.zeros(pcd.shape[0]).astype(np.int32)
    instance_count = 0
    ring_start = None
    for i in range(pcd.shape[0]):
        if i == 0:
            # The first point.
            instance_count += 1
            ring_start = i
            instance_label[i] = instance_count
            continue
            # continue​# Finds cluser in the same ring.
        if rings[i] != rings[i-1]:
            ring_start = i
        else:
            # The last point is the same ring.
            if breaks[i] == 0:
                instance_label[i] = instance_label[i-1]
            if i < pcd.shape[0] - 1 and rings[i] != rings[i+1]:
                # It is the last point in this ring.
                if np.sqrt(np.sum((pcd[i] - pcd[ring_start]) ** 2)) <= cluster_thres:
                    if instance_label[i] == 0:
                        instance_label[i] = instance_label[ring_start]
                    else:
                        instance_label[instance_label == instance_label[i]] = instance_label[ring_start]
        if instance_label[i] > 0:
            continue

        # Finds cluster in the last ring.
        last_ring_indices = np.argwhere((rings >= rings[i] - 1) * (rings < rings[i])).reshape(-1)
        if len(last_ring_indices) == 0:
            instance_count += 1
            instance_label[i] = instance_count
            continue
        last_ring_distance = np.sqrt(np.sum((pcd[i] - pcd[last_ring_indices]) ** 2, axis=1))
        if np.min(last_ring_distance) <= cluster_thres:
            instance_label[i] = instance_label[last_ring_indices[np.argmin(last_ring_distance)]]
        if instance_label[i] > 0:
            continue

        instance_count += 1
        instance_label[i] = instance_count
        # TODO: class of instances.
    info = []

    for label in range(1, instance_count + 1):
        indices = np.argwhere(instance_label == label).reshape(-1)
        if len(indices) < min_num_points:
            continue
        info.append({"indices": non_ground_indices[indices]})

    return {"info": info}


if __name__ == '__main__':
    from mmdet3d.ops.ground_detection.grf import get_ground_plane_grf
    from nuscenes.utils.data_classes import LidarPointCloud
    from calico_tools.visualize.general import draw_multi_cluster_matplotlib, draw_matplotlib, draw_mutlti_cluster_polygon_matplotlib
    from calico_tools.visualize.utils import *
    from nuscenes.nuscenes import NuScenes
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes/', help='path to nuscenes dataset')
    args = parser.parse_args()
       
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=True)
    bbox_path = os.path.join(args.dataroot,'samples/POOLED_BBOX')
    os.makedirs(bbox_path,exist_ok=True)

    for index in range(0, len(nusc.sample)):
        my_sample = nusc.sample[index]
        # nusc.render_sample(my_sample['token'],out_path='./data/temp_test/render_'+str(index)+'.png',verbose=False)
        lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
        pcd_path = os.path.join(nusc.dataroot, lidar_data['filename'])
        pointcloud = np.fromfile(pcd_path, dtype=np.float32).reshape((-1, 5))
        _, inlier = get_ground_plane_grf(pointcloud)
        ret = lidar_segmentation_dbscan_sklearn(pointcloud, inlier)
        pointcloud_segments = []
        for indice in ret['info']:
            pointcloud_segments.append(pointcloud[indice['indices']])
        pointcloud_segments = filter_segmentation(pointcloud_segments)
        bboxes = generate_bbox(pointcloud_segments)
        bbox_path = str(pcd_path).replace('LIDAR_TOP','POOLED_BBOX')
        bbox_path = bbox_path.replace('pcd.bin','npy')
        # np.save(bbox_path,bboxes)
        draw_mutlti_cluster_polygon_matplotlib(pointcloud_segments,bboxes=bboxes,save='./data/temp_test/segments_'+str(index)+'.png')
