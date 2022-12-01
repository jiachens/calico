import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import matplotlib
import os
from .utils import *

def draw_bbox_2d(ax, bboxes_id_color):
    for bboxes, bboxes_ids, color in bboxes_id_color:
        for i in range(bboxes.shape[0]):
            boxp = cv2.boxPoints(((bboxes[i][0], bboxes[i][1]), (bboxes[i][3], bboxes[i][4]), bboxes[i][6]/np.pi*180))
            boxp = np.insert(boxp, boxp.shape[0], boxp[0,:], 0)
            xs, ys = zip(*boxp)
            ax.plot(xs, ys, linewidth=1, color=color)
            if bboxes_ids is not None:
                ax.text(xs[0], ys[0], str(bboxes_ids[i]), fontsize='xx-small')

def draw_simplified_bbox_2d(ax, bboes):
    for bbox in bboes:
        boxp = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2],bbox[1]), (bbox[0], bbox[1])]
        xs, ys = zip(*boxp)
        ax.plot(xs, ys, linewidth=1, color='black')

def draw_matplotlib(pointclouds, gt_bboxes=None, pred_bboxes=None, gt_bboxes_ids=None, pred_bboxes_ids=None, show=False, save=None):
    if isinstance(pointclouds, list):
        pointcloud_all = np.vstack([p[:,:3] for p in pointclouds])
    else:
        pointcloud_all = pointclouds[:,:3]

    fig, ax = plt.subplots(figsize=(40,40))
    xlim, ylim = get_xylims(pointcloud_all)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(pointcloud_all[:,0], pointcloud_all[:,1], s=25, c="blue")

    total_bboxes = []
    if gt_bboxes is not None:
        total_bboxes.append((gt_bboxes, gt_bboxes_ids, "g"))
    if pred_bboxes is not None:
        total_bboxes.append((pred_bboxes, pred_bboxes_ids, "black"))

    draw_bbox_2d(ax, total_bboxes)

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()

def draw_multi_cluster_matplotlib(pointclouds, gt_bboxes=None, pred_bboxes=None, gt_bboxes_ids=None, pred_bboxes_ids=None, show=False, save=None):
    assert isinstance(pointclouds, list), "pointclouds must be a list for mutli-cluster visualization"
    pointcloud_all = np.vstack([p[:,:3] for p in pointclouds])

    fig, ax = plt.subplots(figsize=(40,40))
    xlim, ylim = get_xylims(pointcloud_all)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for pointcloud in pointclouds:
        ax.scatter(pointcloud[:,0], pointcloud[:,1], s=25)

    total_bboxes = []
    if gt_bboxes is not None:
        total_bboxes.append((gt_bboxes, gt_bboxes_ids, "g"))
    if pred_bboxes is not None:
        total_bboxes.append((pred_bboxes, pred_bboxes_ids, "r"))

    draw_bbox_2d(ax, total_bboxes)

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()

def draw_mutlti_cluster_polygon_matplotlib(pointclouds, show=False, bboxes=None, save=None):
    fig, ax = plt.subplots(figsize=(40,40))
    assert isinstance(pointclouds, list), "pointclouds must be a list for mutli-cluster visualization"
    pointcloud_all = np.vstack([p[:,:3] for p in pointclouds])
    xlim, ylim = get_xylims(pointcloud_all)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for pointcloud in pointclouds:
        ax.scatter(pointcloud[:,0], pointcloud[:,1], s=25)
        area = points_to_polygon(pointcloud[:, :3])
        x, y = area.exterior.coords.xy
        plt.plot(x, y, "r", alpha=0.8)
    if bboxes is not None:
        draw_simplified_bbox_2d(ax, bboxes)
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()



if __name__ == '__main__':
    from nuscenes.utils.data_classes import LidarPointCloud
    pointcloud = LidarPointCloud.from_file("./data/nuscenes/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800543047452.pcd.bin")
    draw_matplotlib(pointcloud.points.T, save="./data/test.png")