import numpy as np
import open3d as o3d
import cv2
import os
from shapely.geometry import Point, shape, MultiPoint, Polygon

def get_xylims(points, dataset = 'nuscenes'):
    if dataset != 'nuscenes':
        xlim, ylim = [points[:,0].min(), points[:,0].max()], [points[:,1].min(), points[:,1].max()]
        lim = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        xlim = [sum(xlim)/2 - lim/2, sum(xlim)/2 + lim/2]
        ylim = [sum(ylim)/2 - lim/2, sum(ylim)/2 + lim/2]
    else:
        xlim = [-54., 54.]
        ylim = [-54., 54.]
    return xlim, ylim


def points_to_polygon(points): 
    area = MultiPoint(points).convex_hull
    return area
