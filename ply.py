import cv2
import os
import torch
import scipy.io
import numpy as np
import torchvision.transforms as T
from PIL import Image
import trimesh
import pytorch3d
import pytorch3d.structures as p3dstr
import torchvision.transforms.functional as F
import pickle




def cal_box(path):
    mesh = trimesh.load(path, force='mesh')
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = torch.Tensor(vertices)
    faces = torch.Tensor(faces)
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)

    the_mesh = pytorch3d.structures.Meshes(vertices, faces)
    bbox = the_mesh.get_bounding_boxes()
    bbox = bbox.squeeze(0)

    bbox = bbox.cpu().detach().data.numpy()

    return bbox

def cal_bbox_center(path):
    box = cal_box(path)

    mid_x = (box[0, 1] + box[0, 0]) / 2.
    mid_y = (box[1, 1] + box[1, 0]) / 2.
    mid_z = (box[2, 1] + box[2, 0]) / 2.
    mid = [mid_x, mid_y, mid_z]

    return mid


def load_targets(path, num_parts):
    path = path.strip()
    vs, fs = [], []
    for i in range(num_parts):
        prim_p = os.path.join(path, str(i) + '.ply')
        mesh = trimesh.load(prim_p, force='mesh', process=False)
        vertices = mesh.vertices
        faces = mesh.faces
        vertices = torch.Tensor(vertices)
        faces = torch.Tensor(faces)

        vs.append(vertices)
        fs.append(faces)

    # calculate centers
    part_centers = cal_bbox_center(vs, fs)

    return vs, fs, part_centers



load_targets()