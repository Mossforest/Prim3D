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




def cal_box(vertices, faces):
    the_mesh = pytorch3d.structures.Meshes(vertices, faces)
    bbox = the_mesh.get_bounding_boxes()
    bbox = bbox.squeeze(0)

    bbox = bbox.cpu().detach().data.numpy()

    return bbox

def cal_bbox_center(vertices, faces):
    box = cal_box(vertices, faces)
    mids = []

    for bbox in box:
        mid_x = (bbox[0, 1] + bbox[0, 0]) / 2.
        mid_y = (bbox[1, 1] + bbox[1, 0]) / 2.
        mid_z = (bbox[2, 1] + bbox[2, 0]) / 2.
        mid = [mid_x, mid_y, mid_z]
        mids.append(mid)

    return torch.Tensor(mids)


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



load_targets('/mnt/nas-new/home/chenxinyan/3d/term_project/SQ_templates/microwave/plys/SQ_ply', 2)