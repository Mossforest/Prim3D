import os
import trimesh
import torch
import numpy as np
import cv2
from PIL import Image
import kornia
import torchvision

import matplotlib
matplotlib.use("agg")
import seaborn as sns
sns.set()

from myutils.visualization_utils import save_prediction_as_ply_v3, save_prediction_as_ply_v5
from myutils.renderer_romp import Py3DR_mask, Py3DR_SQMesh, Py3DR_mask_syn, Py3DR_mask_test_align
from myutils.romp_vis_utils import rotate_view_perspective_my


def get_colors(M):
    return sns.color_palette("Paired")


def visualize_sq(sq_surface_points, image_name, output_path):
    '''
    Args:
        sq_surface_points: bs x bone_num x num_sampled_points x 3
    '''
    # print ('In utils.visualize.visualize_sq:')
    # print ('sq_surface_points.size is', sq_surface_points.size())
    # os._exit(0)

    batch_size = sq_surface_points.size(0)
    n_primitives = sq_surface_points.size(1)
    colors = get_colors(n_primitives)
    # colors = np.array(colors)

    # for img_n in image_name:
    for bsi in range(batch_size):
        output_directory = os.path.join(output_path, image_name[bsi])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # save_prediction_as_ply_v2(
        #     sq_surface_points[bsi].cpu().detach().data.numpy(),
        #     colors,
        #     os.path.join(output_directory, "primitives.ply")
        # )
        save_prediction_as_ply_v3(
            sq_surface_points[bsi].cpu().detach().data.numpy(),
            colors,
            output_directory
        )


def visualize_sq_new(sq_surface_points, output_path, human_or_object):

    # print ('In utils.visualize.visualize_sq_new:')
    # print ('sq_surface_points.size is', sq_surface_points.size())
    # [num_bone, num_points, 3]
    # os._exit(0)

    n_primitives = sq_surface_points.size(0)
    colors = get_colors(n_primitives)

    output_directory = os.path.join(output_path, human_or_object)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    save_prediction_as_ply_v3(
        sq_surface_points.cpu().detach().data.numpy(),
        colors,
        output_directory
    )


def visualize_sq_new_pts(sq_surface_points, faces, output_path):

    n_primitives = len(sq_surface_points)
    colors = get_colors(n_primitives)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_prediction_as_ply_v5(
        sq_surface_points, faces,
        colors,
        output_path
    )

def render_seg_mask_new_pts(vs_list, fs_list, img, output_path, frame_id):
    img = np.array(img.cpu())  # (H, W, 3)
    img = np.transpose(img, (1, 2, 0))
    height = img.shape[0]
    width = img.shape[1]

    renderer = Py3DR_mask(height=height, width=width)
    # renderer = Py3DR_SQMesh(height=height, width=width)

    result_image = []
    result_image.append(img)

    output_directory = os.path.join(output_path, str(frame_id) + '.jpg')

    mesh_colors = np.array([[.9, .9, .8] for _ in range(2)])
    vs_list_numpy = []
    for vs in vs_list:
        vs_list_numpy.append(vs.cpu().detach().data.numpy())
    fs_list_numpy = []
    for fs in fs_list:
        fs_list_numpy.append(fs.cpu().detach().data.numpy())
    rendered_mask = renderer(vs_list_numpy, fs_list_numpy, img, mesh_colors=mesh_colors)

    result_image.append(rendered_mask)
    result_image = np.concatenate(result_image, axis=0)

    cv2.imwrite(output_directory, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    # return result_image


def render_from_three_views(image_path, vs_list, fs_list, output_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)  # (H, W, 3)
    height = img.shape[0]
    width = img.shape[1]

    background = np.ones([height, width, 3], dtype=np.uint8) * 255

    renderer = Py3DR_SQMesh(height=height, width=width)

    result_image = []
    result_image.append(img)

    output_directory = os.path.join(output_path, 'total.jpg')

    mesh_colors = np.array([[.9, .9, .8] for _ in range(2)])
    vs_list_numpy = []
    for vs in vs_list:
        vs_list_numpy.append(vs.cpu().detach().data.numpy())
    fs_list_numpy = []
    for fs in fs_list:
        fs_list_numpy.append(fs.cpu().detach().data.numpy())
    rendered_rgb = renderer(vs_list_numpy, fs_list_numpy, img, mesh_colors=mesh_colors)
    # print ('rendered_rgb.shape is', rendered_rgb.shape)
    renderer.delete()
    cv2.imwrite(output_directory, cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))

    result_image.append(rendered_rgb)
    # result_image = np.concatenate(result_image, axis=0)

    # renderer = Py3DR_SQMesh(height=max(height, width), width=max(height, width))
    # # bird view
    # vs_bird_view_list = rotate_view_perspective_my(vs_list_numpy, rx=90, ry=0)
    # rendered_bv_image = renderer(vs_bird_view_list, fs_list_numpy,
    #                                          background,
    #                                          mesh_colors=mesh_colors)
    # # print (rendered_bv_image.shape)
    # result_image.append(rendered_bv_image)
    # # result_image = np.concatenate(result_image, axis=0)
    # renderer.delete()
    #
    # # side view
    # renderer = Py3DR_SQMesh(height=max(height, width), width=max(height, width))
    # vs_side_view_list = rotate_view_perspective_my(vs_list_numpy, rx=0, ry=90)
    # rendered_sv_image = renderer(vs_side_view_list, fs_list_numpy, background,
    #                                          mesh_colors=mesh_colors)
    # # result_image.append(cv2.resize(rendered_sv_image, (image_height, image_width)))
    # result_image.append(rendered_sv_image)
    # result_image = np.concatenate(result_image, axis=1)
    # renderer.delete()

    # cv2.imwrite(output_directory, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    return result_image


def regroup_parts(verts_list, faces_list):
    bone_num = len(verts_list)
    batch_size = verts_list[0].size(0)

    regrouped_vs_list = [[] for i in range(batch_size)]
    regrouped_fs_list = [[] for i in range(batch_size)]
    for i in range(bone_num):
        for j in range(batch_size):
            part_i_vs = verts_list[i][j]
            part_i_fs = faces_list[i][j].cuda()
            regrouped_vs_list[j].append(part_i_vs)
            regrouped_fs_list[j].append(part_i_fs)

    return regrouped_vs_list, regrouped_fs_list


def visualize_predictions_training_data(
    pred_object_vs,
    object_fs,
    seg_mask_image,
    image_names, out_path, frame_id
):
    batch_size = 1

    pred_object_vs_group, object_fs_group = regroup_parts(pred_object_vs, object_fs)

    rendered_imgs = []
    for bsi in range(batch_size):
        img_id = image_names[bsi]
        img_id = img_id.split('.')[0]
        outdir = os.path.join(out_path, img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # save pred meshes
        pred_object_vi = pred_object_vs_group[bsi]
        object_fs_i = object_fs_group[bsi]
        visualize_sq_new_pts(pred_object_vi, object_fs_i, outdir)

        # render segmasks and save them
        rendered = render_seg_mask_new_pts(pred_object_vi, object_fs_i, seg_mask_image[bsi], outdir, frame_id)
        rendered_imgs.append(rendered)

    return rendered_imgs


def kornia_projection(kp3d, img_h, img_w):
    FOV = 60
    focal_length = 1 / (np.tan(np.radians(FOV / 2)))
    focal_length = focal_length * max(img_h, img_w) / 2
    K = torch.Tensor(
        [
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ]
    ).cuda()

    # print ('kp3d.size is', kp3d.size())
    # kp3d.size is torch.Size([24, 3])
    # os._exit(0)
    # kp3d = kp3d.unsqueeze(0)
    kp2d = kornia.geometry.camera.perspective.project_points(kp3d, K[None, :, :].repeat(kp3d.size(0), 1, 1))
    # print ('kp2d.size is', kp2d.size())
    # kp2d.size is torch.Size([24, 2])
    # os._exit(0)
    kp2d = kp2d.cpu().detach().data.numpy()

    return kp2d
