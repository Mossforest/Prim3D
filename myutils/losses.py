# -----------------------------------------------------------------------------
# Code adapted from:
# https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/blob/719b0f1ca5ba370616cb837c03ab88d9a88173ff/chamfer_python.py
#
# MIT License
#
# Copyright (c) 2019 ThibaultGROUEIX
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import numpy as np
# import pytorch3d
# import pytorch3d.loss as p3dloss
from lapsolver import solve_dense
import torch.nn.functional as F



def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    P = rx.t() + ry - 2 * zz
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    x, y = a, b
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return torch.min(P, 2)[0], torch.min(P, 1)[0], torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()


# def chamfer_loss(predictions, targets):
#     """Compute the bidirectional Chamfer loss between the target and the
#     predicted shape.
#     """
#     point_num = targets.size(1)
#     pred_points_num = predictions.size(1)
#     pred_select_idx = torch.randint(low=0, high=pred_points_num, size=(point_num,))

#     preds = predictions[:, pred_select_idx, :]
#     # print ('preds.requires_grad', preds.requires_grad)
#     # os._exit(0)
#     if len(targets.size()) == 4:
#         targets = torch.reshape(targets, (targets.size(0), targets.size(2), targets.size(3)))

#     loss = p3dloss.chamfer_distance(preds, targets)

#     return loss[0]


def mseloss(input, target):
    loss_fn = nn.MSELoss()
    return loss_fn(input, target)


def weighted_mseloss(input, target, weights_map):
    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(input, target)  # [3, H, W]
    loss = loss * weights_map[:, :, :].repeat(3, 1, 1)

    return loss.mean()


def L1loss(input, target):
    loss_fn = nn.L1Loss()
    return loss_fn(input, target)


def project_points(p, cam):
    '''
    p: batch_size x bone_num x 2 x 3
    '''
    s, tx, ty = cam
    p = torch.reshape(p, (p.size(0), -1))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    new_x = s * x + tx
    new_y = s * y + ty

    new_p = torch.cat([new_x, new_y], dim=-1)
    return new_p


def keypoint_loss_v2(kp_pred, joints, skeleton, camera):
    '''
    Assume the shapes are like this:
    kp_pred: batch_size x bone_num x 2 x 3
    joints: num_J x 3
    skeleton: 15 x 2

    compute kp loss in both 2D and 3D space
    '''
    # print ('In keypoint_loss_v2:')
    # print ('kp_pred.size is', kp_pred.size())
    # os._exit(0)
    kp_pred = torch.reshape(kp_pred, (kp_pred.size(0), -1, 2, 3))
    batch_size = kp_pred.size(0)
    num_lines = len(skeleton)

    kp_gt_list = []
    for i in range(num_lines):
        kp_gt_list.append(joints[skeleton[i]])
    kp_gt = torch.cat(kp_gt_list, dim=0)
    kp_gt = torch.reshape(kp_gt, (-1, 2, 3))
    # print ('kp_gt.size is', kp_gt.size())
    # (15, 2, 3)
    # os._exit(0)
    kp_gt = kp_gt[None, :, :, :].repeat(batch_size, 1, 1, 1)
    loss_3D = mseloss(kp_pred, kp_gt)

    return loss_3D

    kp_pred_2D = project_points(kp_pred, camera)
    kp_gt_2D = project_points(kp_gt, camera)
    loss_2D = mseloss(kp_pred_2D, kp_gt_2D)

    print ('loss_3D is', loss_3D, ', loss_2D is', loss_2D)

    loss = loss_2D + loss_3D
    return loss


def keypoint_loss(kp_pred, kp_exist_pred, kp_connect_pred, focal_p_pred, threshold=0.):
    # print ('-----------------------------In keypoint_loss-----------------------------')
    batch_size = kp_pred.size(0)
    max_kn = kp_pred.size(1)

    kp_pred = torch.nn.functional.normalize(kp_pred)
    #
    # print ('kp_pred max is', torch.max(kp_pred))
    # print ('focal_p_pred max is', torch.max(focal_p_pred[0]))
    # os._exit(0)

    # threshold = torch.ones(batch_size, max_kn, 1) * threshold
    kp_exist_pred = (kp_exist_pred - threshold) / (kp_exist_pred - threshold + 0.000001)

    # deal with pred keypoints
    if len(kp_exist_pred.size()) != 3:
        kp_exist_pred = torch.reshape(kp_exist_pred, (kp_exist_pred.size(0), kp_exist_pred.size(1), 1))
    kp_pred_masked = kp_pred * kp_exist_pred # (B, kn, 2)
    kp_connect_pred_masked = kp_connect_pred * kp_exist_pred # (B, kn, kn)
    # print (kp_connect_pred_masked)
    # os._exit(0)
    # kp_connect_pred_masked = torch.gt(kp_connect_pred_masked, 0.) # (B, kn, kn) True/False matrix - This will make requires_grad becomes False
    connect_idx = torch.nonzero(torch.nn.functional.relu(kp_connect_pred_masked - threshold))
    # print ('connect_idx size is', connect_idx.size(), ', kp_connect_pred_masked size is', kp_connect_pred_masked.size(), 'kp_connect_pred_masked.requires_grad is', kp_connect_pred_masked.requires_grad)
    select_kp = torch.zeros(batch_size, 2 * max_kn * max_kn, 2).cuda()
    for bs in range(batch_size):
        tmp_connect_idx = connect_idx[connect_idx[:, 0]==bs]
        # print ('tmp_connect_idx size is', tmp_connect_idx.size())
        # print ('kp_pred_masked size is', kp_pred_masked.size(), kp_pred_masked.requires_grad)
        select_kp_1 = kp_pred_masked[tmp_connect_idx[:, 0], tmp_connect_idx[:, 1], :]
        select_kp_2 = kp_pred_masked[tmp_connect_idx[:, 0], tmp_connect_idx[:, 2], :]
        # print ('select_kp_1 size is', select_kp_1.size(), 'select_kp_2 size is', select_kp_2.size())
        tmp_select_kp = torch.cat([select_kp_1, select_kp_2], dim=0)
        # print ('tmp_select_kp size is', tmp_select_kp.size())
        # print ('tmp_select_kp.requires_grad is', tmp_select_kp.requires_grad, 'select_kp_1.requires_grad is', select_kp_1.requires_grad)
        for i in range(tmp_select_kp.size(0)):
            select_kp[bs][i] = tmp_select_kp[i]

    # deal with pred focal points
    focal_p_1, focal_p_2 = focal_p_pred # (B, bone_num, 2), (B, bone_num, 2)
    focal_p_cat1 = torch.cat([focal_p_1, focal_p_2], dim=1) # (B, bone_num * 2, 2)
    focal_p_cat2 = torch.cat([focal_p_2, focal_p_1], dim=1) # (B, bone_num * 2, 2)
    focal_p_cat3 = torch.cat([focal_p_1, focal_p_1], dim=1) # (B, bone_num * 2, 2)
    focal_p_cat4 = torch.cat([focal_p_2, focal_p_2], dim=1) # (B, bone_num * 2, 2)
    focal_ps = torch.cat([focal_p_cat1, focal_p_cat2, focal_p_cat3, focal_p_cat4], dim=1) # (B, bone_num * 8, 2)
    select_fp = torch.zeros(batch_size, 2 * max_kn * max_kn, 2).cuda()
    for bs in range(batch_size):
        for i in range(focal_ps.size(1)):
            select_fp[bs][i] = focal_ps[bs][i]

    kps = select_kp.clone()
    fps = select_fp.clone()
    # print ('kps size is', kps.size(), ', fps size is', fps.size())
    # print ('select_kp.requires_grad is', select_kp.requires_grad, ', select_fp.requires_grad is', select_fp.requires_grad)

    # calculate cost matrix
    input_ds = select_kp.cpu().detach().data.numpy()
    target_ds = select_fp.cpu().detach().data.numpy()
    row_cols = []
    for bs in range(batch_size):
        # cost_matrix = []
        input_d = input_ds[bs]
        target_d = target_ds[bs]
        cost_matrix = np.array([[np.linalg.norm(i-j) for j in target_d] for i in input_d])
        # print ('cost_matrix shape is', cost_matrix.shape)
        # hungarian = Hungarian(cost_matrix)
        # hungarian.calculate()
        # row_col = hungarian.get_results()
        # row_col = np.array(row_col)
        rids, cids = solve_dense(cost_matrix)
        row_col = np.vstack((rids, cids))
        row_col = row_col.transpose(1, 0)
        # print ('row_col shape is', row_col.shape)
        row_cols.append(row_col)
    row_cols = np.array(row_cols)
    # print ('row_cols shape is', row_cols.shape)
    # print ('kps is_cuda:', kps.is_cuda, ', fps is_cuda', fps.is_cuda)
    # print ('kps.requires_grad:', kps.requires_grad, 'fps.requires_grad:', fps.requires_grad)
    # os._exit(0)

    # calculate loss
    row_cols = torch.Tensor(row_cols).long().cuda()
    loss = 0.
    for bs in range(row_cols.size(0)):
        row_col = row_cols[bs]
        for i in range(row_cols.size(1)):
            loss += ((kps[bs, row_col[i][0]] - fps[bs, row_col[i][1]]) ** 2).sum()
    return loss

##weights batch*point_num, weights.sum(1)==1
def compute_geometric_loss_l1(gt_pc, predict_pc, weights=[]):
    if (len(weights) == 0):
        loss = torch.abs(gt_pc - predict_pc).mean()

        return loss
    else:
        batch = gt_pc.shape[0]
        point_num = gt_pc.shape[1]
        pc_weights = weights.view(batch, point_num, 1).repeat(1, 1, 3)
        loss = torch.abs(gt_pc * pc_weights - predict_pc * pc_weights).sum() / (batch * 3)

        return loss

# in_pc size batch*in_size*3
# out_pc batch*in_size*3
# neighbor_id_lstlst out_size*max_neighbor_num
# neighbor_dist_lstlst out_size*max_neighbor_num
def compute_laplace_loss_l1(param, gt_pc_raw, predict_pc_raw):
    initial_neighbor_id_lstlst = torch.LongTensor(param.neighbor_id_lstlst).cuda()  # point_num*max_neighbor_num
    initial_neighbor_num_lst = torch.FloatTensor(param.neighbor_num_lst).cuda()  # point_num
    point_num = param.point_num
    initial_max_neighbor_num = initial_neighbor_id_lstlst.shape[1]

    gt_pc = gt_pc_raw * 1
    predict_pc = predict_pc_raw * 1

    batch = gt_pc.shape[0]

    gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).cuda()), 1)
    predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).cuda()), 1)

    batch = gt_pc.shape[0]

    gt_pc_laplace = gt_pc[:, initial_neighbor_id_lstlst[:, 0]]  ## batch*point_num*3 the first point is itself
    gt_pc_laplace = gt_pc_laplace * (
                initial_neighbor_num_lst.view(1, point_num, 1).repeat(batch, 1, 3) - 1)

    for n in range(1, initial_max_neighbor_num):
        # print (neighbor_id_lstlst[:,n])
        neighbor = gt_pc[:, initial_neighbor_id_lstlst[:, n]]
        gt_pc_laplace -= neighbor

    predict_pc_laplace = predict_pc[:,
                         initial_neighbor_id_lstlst[:, 0]]  ## batch*point_num*3 the first point is itself
    predict_pc_laplace = predict_pc_laplace * (
                initial_neighbor_num_lst.view(1, point_num, 1).repeat(batch, 1, 3) - 1)

    for n in range(1, initial_max_neighbor_num):
        # print (neighbor_id_lstlst[:,n])
        neighbor = predict_pc[:, initial_neighbor_id_lstlst[:, n]]
        predict_pc_laplace -= neighbor

    loss_l1 = torch.abs(gt_pc_laplace - predict_pc_laplace).mean()

    # gt_pc_curv= gt_pc_laplace.pow(2).sum(2).pow(0.5)
    # predict_pc_curv = predict_pc_laplace.pow(2).sum(2).pow(0.5)
    # loss_curv = (gt_pc_curv-predict_pc_curv).pow(2).mean()

    return loss_l1  # , loss_curv

# in_pc size batch*in_size*3
# out_pc batch*in_size*3
# neighbor_id_lstlst out_size*max_neighbor_num
# neighbor_dist_lstlst out_size*max_neighbor_num
def compute_laplace_Mean_Euclidean_Error(self, gt_pc_raw, predict_pc_raw):
    gt_pc = gt_pc_raw * 1
    predict_pc = predict_pc_raw * 1

    batch = gt_pc.shape[0]

    gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).cuda()), 1)
    predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).cuda()), 1)

    batch = gt_pc.shape[0]

    gt_pc_laplace = gt_pc[:, self.initial_neighbor_id_lstlst[:, 0]]  ## batch*point_num*3 the first point is itself
    gt_pc_laplace = gt_pc_laplace * (
                self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1, 3) - 1)

    for n in range(1, self.initial_max_neighbor_num):
        # print (neighbor_id_lstlst[:,n])
        neighbor = gt_pc[:, self.initial_neighbor_id_lstlst[:, n]]
        gt_pc_laplace -= neighbor

    predict_pc_laplace = predict_pc[:,
                         self.initial_neighbor_id_lstlst[:, 0]]  ## batch*point_num*3 the first point is itself
    predict_pc_laplace = predict_pc_laplace * (
                self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1, 3) - 1)

    for n in range(1, self.initial_max_neighbor_num):
        # print (neighbor_id_lstlst[:,n])
        neighbor = predict_pc[:, self.initial_neighbor_id_lstlst[:, n]]
        predict_pc_laplace -= neighbor

    error = torch.pow(torch.pow(gt_pc_laplace - predict_pc_laplace, 2).sum(2), 0.5).mean()

    # gt_pc_curv= gt_pc_laplace.pow(2).sum(2).pow(0.5)
    # predict_pc_curv = predict_pc_laplace.pow(2).sum(2).pow(0.5)
    # loss_curv = (gt_pc_curv-predict_pc_curv).pow(2).mean()

    return error  # , loss_curv


def iou_pytorch(outputs, labels):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    if len(outputs.size()) == 4:
        outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    if len(labels.size()) == 4:
        labels = labels.squeeze(1)

    SMOOTH = 1e-6

    batch_size = outputs.size(0)

    # intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    # union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    intersection = (outputs * labels).view(batch_size, -1).sum(-1)
    union = torch.max(outputs, labels).view(batch_size, -1).sum(-1)

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    # print ('iou.size is', iou.size())
    # iou.size is torch.Size([bs])
    # os._exit(0)

    return iou.mean()


def part_idx_loss(pred_idx, target_idx):
    # print ('pred_idx.size is', pred_idx.size())
    # pred_idx.size is torch.Size([8, 3, 224, 224])
    # os._exit(0)
    # print ('target_idx.size is', target_idx.size())
    # os._exit(0)
    batch_size = pred_idx.size(0)
    data_dim = pred_idx.size(1)
    pred_idx = torch.reshape(pred_idx, (batch_size, data_dim, -1))
    target_idx = torch.argmax(target_idx, dim=1)
    target_idx = torch.reshape(target_idx, (batch_size, -1))

    return F.cross_entropy(pred_idx, target_idx.long())


def uv_loss(pred_idx, target_idx, pred_uv, target_uv, bone_num=2):
    batch_size = pred_idx.size(0)

    data_dim = bone_num + 1
    pred_u = pred_uv[:, :data_dim]
    pred_v = pred_uv[:, data_dim:]
    # print ('target_uv.size is', target_uv.size())
    # os._exit(0)
    target_u = target_uv[:, 0]
    target_v = target_uv[:, 1]
    target_u = target_u[:, None, :, :].repeat(1, data_dim, 1, 1)
    target_v = target_v[:, None, :, :].repeat(1, data_dim, 1, 1)
    # print ('target_u.size is', target_u.size())
    # os._exit(0)
    # print ('pred_u.size is', pred_u.size(), ', pred_v.size is', pred_v.size())
    # pred_u.size is torch.Size([8, 3, 224, 224]) , pred_v.size is torch.Size([8, 3, 224, 224])
    # os._exit(0)
    loss_U = F.smooth_l1_loss(pred_u[target_idx > 0], target_u[target_idx > 0], reduction='sum') / batch_size
    loss_V = F.smooth_l1_loss(pred_v[target_idx > 0], target_v[target_idx > 0], reduction='sum') / batch_size

    # os._exit(0)

    return loss_U + loss_V
