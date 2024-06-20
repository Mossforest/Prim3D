import cv2
import os
import torch
import torch.nn as nn
import pytorch3d
import pytorch3d.transforms as p3dt
import faiss
import numpy as np
import romp
import math
import torchvision

from networks.vitextractor import ViTExtractor
from networks.graphae import GraphAE

import sys

sys.path.append('../')
from learnable_primitives.equal_distance_sampler_sq import get_sampler
from myutils.tools import compute_rotation_matrix_from_ortho6d


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=False):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer(feat)
        return trans


class TotalRot6dPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(TotalRot6dPredictor, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat):
        out = self.fc(feat)
        out = self.relu(out)
        out = self.out(out)

        return out


class TotalScalePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(TotalScalePredictor, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat):
        out = self.fc(feat)
        out = self.relu(out)
        out = self.out(out)
        out = torch.sigmoid(out)  # the template is no smaller than the instances
        # out = nn.functional.sigmoid(out)  # the template is no smaller than the instances

        return out


class TotalTransPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(TotalTransPredictor, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat):
        out = self.fc(feat)
        out = self.relu(out)
        out = self.out(out)

        return out


class PartRotAnglePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bone_nums):
        super(PartRotAnglePredictor, self).__init__()
        self.num_bones = bone_nums
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList(
            [predictor_block for i in range(self.num_bones)]
        )

    def forward(self, feat):
        batch_size = feat.size(0)

        # print ('feat.size is', feat.size())
        # [bs, num_bone, in_dim]
        # os._exit(0)

        preds = []
        for i in range(self.num_bones):
            x = self.fcs[i](feat[:, i])
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out


class PrimitiveRot6dPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_bones):
        super(PrimitiveRot6dPredictor, self).__init__()
        self.num_bones = num_bones
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList(
            [predictor_block for i in range(self.num_bones)]
        )

    def forward(self, feat):
        batch_size = feat.size(0)

        preds = []
        for i in range(self.num_bones):
            # x = self.fcs[i](feat[:, i])
            x = self.fcs[i](feat)
            # print ('In PrimitiveRot6dPredictor.forward, x.size is', x.size(), ', feat.size is', feat.size())
            # os._exit(0)
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out


class PrimitiveQuatPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_bones):
        super(PrimitiveQuatPredictor, self).__init__()
        self.num_bones = num_bones
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList(
            [predictor_block for i in range(self.num_bones)]
        )

    def forward(self, feat):
        batch_size = feat.size(0)

        preds = []
        for i in range(self.num_bones):
            x = self.fcs[i](feat)
            x = 2. * nn.functional.softmax(x) - 1.
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out


class PartScalePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=1, bone_nums=2):
        super(PartScalePredictor, self).__init__()
        self.num_bones = bone_nums
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList(
            [predictor_block for i in range(self.num_bones)]
        )

    def forward(self, feat):
        batch_size = feat.size(0)

        preds = []
        for i in range(self.num_bones):
            x = self.fcs[i](feat[:, i])
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out


class LeafAnglePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bone_nums):
        super(LeafAnglePredictor, self).__init__()
        self.num_bones = bone_nums
        self.out_dim = out_dim
        predictor_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.fcs = nn.ModuleList(
            [predictor_block for i in range(self.num_bones)]
        )

    def forward(self, feat):
        batch_size = feat.size(0)

        preds = []
        for i in range(self.num_bones):
            x = self.fcs[i](feat[:, i])
            x = torch.reshape(x, (batch_size, 1, self.out_dim))
            x = torch.tanh(x) * math.pi / 2.
            preds.append(x)

        out = torch.cat(preds, dim=1)

        return out


class ViTExtractorCNN(nn.Module):
    def __init__(self, in_channel, out_channel=384):
        super(ViTExtractorCNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel, 4, 2, 1)
        self.group_norm_1 = nn.GroupNorm(64, out_channel)
        self.LRelu_1 = nn.LeakyReLU(0.2)

        self.conv_2 = nn.Conv2d(out_channel, out_channel, 4, 2, 1)
        self.group_norm_2 = nn.GroupNorm(64, out_channel)
        self.LRelu_2 = nn.LeakyReLU(0.2)

        self.conv_3 = nn.Conv2d(out_channel, out_channel, 4, 2, 0)

    def forward(self, feat):
        out = self.conv_1(feat)
        out = self.group_norm_1(out)
        out = self.LRelu_1(out)

        out = self.conv_2(out)
        out = self.group_norm_2(out)
        out = self.LRelu_2(out)

        out = self.conv_3(out)

        return out


class ObjectNetwork_pts(nn.Module):
    def __init__(self, graphAE_param, test_mode, model_type, stride, device, vit_f_dim, mesh_f_dim=72, hidden_dim=256, bone_num=2):
        super(ObjectNetwork_pts, self).__init__()

        self.bone_num = bone_num
        self.whole_vitextractor = ViTExtractor(model_type, stride, device=device)
        self.object_vitextractor = ViTExtractor(model_type, stride, device=device)
        self.vitextractor_fc = nn.Linear(vit_f_dim, vit_f_dim)
        self.vitextractor_cnn = ViTExtractorCNN(vit_f_dim)

        self.total_rot6d_predictor = TotalRot6dPredictor(vit_f_dim+mesh_f_dim, hidden_dim=hidden_dim, out_dim=6)
        self.part_rotangle_predictor = PartRotAnglePredictor(vit_f_dim+mesh_f_dim, hidden_dim=hidden_dim, out_dim=1, bone_nums=bone_num-1)
        self.part_scale_predictor = PartScalePredictor(vit_f_dim+mesh_f_dim, hidden_dim=hidden_dim, out_dim=3, bone_nums=bone_num)
        self.total_trans_predictor = TotalTransPredictor(vit_f_dim+mesh_f_dim, hidden_dim=hidden_dim, out_dim=3)
        self.leaf_angle_predictor = LeafAnglePredictor(vit_f_dim+mesh_f_dim, hidden_dim=hidden_dim, out_dim=1, bone_nums=bone_num-1)
        self.total_scale_predictor = TotalScalePredictor(vit_f_dim+mesh_f_dim, hidden_dim=hidden_dim, out_dim=1)

        # mesh encoder
        self.mesh_encoder = GraphAE(graphAE_param, test_mode)
        # init graphAE with pre-trained weight and fix weight
        pretained_model = torch.load(graphAE_param.read_weight_path)
        pretrained_dict = pretained_model['model_state_dict']
        graphAE_dict = self.mesh_encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in graphAE_dict}
        # 2. overwrite entries in the existing state dict
        graphAE_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.mesh_encoder.load_state_dict(pretrained_dict)
        # forzen parameters of graphAE
        for param in self.mesh_encoder.parameters():
            param.requires_grad = False

    def cal_rotmat_from_angle_around_axis(self, all_joint_axis, all_the_pred_angle):
        batch_size = all_joint_axis.size(0)
        all_rotmat = []
        for i in range(batch_size):
            joint_axis = all_joint_axis[i].unsqueeze(0)
            the_pred_angle = all_the_pred_angle[i]

            axis_2p_sub = torch.sub(joint_axis, torch.zeros(1, 3).cuda())
            axis_len = 0.
            axis_2p_sub = axis_2p_sub.squeeze(0)
            for sub_i in axis_2p_sub:
                axis_len = axis_len + sub_i * sub_i
            axis_len = torch.sqrt(axis_len)
            axis_norm = axis_2p_sub / axis_len

            # R = torch.deg2rad(the_pred_angle)
            R = the_pred_angle
            rot_mat = pytorch3d.transforms.axis_angle_to_matrix(axis_norm * R)
            all_rotmat.append(rot_mat)
        all_rotmat = torch.stack(all_rotmat, dim=0)

        return all_rotmat

    def deform(
            self,
            input_pts, old_centers, bone_num, joint_tree, primitive_align,
            joint_parameter_leaf, pred_rotmat_root, pred_angle_leaf,
            pred_part_scale, pred_total_scale, pred_total_trans, cam_trans):
        '''
            input_pts: name id order
            old_centers: name id order
            pred_part_scale: joint tree order
        '''

        end_point0 = joint_parameter_leaf[0]
        end_point1 = joint_parameter_leaf[1]
        pivot_loc = joint_parameter_leaf[2]

        x_deformed = []
        new_part_centers = []
        new_pivot_loc = pivot_loc.clone()

        end_point0_scaled = end_point0.clone()
        end_point1_scaled = end_point1.clone()
        pivot_scaled = pivot_loc.clone()
        pivot_scaled_store = pivot_loc.clone()
        for bone_i in range(bone_num):
            if bone_i == 0:
                the_v = input_pts[primitive_align[bone_i]].clone().cuda()
                the_old_center = old_centers[primitive_align[bone_i]].clone()  #!!!
                the_old_center = the_old_center[None, :].repeat(the_v.size(0), 1)
                v_zero_centered = torch.sub(the_v, the_old_center[:, None, :].repeat(1, the_v.size(1), 1))
                the_pred_scale = pred_part_scale[:, bone_i]
                # the_pred_scale = pred_total_scale
                the_v_scaled = v_zero_centered * the_pred_scale[:, None, :].repeat(1, v_zero_centered.size(1), 1)  # [8, 738, 3]
                the_v_rotated = torch.matmul(pred_rotmat_root, torch.permute(the_v_scaled, (0, 2, 1)))  # [8, 3, 738]
                the_v = torch.add(torch.permute(the_v_rotated, (0, 2, 1)), the_old_center[:, None, :].repeat(1, the_v.size(1), 1))
                the_v = torch.add(the_v, pred_total_trans[:, None, :].repeat(1, the_v.size(1), 1))

                p0 = torch.sub(end_point0[None, :].repeat(the_old_center.size(0), 1), the_old_center)
                end_point0_scaled = p0 * the_pred_scale
                end_point0_scaled = torch.add(end_point0_scaled, the_old_center)
                p1 = torch.sub(end_point1[None, :].repeat(the_old_center.size(0), 1), the_old_center)
                end_point1_scaled = p1 * the_pred_scale
                end_point1_scaled = torch.add(end_point1_scaled, the_old_center)

                pivot_zero_centered = torch.sub(pivot_loc[None, :].repeat(the_old_center.size(0), 1), the_old_center)
                pivot_scaled = pivot_zero_centered * the_pred_scale
                pivot_scaled_store = torch.add(pivot_scaled, the_old_center)
                pivot_rotated = torch.matmul(pred_rotmat_root, pivot_scaled.unsqueeze(-1))
                new_pivot_loc = torch.add(pivot_rotated.squeeze(-1), the_old_center)
                new_pivot_loc = torch.add(new_pivot_loc, pred_total_trans)

                new_center = torch.add(the_old_center, pred_total_trans)
            else:
                the_v = input_pts[primitive_align[bone_i]].clone().cuda()
                the_old_center = old_centers[primitive_align[bone_i]].clone()
                the_old_center = the_old_center[None, :].repeat(the_v.size(0), 1)
                joint_axis = torch.sub(end_point1_scaled, end_point0_scaled)
                old_pivot_loc = pivot_loc[None, :].repeat(the_v.size(0), 1)
                the_v_zero_centered = torch.sub(the_v, the_old_center[:, None, :].repeat(1, the_v.size(1), 1))  # [8, 738, 3]
                old_pivot_loc_zero_centered = torch.sub(old_pivot_loc, the_old_center)  # [8, 3]
                the_pred_scale = pred_part_scale[:, bone_i]
                # the_pred_scale = pred_total_scale
                the_v_scaled = the_v_zero_centered * the_pred_scale[:, None, :].repeat(1, the_v_zero_centered.size(1), 1)  # [8, 738, 3]
                old_pivot_loc_scaled = old_pivot_loc_zero_centered * the_pred_scale
                the_v = torch.add(the_v_scaled, the_old_center[:, None, :].repeat(1, the_v.size(1), 1))
                old_pivot_loc = torch.add(old_pivot_loc_scaled, the_old_center)

                the_v = torch.sub(the_v, old_pivot_loc[:, None, :].repeat(1, the_v.size(1), 1))
                new_center = torch.sub(the_old_center, old_pivot_loc)
                the_pred_angle = pred_angle_leaf[:, bone_i-1]
                the_rotmat = self.cal_rotmat_from_angle_around_axis(joint_axis, the_pred_angle)  # [8, 3, 3]

                # rotate around the axis
                the_v = torch.matmul(the_rotmat, torch.permute(the_v, (0, 2, 1)))
                # move back
                the_v = torch.permute(the_v, (0, 2, 1))
                the_v = torch.add(the_v, pivot_scaled_store[:, None, :].repeat(1, the_v.size(1), 1))
                # do the same thing with the_old_center
                new_center = torch.matmul(the_rotmat, torch.permute(new_center[:, None, :].repeat(1, the_v.size(1), 1), (0, 2, 1)))
                new_center = torch.permute(new_center, (0, 2, 1))
                new_center = torch.add(new_center, pivot_scaled_store[:, None, :].repeat(1, the_v.size(1), 1))

                old_center_parent = old_centers[primitive_align[joint_tree[bone_i]]]
                old_center_parent = old_center_parent[None, :].repeat(the_v.size(0), 1)
                the_v = torch.sub(the_v, old_center_parent[:, None, :].repeat(1, the_v.size(1), 1))
                new_center = torch.sub(new_center, old_center_parent[:, None, :].repeat(1, the_v.size(1), 1))

                # global rot and trans
                the_v = torch.matmul(pred_rotmat_root, torch.permute(the_v, (0, 2, 1)))
                the_v = torch.permute(the_v, (0, 2, 1))
                the_v = torch.add(the_v, old_center_parent[:, None, :].repeat(1, the_v.size(1), 1))
                new_center = torch.matmul(pred_rotmat_root, torch.permute(new_center, (0, 2, 1)))
                new_center = torch.permute(new_center, (0, 2, 1))
                new_center = torch.add(new_center, old_center_parent[:, None, :].repeat(1, the_v.size(1), 1))

                the_delta_trans = pred_total_trans
                the_v = torch.add(the_v, the_delta_trans[:, None, :].repeat(1, the_v.size(1), 1))
                new_center = torch.add(new_center, the_delta_trans[:, None, :].repeat(1, the_v.size(1), 1))
                new_center = torch.mean(new_center, dim=-2)
            x_deformed.append(the_v)
            new_part_centers.append(new_center)

        x_deformed_realign = [[] for i in range(bone_num)]
        for i in range(bone_num):
            x_deformed_realign[primitive_align[i]] = x_deformed[i]
        new_center_realign = [[] for i in range(bone_num)]
        for i in range(bone_num):
           new_center_realign[primitive_align[i]] = new_part_centers[i]
        new_center_realign = torch.stack(new_center_realign, dim=1)

        return x_deformed_realign, new_center_realign, new_pivot_loc

    def forward(self, rgb_image, o_image,
                object_input_pts, init_object_old_center, object_num_bones,
                object_joint_tree, object_primitive_align, object_joint_parameter_leaf,
                cam_trans,
                layer, facet, bin):
        batch_size = rgb_image.size(0)  #torch.Size([1, 3, 224, 224])
        # extract feature of the whole image

        ################### for DINOv2 ###################
        transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=0.5, std=0.2)])
        rgb_image = transform(rgb_image)
        whole_result = self.whole_vitextractor.forward_features(rgb_image)
        vit_feature_whole = whole_result['x_norm_patchtokens']
        vit_feature_whole = torch.reshape(vit_feature_whole, (batch_size, 16, 16, -1))  # #torch.Size([1, 16, 16, 384])

        transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=0.5, std=0.2)])
        o_image = transform(o_image)
        object_result = self.object_vitextractor.forward_features(o_image)
        vit_feature_object = object_result['x_norm_patchtokens']
        vit_feature_object = torch.reshape(vit_feature_object, (batch_size, 16, 16, -1))  #torch.Size([1, 16, 16, 384])
        ################### for DINOv2 ###################

        total_vit_feature = vit_feature_whole + vit_feature_object

        # # check.
        total_vit_feature = torch.permute(total_vit_feature, (0, 3, 1, 2))
        total_vit_feature = self.vitextractor_cnn(total_vit_feature)
        total_vit_feature = torch.reshape(total_vit_feature, (batch_size, -1))

        # joint_tree = object_joint_tree[0]  #(1,1,2)->(1,2)
        joint_tree = object_joint_tree[0][0]  #(1,2)->(2)
        # primitive_align = object_primitive_align[0]  #(1,1,2)->(1,2)
        primitive_align = object_primitive_align[0][0]  #(1,1,2)->(1,2)
        joint_parameter_leaf = object_joint_parameter_leaf[0] #(1,3,3)->(3,3)

        init_object_old_center=init_object_old_center[0]  #??

        # mesh feature
        mesh_v = object_input_pts
        mesh_feats = []
        for bone_i in range(object_num_bones):
            mesh_vi = mesh_v[primitive_align[bone_i]].clone().cuda()  #torch.Size([1, 873, 3])
            mesh_feat_i = self.mesh_encoder(mesh_vi) #torch.Size([1, 8, 9])
            mesh_feat_i = torch.reshape(mesh_feat_i, (mesh_feat_i.size(0), -1))
            mesh_feats.append(mesh_feat_i)
        mesh_feats = torch.stack(mesh_feats, dim=1)  # in tree order, example: 0-laptop keyboard, 1-laptop screen
        # mesh_feat = torch.mean(mesh_feats, dim=1)

        mesh_feat_root = mesh_feats[:, 0]
        mesh_feats_leaf = mesh_feats[:, 1:]

        feat_root = torch.cat([total_vit_feature, mesh_feat_root], dim=-1)
        feats_leaf = torch.cat([total_vit_feature[:, None, :].repeat(1, mesh_feats_leaf.size(1), 1), mesh_feats_leaf], dim=-1)

        pred_rot6d_root = self.total_rot6d_predictor(feat_root)
        pred_rotmat_root = compute_rotation_matrix_from_ortho6d(pred_rot6d_root.view(-1, 6))
        pred_rotmat_root = torch.reshape(pred_rotmat_root, (batch_size, 3, 3))
        pred_angle_leaf = self.leaf_angle_predictor(feats_leaf)

        total_feat = torch.cat([total_vit_feature[:, None, :].repeat(1, mesh_feats.size(1), 1), mesh_feats], dim=-1)
        # pred_part_scale = self.part_scale_predictor(total_feat)
        pred_part_scale = torch.ones(batch_size, object_num_bones, 3).cuda()
        pred_total_trans = self.total_trans_predictor(torch.mean(total_feat, dim=-2))
        pred_total_scale = self.total_scale_predictor(torch.mean(total_feat, dim=-2))

        deformed_vs, deformed_part_center, new_pivot_loc = self.deform(
            object_input_pts, init_object_old_center, object_num_bones,
            joint_tree, primitive_align, joint_parameter_leaf,
            pred_rotmat_root, pred_angle_leaf, pred_part_scale,
            pred_total_scale, pred_total_trans,
            cam_trans
        )

        # return object_input_pts, deformed_part_center
        return deformed_vs, deformed_part_center, pred_rotmat_root, pred_angle_leaf, pred_total_trans, new_pivot_loc


class Network_pts(nn.Module):
    def __init__(self,
                 graphAE_param,
                 test_mode,
                 model_type,
                 stride,
                 device,
                 vit_f_dim,
                 hidden_dim=256, object_bone_num=2, human_bone_num=20):
        super(Network_pts, self).__init__()

        self.object_network = ObjectNetwork_pts(graphAE_param, test_mode, model_type, stride, device, vit_f_dim, hidden_dim=hidden_dim, bone_num=object_bone_num)

    def forward(self,
                rgb_image, o_image,
                object_input_pts, init_object_old_center, object_num_bones,
                object_joint_tree, object_primitive_align, object_joint_parameter_leaf,
                cam_trans,
                layer, facet, bin
                ):

        object_deformed, object_deformed_part_center, object_pred_rotmat_root, \
        object_pred_angle_leaf, object_pred_total_trans, new_pivot_loc = self.object_network(
            rgb_image, o_image,
            object_input_pts, init_object_old_center, object_num_bones,
            object_joint_tree, object_primitive_align, object_joint_parameter_leaf,
            cam_trans,
            layer, facet, bin)

        pred_dict = {}
        pred_dict['deformed_object'] = object_deformed
        pred_dict['deformed_object_part_center'] = object_deformed_part_center
        pred_dict['object_pred_rotmat_root'] = object_pred_rotmat_root
        pred_dict['object_pred_angle_leaf'] = object_pred_angle_leaf
        pred_dict['object_pred_total_trans'] = object_pred_total_trans
        pred_dict['deformed_object_pivot_loc'] = new_pivot_loc

        return pred_dict
