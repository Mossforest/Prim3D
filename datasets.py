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

import sys
# sys.path.append('/Disk2/siqi/NewPrimReg')


class Resize_with_pad:
    def __init__(self, w=224, h=224):
        self.w = w
        self.h = h

    def __call__(self, image):

        _, w_1, h_1 = image.size()
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (hp, 0, hp, 0), 0, "constant")
                return F.resize(image, (self.h, self.w))

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (0, wp, 0, wp), 0, "constant")
                return F.resize(image, (self.h, self.w))

        else:
            return F.resize(image, (self.h, self.w))


class Datasets(object):
    def __init__(self, data_path, template_path, train, image_size, data_load_ratio=1.0):
        self.train = train
        self.image_size = image_size
        self.transform = T.Resize((self.image_size, self.image_size))
        
        self.data_key = ['frames', 'gt_mask', 'joints3d', 'smplv2d', '3d_info', 'jointstate']

        self.data_list = self.load_data(data_path, data_load_ratio)
        self.load_template(template_path)

    def __len__(self):
        return len(self.data_list)

    def padding_image(self, image):
        h, w = image.shape[:2]
        side_length = max(h, w)
        pad_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
        top, left = int((side_length - h) // 2), int((side_length - w) // 2)
        bottom, right = int(top + h), int(left + w)
        pad_image[top:bottom, left:right] = image
        image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
        return pad_image, image_pad_info

    def img_preprocess(self, image, input_size=224):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pad_image, image_pad_info = self.padding_image(image)
        input_image = torch.from_numpy(cv2.resize(pad_image, (input_size, input_size), interpolation=cv2.INTER_CUBIC))[
            None].float()
        return input_image, image_pad_info

    def load_images(self, path):
        path = path.strip()
        img = cv2.imread(path)

        img, img_pad_info = self.img_preprocess(img)
        img = img.squeeze(0)
        img = torch.permute(img, (2, 0, 1))

        return img, img_pad_info

    def load_images_v2(self, path):
        path = path.strip()
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        img = torch.Tensor(np.array(img))
        img = torch.permute(img, (2, 0, 1))

        return img

    def load_masks(self, path):
        path = path.strip()
        img = np.load(path)
        img = np.reshape(img, (1, img.shape[0], img.shape[1]))
        img = torch.Tensor(img)
        rwp = Resize_with_pad()
        img = rwp(img)

        return img
    
    def load_3d_info(self, path):
        def parse_line(line):
            """解析单行数据并返回键值对"""
            try:
                key, value = line.split(': ')
                try:
                    # 尝试将值转换为浮点数
                    value = float(value)
                except ValueError:
                    # 如果包含逗号，可能是一个元组，将其转换为浮点数列表
                    if ',' in value:
                        value = [float(v) for v in value.split(',')]
                return key, value
            except ValueError:
                return None

        def load_data_from_file(file_path):
            """从文件中加载数据并存储到字典"""
            data_dict = {}
            with open(file_path, 'r') as file:
                for line in file:
                    res = parse_line(line.strip())
                    if res:
                        key, value = res
                        data_dict[key] = value
            return data_dict

        data = load_data_from_file(path)
        return data
    
    
    def load_jointstate(self, path):
        def load_numbers_from_file(file_path):
            """从文件中加载数字并存储到列表"""
            numbers_list = []
            with open(file_path, 'r') as file:
                for line in file:
                    # 去除每行末尾的换行符并尝试将其转换为整数
                    number = int(line.strip())
                    if not number:
                        break
                    numbers_list.append(number)
            return numbers_list

        numbers = load_numbers_from_file(path)
        return torch.Tensor(numbers)
    
    def load_data(self, data_path, data_load_ratio):
        instance_dict = {key: [] for key in self.data_key}
        instance_paths = sorted([f'{data_path}/{f}' for f in os.listdir(f'{data_path}')])
        if self.train:
            instance_paths = instance_paths[6:]
            instance_paths = instance_paths[:int(len(instance_paths) * data_load_ratio)]
        else:
            instance_paths = instance_paths[:6]
        
        for instance_path in instance_paths:
            # 1. load frames (imgs)
            frame_list = []
            frame_path = f'{instance_path}/frames'
            files = sorted([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
            for file in files:
                file_path = os.path.join(frame_path, file)
                # img = self.load_images(file_path)
                img = self.load_images_v2(file_path)
                frame_list.append(img)
            instance_dict['frames'].append(torch.stack(frame_list))
            
            # 2. load gt_mask (256, 256)
            mask_list = []
            mask_path = f'{instance_path}/gt_mask'
            files = sorted([f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))])
            for file in files:
                file_path = os.path.join(mask_path, file)
                img = self.load_masks(file_path)
                mask_list.append(img)
            instance_dict['gt_mask'].append(torch.stack(mask_list))
            
            # 3. load joints3d (49, 3)
            joints_list = []
            joints_path = f'{instance_path}/joints3d'
            files = sorted([f for f in os.listdir(joints_path) if os.path.isfile(os.path.join(joints_path, f))])
            for file in files:
                file_path = os.path.join(joints_path, file)
                img = torch.Tensor(np.load(file_path))
                joints_list.append(img)
            instance_dict['joints3d'].append(torch.stack(joints_list))

            # 4. load smplv2d (6890, 2)
            smplv2d_list = []
            smplv2d_path = f'{instance_path}/smplv2d'
            files = sorted([f for f in os.listdir(smplv2d_path) if os.path.isfile(os.path.join(smplv2d_path, f))])
            for file in files:
                file_path = os.path.join(smplv2d_path, file)
                img = torch.Tensor(np.load(file_path))
                smplv2d_list.append(img)
            instance_dict['smplv2d'].append(torch.stack(smplv2d_list))
            
            # 5. other files
            instance_dict['3d_info'].append(self.load_3d_info(f'{instance_path}/3d_info.txt'))
            instance_dict['jointstate'].append(self.load_jointstate(f'{instance_path}/jointstate.txt'))
        
        return instance_dict

    def load_sqs(self, path):
        sqs_delta_rots = np.load(os.path.join(path, 'delta_rots.npy'))
        sqs_pred_rots = np.load(os.path.join(path, 'pred_rots.npy'))
        sqs_pred_sq = np.load(os.path.join(path, 'pred_sq.npy'))

        sqs_delta_rots = torch.Tensor(sqs_delta_rots).squeeze()
        sqs_pred_rots = torch.Tensor(sqs_pred_rots).squeeze()
        sqs_pred_sq = torch.Tensor(sqs_pred_sq).squeeze()

        return sqs_delta_rots, sqs_pred_rots, sqs_pred_sq

    def cal_box(self, vertices, faces):
        the_mesh = pytorch3d.structures.Meshes(vertices, faces)
        bbox = the_mesh.get_bounding_boxes()
        bbox = bbox.squeeze(0)

        bbox = bbox.cpu().detach().data.numpy()

        return bbox

    def cal_bbox_center(self, vertices, faces):
        box = self.cal_box(vertices, faces)
        mids = []

        for bbox in box:
            mid_x = (bbox[0, 1] + bbox[0, 0]) / 2.
            mid_y = (bbox[1, 1] + bbox[1, 0]) / 2.
            mid_z = (bbox[2, 1] + bbox[2, 0]) / 2.
            mid = [mid_x, mid_y, mid_z]
            mids.append(mid)

        return torch.Tensor(mids)


    def load_targets(self, path, num_parts):
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
        part_centers = self.cal_bbox_center(vs, fs)

        return vs, fs, part_centers
    
    def load_template(self, template_path):
        
        self.delta_rots, self.pred_rots, self.pred_sq = self.load_sqs(f'{template_path}/plys')
        self.meshs = self.load_targets(f'{template_path}/plys/SQ_ply', 2)
        self.joint_info = scipy.io.loadmat(f'{template_path}/joint_info.mat')
        self.part_centers = np.load(f'{template_path}/part_centers.npy')
    
    def get_template(self):
        sq_dict = {}
        mesh_dict = {}
        sq_dict['delta_rots'], sq_dict['pred_rots'], sq_dict['pred_sq'] = self.delta_rots, self.pred_rots, self.pred_sq
        mesh_dict['vertices'] = self.meshs[0]
        mesh_dict['faces'] = self.meshs[1]
        mesh_dict['part_centers'] = self.meshs[2]
        return sq_dict, mesh_dict, self.joint_info, self.part_centers

    def __getitem__(self, index):
        data_dict = {}
        for key in self.data_key:
            data_dict[key] = data_dict[key][index]

        return data_dict


