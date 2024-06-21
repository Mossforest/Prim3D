'''
The main code for training the model
'''
import os
from datetime import datetime
import argparse
import numpy as np
import scipy.io
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from tensorboardX import SummaryWriter
import yaml
import json
import torch.utils.data
import string
import random
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from datasets import Datasets
from renderer_nvdiff import Nvdiffrast

from networks.baseline_network import Network_pts
import configparser
from myutils.losses import mseloss, distChamfer
from learnable_primitives.loss_functions import sample_surface_vertices

class Mesh:
    def __init__(self, vertices, faces, colors):
        self.vertices = vertices
        self.faces = faces
        self.colors = colors


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='Determine if we run the code for training or testing. Chosen from [train, test]')
parser.add_argument('--log_dir', type=str, default='/root/Prim3D/output/logs')
parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--output_directory', type=str, default='../../NewPrimReg_outputs_iccv/baseline/output_dir')
parser.add_argument('--output_directory', type=str, default='/root/Prim3D/output')
parser.add_argument('--experiment_tag', type=str, default='microwave')
parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--vit_f_dim', type=int, default=3025) # dino
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--vit_f_dim', type=int, default=384) # dinov2
# parser.add_argument('--res', type=int, default=112)
parser.add_argument('--res', type=int, default=224)
parser.add_argument('--batch_size_train', type=int, default=1, help='Batch size of the training dataloader.')
parser.add_argument('--batch_size_val', type=int, default=1, help='Batch size of the val dataloader.')
parser.add_argument('--data_path', type=str, default='/root/Prim3D/d3dhoi_video_data/microwave')
parser.add_argument('--template_path', type=str, default='/root/Prim3D/SQ_templates/microwave')
parser.add_argument('--data_load_ratio', type=float, default=1.0)
parser.add_argument('--save_every', type=int, default=200)
parser.add_argument('--val_every', type=int, default=200)
parser.add_argument('--config_file', type=str, default="config/tmp_config.yaml")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--annealing_lr', type=bool, default=True)
parser.add_argument('--continue_from_epoch', type=int, default=0)

args = parser.parse_args()


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


class OptimizerWrapper(object):
    def __init__(self, optimizer, aggregate=1):
        self.optimizer = optimizer
        self.aggregate = aggregate
        self._calls = 0

    def zero_grad(self):
        if self._calls == 0:
            self.optimizer.zero_grad()

    def step(self):
        self._calls += 1
        if self._calls == self.aggregate:  # ?
            self._calls = 0
            self.optimizer.step()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer == "SGD":
        return OptimizerWrapper(
            torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                            weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "Adam":
        return OptimizerWrapper(
            torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "RAdam":
        return OptimizerWrapper(
            torch.optim.RAdam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    else:
        raise NotImplementedError()


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def id_generator(text):
    # 获取当前时间
    now = datetime.now()

    # 格式化时间字符串，例如：20240611_153045
    formatted_time = now.strftime("%d_%H%M%S")

    # 拼接字符串 "xxxx_time"
    time_string = f"{text}_{formatted_time}"
    return time_string


def save_checkpoints(epoch, model, optimizer, experiment_directory, args):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    # The optimizer is wrapped with an object implementing gradient
    # accumulation
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    if args.checkpoint_model_path is None:
        model_files = [
            f for f in os.listdir(experiment_directory)
            if f.startswith("model_")
        ]

        if len(model_files) == 0:
            return
        ids = [int(f[6:]) for f in model_files]
        max_id = max(ids)
        model_path = os.path.join(
            experiment_directory, "model_{:05d}"
        ).format(max_id)
        opt_path = os.path.join(experiment_directory, "opt_{:05d}").format(max_id)
        if not (os.path.exists(model_path) and os.path.exists(opt_path)):
            return

        print("Loading model checkpoint from {}".format(model_path))
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loading optimizer checkpoint from {}".format(opt_path))
        optimizer.load_state_dict(
            torch.load(opt_path, map_location=device)
        )
        args.continue_from_epoch = max_id+1
    else:
        print("Loading model checkpoint from {}".format(args.checkpoint_model_path))
        model.load_state_dict(torch.load(args.checkpoint_model_path, map_location=device))
        print("Loading optimizer checkpoint from {}".format(args.checkpoint_opt_path))
        optimizer.load_state_dict(
            torch.load(args.checkpoint_opt_path, map_location=device)
        )


def test():
    print ('To be finished...')


def train():
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d"%(args.gpu_id))
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    experiment_tag = id_generator(args.experiment_tag)

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Get the parameters and their ordering for the spreadsheet
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_tag))

    # Set log_dir for tensorboard
    log_dir = os.path.join(args.log_dir, experiment_tag)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    if hasattr(args, 'config_file'):
        config = load_config(args.config_file)
        epochs = config["training"].get("epochs", 500)
    else:
        epochs = args.epoch
    
    from myutils.graphAE_param import Parameters
    GraphAE_config = Parameters()
    GraphAE_config.read_config("graphAE_weight/my_graphAE.config")

    # TODO: Create the network
    net = Network_pts(
        graphAE_param=GraphAE_config,
        test_mode=False,  #?
        model_type='dinov2_vits14',
        stride=4, # 默认值
        device=device,
        vit_f_dim=args.vit_f_dim,
    )
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  #先不用 factory
    if args.annealing_lr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # Load the checkpoints if they exist in the experiment directory
    # load_checkpoints(net, optimizer, experiment_directory, args, device)

    # TODO: create the dataloader
    train_dataset = Datasets(data_path=args.data_path, template_path=args.template_path, train=True, image_size=args.res, data_load_ratio=args.data_load_ratio)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=0, drop_last=True)
    val_dataset = Datasets(data_path=args.data_path, template_path=args.template_path, train=False, image_size=args.res)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=0)
    print ('Dataloader finished!')
    
    sq_template = train_dataset.get_template()
    sq_attr, sq_mesh, sq_joint_info, sq_part_centers = sq_template
    
    # TODO: create the differtiable renderer
    renderer = Nvdiffrast(FOV=39.6)
    print ('Renderer set!')

    print ('Start Training!')
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.continue_from_epoch, epochs):
        net.train()

        total_loss = 0.
        iter_num = 0.
        for _, X in enumerate(tqdm(train_dataloader)):
            data_dict = X

            # TODO: load all the data you need from dataloader, not limited
            whole_rgb_image = data_dict['rgb_image'].cuda() #torch.Size([1, 27, 3, 224, 224])
            whole_o_image = data_dict["o_image"].cuda() #torch.Size([1, 27, 3, 224, 224])
            object_white_mask = data_dict['gt_mask'].cuda() #torch.Size([1, 27, 1, 224, 224])
            object_input_pts= [x.cuda() for x in data_dict["object_input_pts"]]
            init_object_old_center=data_dict["init_object_old_center"].cuda()
            object_joint_tree = data_dict["object_joint_tree"].cuda()
            object_primitive_align = data_dict["object_primitive_align" ].cuda()
            object_joint_parameter_leaf = data_dict["object_joint_parameter_leaf"].cuda()
            global_faces = [x.cuda() for x in data_dict["faces"]]
            
            sq_dict, mesh_dict, joint_info, part_centers = train_dataset.get_template()
            sq_params = sq_dict['pred_sq']  # (2, 5)
            sq_rots = sq_dict['pred_rots']  # (2, 3, 3)

            # 计算每帧平均loss
            total_frame_loss = 0.
            frame_num = whole_rgb_image.shape[1]
            # TODO: pass the input data to the network and generate the predictions
            for frame_id in range(frame_num):  #frame length
                rgb_image= whole_rgb_image[:, frame_id, :, :, :]  #torch.Size([1, 3, 224, 224])
                o_image = whole_o_image[:, frame_id, :, :, :] #torch.Size([1, 3, 224, 224])
                
                pred_dict = net(
                    rgb_image=rgb_image,  #没问题
                    o_image=o_image,  #没问题
                    object_input_pts=object_input_pts,    #==
                    init_object_old_center=init_object_old_center, #==
                    object_num_bones=2,
                    object_joint_tree=object_joint_tree,  # 找到了
                    object_primitive_align=object_primitive_align,   # 找到了
                    object_joint_parameter_leaf=object_joint_parameter_leaf,  # 找到了
                    cam_trans=None, #没用！
                    layer=None, #没用！
                    facet=None, #没用！
                    bin=None    #没用
                )

                # Part segment masks loss
                seg_map_list=[]
                for i in range(2):
                    vertices = pred_dict['deformed_object'][i][0] #torch.Size([1, 656, 3])
                    faces = global_faces[i][0].to(dtype=torch.int32)  #torch.Size([1308, 3])
                    num_vertices = vertices.shape[0]  # 656
                    colors = torch.ones(num_vertices,3).cuda() #torch.Size([656, 3])  #白色的mask
                    mesh = Mesh(vertices,faces,colors)
                    seg_map = renderer(mesh,(0,0,0,0,224,224),focal_length=0) #focal_length用不上  #torch.Size([1, 224, 224, 3])
                    seg_map = seg_map[0].permute(2,0,1) #torch.Size([3, 224, 224])
                    seg_map = seg_map[0].unsqueeze(0)
                    seg_map_list.append(seg_map)
                whole_seg_map = seg_map_list[0]+seg_map_list[1] #torch.Size([1, 224, 224])
                part_segment_mask_loss = mseloss(whole_seg_map, object_white_mask[0][frame_id])

                    
                # TODO 3D keypoint loss
                pred_3d_keypoint = pred_dict['deformed_object_pivot_loc']
                # gt_3d_keypoint = ???
                # keypoint_loss=L1Loss(pred_3d_keypoint,gt_3d_keypoint)

                # Joint angle loss
                pred_angle= pred_dict['object_pred_angle_leaf'][0][0]
                gt_angle = data_dict['jointstate'][0][frame_id].cuda()
                joint_angle_loss = mseloss(pred_angle, gt_angle) * 10
                
                # surface vertices loss
                pred_rot = pred_dict['object_pred_rotmat_root'].squeeze()  # (1, 3, 3)
                pred_trans = pred_dict['object_pred_total_trans'].view(3, 1)  # (1, 1, 3)
                surface_vertices_loss = 0
                for i in range(2):
                    pred_surface_sample = sample_surface_vertices(sq_params[i], pred_rot, pred_trans)
                    gt_surface_sample = sample_surface_vertices(sq_params[i], sq_rots[i], torch.zeros((3, 1)))
                    surface_vertices_loss += distChamfer(pred_surface_sample, gt_surface_sample)

                loss = part_segment_mask_loss + joint_angle_loss + surface_vertices_loss
                total_frame_loss += loss

            # TODO: compute loss functions
            loss = total_frame_loss / frame_num 

            # TODO: write the loss to tensorboard
            writer.add_scalar('train/loss', loss, epoch)

            total_loss += loss.item()

            iter_num += 1
            optimizer.zero_grad()
            loss.backward()
            if args.annealing_lr:
                scheduler.step()
            optimizer.step()

        total_loss = float(total_loss) / float(iter_num)
        print ('[Epoch %d/%d] Total_loss = %f.' % (epoch, epochs, total_loss))

        if epoch % args.save_every == 0:
            save_checkpoints(
                epoch,
                net,
                optimizer,
                experiment_directory,
                args
            )
        
        continue

        if epoch % args.val_every == 0:
            print("====> Validation Epoch ====>")
            net.eval()

            total_eval_loss = 0.
            iter_num = 0.
            for imgi, X in enumerate(tqdm(val_dataloader)):
                # TODO: load data and generate the predictions, loss
                iter_num += 1

                data_dict = X
                whole_rgb_image = data_dict['rgb_image'].cuda() #torch.Size([1, 27, 3, 224, 224])
                whole_o_image = data_dict["o_image"].cuda() #torch.Size([1, 27, 3, 224, 224])
                object_white_mask = data_dict['gt_mask'].cuda() #torch.Size([1, 27, 1, 224, 224])
                object_input_pts= [x.cuda() for x in data_dict["object_input_pts"]]
                init_object_old_center=data_dict["init_object_old_center"].cuda()
                object_joint_tree = data_dict["object_joint_tree"].cuda()
                object_primitive_align = data_dict["object_primitive_align" ].cuda()
                object_joint_parameter_leaf = data_dict["object_joint_parameter_leaf"].cuda()
                global_faces = [x.cuda() for x in data_dict["faces"]]

                # 计算每帧平均loss
                total_frame_loss = 0.
                frame_num = whole_rgb_image.shape[1]
                
                pred_dict_list = []

                for frame_id in range(frame_num):  #frame length
                    rgb_image= whole_rgb_image[:, frame_id, :, :, :]  #torch.Size([1, 3, 224, 224])
                    o_image = whole_o_image[:, frame_id, :, :, :] #torch.Size([1, 3, 224, 224])
                    
                    pred_dict = net(
                        rgb_image=rgb_image,  #没问题
                        o_image=o_image,  #没问题
                        object_input_pts=object_input_pts,    #==
                        init_object_old_center=init_object_old_center, #==
                        object_num_bones=2,
                        object_joint_tree=object_joint_tree,  # 找到了
                        object_primitive_align=object_primitive_align,   # 找到了
                        object_joint_parameter_leaf=object_joint_parameter_leaf,  # 找到了
                        cam_trans=None, #没用！
                        layer=None, #没用！
                        facet=None, #没用！
                        bin=None    #没用
                    )
                    pred_dict['rgb_image'] = rgb_image
                    pred_dict['o_image'] = o_image
                    pred_dict_list.append(pred_dict)

                if epoch % args.save_every == 0:
                    out_path = os.path.join(args.output_directory, experiment_tag)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    out_path = os.path.join(out_path, 'visualiza_results_epoch_%d' % (epoch))
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    # TODO: visualze the predicted results
                    from myutils.visualize_baseline import visualize_predictions_training_data
                    object_fs=[]
                    for f in global_faces:
                        object_fs.append(f.to(dtype=torch.int32))  #torch.Size([1308, 3])
                    
                    for frame_id in range(frame_num):
                        pred_dict = pred_dict_list[frame_id]
                        
                        pred_object_vs = pred_dict['deformed_object'] #torch.Size([1, 656, 3])
                        seg_mask_image = object_white_mask[:, frame_id, :, :, :]
                        rgb_image = pred_dict['rgb_image']
                        image_names = data_dict['img_id']
                        visualize_predictions_training_data(
                            pred_object_vs, object_fs,
                            rgb_image,
                            image_names, out_path, frame_id
                        )

            print("====> Validation Epoch ====>")


    print("Saved statistics in {}".format(experiment_tag))


def debug():
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d"%(args.gpu_id))
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    experiment_tag = id_generator(args.experiment_tag)

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Get the parameters and their ordering for the spreadsheet
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_tag))

    # Set log_dir for tensorboard
    log_dir = os.path.join(args.log_dir, experiment_tag)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    if hasattr(args, 'config_file'):
        config = load_config(args.config_file)
        epochs = config["training"].get("epochs", 500)
    else:
        epochs = args.epoch
    
    # from graphAE_weight.graphAE_config import GraphAEConfig=
    # GraphAE_config=GraphAEConfig()
    # config = configparser.ConfigParser()
    # config.read("graphAE_weight/graphAE.config")
    # print(config)

    from myutils.graphAE_param import Parameters
    GraphAE_config = Parameters()
    GraphAE_config.read_config("graphAE_weight/my_graphAE.config")

    # TODO: create the dataloader
    
    train_dataset = Datasets(data_path=args.data_path, template_path=args.template_path, train=True, image_size=args.res, data_load_ratio=args.data_load_ratio)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=0, drop_last=True)
    val_dataset = Datasets(data_path=args.data_path, template_path=args.template_path, train=False, image_size=args.res)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=0)
    print ('Dataloader finished!')
    
    for _, X in enumerate(tqdm(train_dataloader)):
        breakpoint()
        data_dict = X
    

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'debug':
        debug()
    else:
        print ('Bad Mode!')
        os._exit(0)