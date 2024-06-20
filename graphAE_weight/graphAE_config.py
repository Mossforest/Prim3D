from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class GraphAEConfig:
# [Record]
    read_weight_path: Optional[str] = None
    write_weight_folder: Optional[str] = "/Disk3/siqi/PrimReg_outputs/MeshConvolution/train/weight_10/"
    write_tmp_folder: Optional[str] = "Disk3/siqi/PrimReg_outputs/MeshConvolution/train/tmp_10/"
    logdir: Optional[str] = "/Disk3/siqi/PrimReg_outputs/MeshConvolution/train/log_10/"



    # [Params] 
    lr: Optional[float] = 0.0001

    batch: int = 6

    w_pose:  int =1
    w_laplace:  int =0 

    augment_data: int = 0

    weight_decay: Optional[float] = 0.00000
    lr_decay: Optional[float] = 0.99
    lr_decay_epoch_step: int = 1


    start_epoch: int = 0
    epoch:  int =201
    evaluate_epoch: int = 2

    perpoint_bias: int = 0


    template_ply_fn: Optional[str] = "/Disk4/siqi/data/ellipsoids/template_primitives/ellipsoid-0.ply"


    point_num: int = 152

    pcs_train: Optional[str] =  "/Disk4/siqi/data/ellipsoids/vertices_npy/cat_13.npy"

    pcs_evaluate: Optional[str] =  "/Disk4/siqi/data/ellipsoids/vertices_npy/cat_13.npy"

    pcs_test: Optional[str] = "/Disk4/siqi/data/ellipsoids/vertices_npy/cat_13.npy"


    connection_folder: Optional[str] =  "/Disk4/siqi/data/ellipsoids/graphAE/ConnectionMatrices/"

    initial_connection_fn: Optional[str] =  "/Disk4/siqi/data/ellipsoids/graphAE/ConnectionMatrices/_pool0.npy"

    connection_layer_lst:  List = field(default_factory=lambda: ["pool0", "pool1",  "pool2", "pool3"] )

    ##residual only layer's channel number should be the same as the previous layer
    channel_lst: List = field(default_factory=lambda: [32, 64,  128,   9] )

    weight_num_lst: List = field(default_factory=lambda: [17,17,17, 17]  )


    ## 0 for conv only, 1 for residual only, 0.X for (1-0.X)*conv+0.X*res 
    residual_rate_lst: List = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])


