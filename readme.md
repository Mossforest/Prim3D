# process

## checklist

- [ ] net
- [ ] loss obj
- [ ] train & test process
- [ ] visualize

## env package

```bash
pip:
    - learnable-primitives [straightly_import]
    - lapsolver [clone_not_possible]
```

## dataset
### SQ_templates
- `joint_info.mat`
  ```python
  '__header__': b'MATLAB 5.0 MAT-file Platform: posix, Created on: Sun Jul 30 22:50:25 2023', 
  '__version__': '1.0', 
  '__globals__': [], 
  'class_name': array(['microwave_1'], dtype='<U11'), 
  'joint_tree': array([[0, 0]]), 
  'primitive_align': array([[1, 0]]), 
  'joint_parameter_leaf': array([[ 0.0000000e+00,  1.0000000e+00, -2.4492937e-16],
       [ 0.0000000e+00, -1.0000000e+00,  2.4492937e-16],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00]], dtype=float32)
  ```
- `part_centers.npy`
  ```python
  array([[-0.057544  ,  0.063503  , -0.5951255 ],
       [ 0.75385904, -0.010965  ,  0.4174035 ]], dtype=float32)
  ```
- `plys/delta_rots.npy`, `plys/pred_rots.npy`
  ```python
  np.array, shape = (1, 2, 3, 3), range = [-1, 1]
  ```
- `plys/pred_sq.npy`
  ```python
  array([[[0.70757157, 0.41402268, 0.01959559, 0.08339135, 0.34984398],
        [0.8210363 , 0.46854043, 0.4449706 , 0.11471916, 0.26079988]]],
      dtype=float32)
  np.array, shape = (1, 2, 5), range = [0, 1]
  ```
- `plys/SQ_ply`: 多边形模型，包含顶点和面（可能含有渲染信息），`open3d`载入
