import os
import glob
import time
import numpy as np
import torch
import argparse
import open3d as o3d
from copy import deepcopy
from tqdm import tqdm
import MinkowskiEngine as ME
from metric.metric import calculate_error, eval_features, eval_trans
from baselines.fcgf import ResUNetBN2C as FCGF
from baselines.dataset import RegBenchmark
from baselines.reg_baseline import ransac_registration, make_o3d_pointcloud, make_o3d_features


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
    """
    Extracts FCGF features.
    Args:
        model (FCGF model instance): model used to inferr the features
        xyz (torch tensor): coordinates of the point clouds [N,3]
        rgb (torch tensor): colors, must be in range (0,1) [N,3]
        normal (torch tensor): normal vectors, must be in range (-1,1) [N,3]
        voxel_size (float): voxel size for the generation of the saprase tensor
        device (torch device): which device to use, cuda or cpu
        skip_check (bool): if true skip rigorous check (to speed up)
        is_eval (bool): flag for evaluation mode
    Returns:
        return_coords (torch tensor): return coordinates of the points after the voxelization [m,3] (m<=n)
        features (torch tensor): per point FCGF features [m,c]
    """

    if is_eval:
        model.eval()

    if not skip_check:
        assert xyz.shape[1] == 3

        N = xyz.shape[0]
        if rgb is not None:
            assert N == len(rgb)
            assert rgb.shape[1] == 3
            if np.any(rgb > 1):
                raise ValueError('Invalid color. Color must range from [0, 1]')

        if normal is not None:
            assert N == len(normal)
            assert normal.shape[1] == 3
            if np.any(normal > 1):
                raise ValueError('Invalid normal. Normal must range from [-1, 1]')

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats = []
    if rgb is not None:
        # [0, 1]
        feats.append(rgb - 0.5)

    if normal is not None:
        # [-1, 1]
        feats.append(normal / 2)

    if rgb is None and normal is None:
        feats.append(np.ones((len(xyz), 1)))

    feats = np.hstack(feats)

    # Voxelize xyz and feats
    coords = np.floor(xyz / voxel_size)
    _, inds = ME.utils.sparse_quantize(coords, return_index=True)
    
    coords = coords[inds]
    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = xyz[inds]

    feats = feats[inds]

    feats = torch.tensor(feats, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.int32)

    stensor = ME.SparseTensor(feats.to(device), coordinates=coords.to(device))

    return return_coords, model(stensor).F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default='3dmatch', type=str, choices=['3dmatch', 'kitti'])
    parser.add_argument('--dataset', default='all', type=str)
    parser.add_argument('--voxel_size', default=0.1, type=float)
    args = parser.parse_args()

    if args.pretrain == '3dmatch':
        model = FCGF(
            1,
            32,
            bn_momentum=0.05,
            conv1_kernel_size=7,
            normalize_feature=True
        ).cuda()
        checkpoint = torch.load("weights/ResUNetBN2C-feat32-3dmatch-v0.05.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    else:
        model = FCGF(
            1,
            32,
            bn_momentum=0.05,
            conv1_kernel_size=5,
            normalize_feature=True
        ).cuda()
        checkpoint = torch.load("weights/ResUNetBN2C-feat32-kitti-v0.3.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

    dataset = RegBenchmark(
        root='data',
        gt_path='devel/registration_pairs',
        regtype='global',
        voxel_size=-1,
        select_dataset=args.dataset,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
    )
    print(f"Total {len(dataset)} pairs")
    res_list = []
    with torch.no_grad():
        for (source_keypts, target_keypts, gt_trans, _dset, _scene, _id) in tqdm(dataloader):
            # source_keypts, target_keypts, gt_trans = source_keypts.cuda(), target_keypts.cuda(), gt_trans.cuda()

            s_time = time.time()
            source_down, source_features = extract_features(model, xyz=source_keypts[0], rgb=None, normal=None, voxel_size=args.voxel_size, skip_check=True)
            target_down, target_features = extract_features(model, xyz=target_keypts[0], rgb=None, normal=None, voxel_size=args.voxel_size, skip_check=True)
            source_features_o3d = make_o3d_features(source_features, 32, source_features.shape[-2])
            target_features_o3d = make_o3d_features(target_features, 32, target_features.shape[-2])
            e_time = time.time()
            feat_time = e_time - s_time


            #################################
            # Registration using local features
            #################################
            source_pcd = make_o3d_pointcloud(source_down)
            target_pcd = make_o3d_pointcloud(target_down)

            s_time = time.time()
            result = ransac_registration(source_pcd, target_pcd, source_features_o3d, target_features_o3d, args.voxel_size)
            pred_trans = result.transformation
            valid = (result.fitness != 0)
            inlier_num_after = len(result.correspondence_set)

            e_time = time.time()
            reg_time = e_time - s_time 

            #################################
            # Evaluate local features 
            #################################
            inlier_ratio, inlier_num = eval_features(source_keypts[0], target_keypts[0], source_features, target_features, gt_trans[0], args.voxel_size)

            #################################
            # Evaluate predicted transformations
            #################################
            if valid:
                aligned_source_pcd = deepcopy(source_pcd)
                aligned_source_pcd.transform(pred_trans)
                source_pcd.transform(gt_trans[0].detach().cpu().numpy())
                error = calculate_error(aligned_source_pcd, source_pcd)
                re, te = eval_trans(gt_trans[0], torch.from_numpy(pred_trans).to(gt_trans.device))
            else:
                error = -1
                re = -1 
                te = -1 

            #################################
            # Save statistics
            #################################
            # dataset, scene, id, #inlier ratio(no mutual), #corr(before ransac), #corr(after ransac), re, te, error, feat_time, reg_time
            res_list += [ [_dset[0], _scene[0], _id[0], inlier_ratio, inlier_num, inlier_num_after, re, te, float(error), feat_time, reg_time] ]
            print(res_list[-1])
    res_list = np.array(res_list)
    if args.dataset is None:
        args.dataset = 'all'
    np.save(f'results/fcgf_{args.pretrain}_{args.dataset}.npy', res_list)