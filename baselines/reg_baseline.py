import open3d as o3d
import numpy as np
import torch
import time
import argparse
from tqdm import tqdm
from copy import deepcopy
from baselines.dataset import RegBenchmark
from metric.metric import calculate_error, eval_features, eval_trans

def make_o3d_pointcloud(pts):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd 

def make_o3d_features(data, dim, npts):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature

def ransac_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, 
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], 
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(confidence=0.999, max_iteration=100000)
    )
    return result

def fgr_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold)
    )
    return result

def icp_registration(source_pcd, target_pcd, voxel_size, initial_trans=None):
    if initial_trans is None:
        initial_trans = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, voxel_size, initial_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', type=str)
    parser.add_argument('--voxel_size', default=0.1, type=float)
    args = parser.parse_args()

    dataset = RegBenchmark(
        root='data',
        gt_path='devel/registration_pairs',
        regtype='global',
        voxel_size=args.voxel_size,
        select_dataset=args.dataset
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

            source_pcd = make_o3d_pointcloud(source_keypts[0])
            target_pcd = make_o3d_pointcloud(target_keypts[0])
            s_time = time.time()
            source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2, max_nn=30))
            target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2, max_nn=30))
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*10, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*10, max_nn=100))
            e_time = time.time()
            feat_time = e_time - s_time

            #################################
            # Registration using local features
            #################################
            s_time = time.time()
            result = ransac_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, args.voxel_size)
            pred_trans = result.transformation
            valid = (result.fitness != 0)
            inlier_num_after = len(result.correspondence_set)

            e_time = time.time()
            reg_time = e_time - s_time

            #################################
            # Evaluate local features 
            #################################
            source_features = np.array(source_fpfh.data).T
            source_features = source_features / (np.linalg.norm(source_features, axis=1, keepdims=True) + 1e-6)
            target_features = np.array(target_fpfh.data).T
            target_features = target_features / (np.linalg.norm(target_features, axis=1, keepdims=True) + 1e-6)
            source_features = torch.from_numpy(source_features).to(source_keypts.device)
            target_features = torch.from_numpy(target_features).to(source_keypts.device)
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
    np.save(f'results/fpfh_10_{args.dataset}.npy', res_list)