import numpy as np
import open3d as o3d
from utils import *

def calculate_error(cloud1: o3d.geometry.PointCloud, cloud2: o3d.geometry.PointCloud) -> float:
    assert len(cloud1.points) == len(cloud2.points), "len(cloud1.points) != len(cloud2.points)"
    
    centroid, _ = cloud1.compute_mean_and_covariance()
    weights = np.linalg.norm(np.asarray(cloud1.points) - centroid, 2, axis=1)
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1)/len(weights)
    return np.sum(distances/weights)

def build_nn(source_desc, target_desc):
    """
    Find the closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 32]
    """
    device = source_desc.device
    if source_desc.shape[0] > 30000:
        source_desc = source_desc.detach().cpu().numpy()
        target_desc = target_desc.detach().cpu().numpy()
        distance = source_desc @ target_desc.T
        source_idx = np.argmax(distance, axis=1)
        corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
        corr = torch.from_numpy(corr).to(device)
        return corr
    distance = source_desc @ target_desc.T
    source_idx = torch.argmax(distance, dim=-1).to(device)
    corr = torch.cat([torch.arange(source_idx.shape[0])[:, None].to(device), source_idx[:, None]], dim=-1)
    return corr

def eval_features(source_keypts, target_keypts, source_features, target_features, gt_trans, voxel_size):
    corr = build_nn(source_features, target_features)
    frag1 = source_keypts[corr[:, 0]]
    frag2 = target_keypts[corr[:, 1]]
    frag1_warp = transform(frag1, gt_trans)
    distance = torch.norm(frag1_warp - frag2, dim=-1)
    # distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
    labels = (distance < voxel_size * 1.5).int()
    inlier_ratio = labels.float().mean()
    inlier_num = labels.sum()
    return float(inlier_ratio), int(inlier_num)

def eval_trans(gt_trans, pred_trans):
    R, t = decompose_trans(pred_trans)
    gt_R, gt_t  = decompose_trans(gt_trans)
    re = torch.acos(torch.clamp((torch.trace(R.T @ gt_R) - 1) / 2.0, min=-1, max=1))
    te = torch.sqrt(torch.sum((t - gt_t) ** 2))
    # re = np.arccos((np.trace(R.T @ gt_R) - 1) / 2.0)
    # te = np.sqrt(np.sum((t - gt_t) ** 2))
    re * 180 / np.pi 
    te = te * 100
    return float(re), float(te)