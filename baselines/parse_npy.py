import numpy as np
import argparse

def print_results(npyfile, re_thre=2, te_thre=10):
    scene_list = set(npyfile[:, 1])
    # dataset, scene, id, #inlier ratio(no mutual), #corr(before ransac), #corr(after ransac), re, te, error, time(s)
    print(f"{'SceneName'.center(24, ' ')} Valid Success Inlier% Inlier(#) Re(deg) Te(cm) Median  Mean  STD  FeatTime RegTime")
    for scene in scene_list:
        scene_stats = npyfile[npyfile[:, 1] == scene]
        error_array = scene_stats[:, -2].astype(np.float)
        valid = np.where(error_array != -1)
        valid_ratio = np.sum(error_array != -1) / scene_stats.shape[0] * 100
        success = np.where( (error_array != -1) * (scene_stats[:, 6].astype(np.float) < re_thre) * (scene_stats[:, 7].astype(np.float) < te_thre))
        success_ratio = success[0].__len__() / scene_stats.shape[0] * 100
        
        print(f"{scene.center(24, ' ')} "
            f"{valid_ratio:3.0f}% "
            f"{success_ratio:5.0f}% "
            f"{np.mean(scene_stats[:, 3].astype(np.float))*100:7.1f}% "
            f"{np.mean(scene_stats[:, 4].astype(np.float)):7.1f} "
            f"{np.mean(scene_stats[:, 6].astype(np.float)):7.2f} "
            f"{np.mean(scene_stats[:, 7].astype(np.float)):7.2f} "
            f"{np.median(error_array[valid]):6.2f} "
            f"{np.mean(error_array[valid]):6.2f} "
            f"{np.std(error_array[valid]):5.2f} "
            f"{np.mean(scene_stats[:, -2].astype(np.float)):7.2f} "
            f"{np.mean(scene_stats[:, -1].astype(np.float)):7.2f} "
        )
    # if scene != 'all':
        # return
    # print('-'*50)
    scene_stats = npyfile
    error_array = scene_stats[:, -2].astype(np.float)
    valid = np.where(error_array != -1)
    valid_ratio = np.sum(error_array != -1) / error_array.shape[0] * 100
    success = np.where( (error_array != -1) * (scene_stats[:, 6].astype(np.float) < re_thre) * (scene_stats[:, 7].astype(np.float) < te_thre))
    success_ratio = success[0].__len__() / scene_stats.shape[0] * 100
    print(f"{'Average'.center(24, ' ')} "
        f"{valid_ratio:03.0f}% "
        f"{success_ratio:5.0f}% "
        f"{np.mean(scene_stats[:, 3].astype(np.float))*100:7.1f}% "
        f"{np.mean(scene_stats[:, 4].astype(np.float)):7.1f} "
        f"{np.mean(scene_stats[:, 6].astype(np.float)):7.2f} "
        f"{np.mean(scene_stats[:, 7].astype(np.float)):7.2f} "
        f"{np.median(error_array[valid]):6.2f} "
        f"{np.mean(error_array[valid]):6.2f} "
        f"{np.std(error_array[valid]):5.2f} "
        f"{np.mean(scene_stats[:, -2].astype(np.float)):7.2f} "
        f"{np.mean(scene_stats[:, -1].astype(np.float)):7.2f} "
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptor', default='fpfh', type=str)
    parser.add_argument('--dataset', default='all', type=str)
    parser.add_argument('--re_thre', default=2, type=float)
    parser.add_argument('--te_thre', default=15, type=float)
    args = parser.parse_args()

    print("*" * 30, f"{args.descriptor} + RANSAC", "*" * 30)
    print_results(np.load(f'results/{args.descriptor}_{args.dataset}.npy'), args.re_thre, args.te_thre)

