import os
import open3d as o3d
import numpy as np
import torch


class RegBenchmark(torch.utils.data.Dataset):
    """
    The .pcd files are already aligned, the provided init_trans matrix representing the initial misplacement to apply.
    """
    def __init__(self, root, gt_path, regtype='local', voxel_size=0.1, select_dataset=None):
        self.root = root
        self.gt_path = gt_path
        self.type = regtype
        self.voxel_size = voxel_size
        self.dataset2scene = {
            'eth': ['apartment', 'gazebo_summer', 'gazebo_winter', 'hauptgebaude', 'plain', 'stairs', 'wood_autumn', 'wood_summer'],
            'kaist': ['urban05'],
            # 'planetary': ['box_met', 'p2at_met', 'planetary_map'],
            'tum': ['long_office_household', 'pioneer_slam', 'pioneer_slam3'], 
        }

        self.pairs = []
        for scene_filename in os.listdir(self.gt_path):
            scene = scene_filename.replace('_global.txt', '').replace('_local.txt', '')
            dataset = self._which_dataset(scene)
            if dataset == -1:
                continue
            if select_dataset is not None and dataset != select_dataset:
                continue
            if self.type in scene_filename:
                self.pairs += self._loadtxt(os.path.join(self.gt_path, scene_filename), dataset, scene)

    def __getitem__(self, index):
        pair_dict = self.pairs[index]
        scene_path = os.path.join(self.root, pair_dict['dataset'], pair_dict['scene'])
        source_pcd = o3d.io.read_point_cloud(os.path.join(scene_path, pair_dict['source']))
        target_pcd = o3d.io.read_point_cloud(os.path.join(scene_path, pair_dict['target']))
        if self.voxel_size > 0:
            source_pcd = source_pcd.voxel_down_sample(self.voxel_size)
            target_pcd = target_pcd.voxel_down_sample(self.voxel_size)
        target_pcd.transform(pair_dict['init_trans'])

        source_keypts = np.array(source_pcd.points)
        target_keypts = np.array(target_pcd.points)
        return source_keypts, target_keypts, pair_dict['init_trans'], pair_dict['dataset'], pair_dict['scene'], pair_dict['id']

    def __len__(self):
        return len(self.pairs)

    def _loadtxt(self, filepath, dataset, scene):
        with open(filepath, 'r') as f:
            context = f.readlines()
        pairs = []
        for line in context[1:]:
            pair_info = line.replace('\n', '').split(' ')
            init_trans = np.array([
                [float(pair_info[4]), float(pair_info[5]), float(pair_info[6]), float(pair_info[7])],
                [float(pair_info[8]), float(pair_info[9]), float(pair_info[10]), float(pair_info[11])],
                [float(pair_info[12]), float(pair_info[13]), float(pair_info[14]), float(pair_info[15])],
                [0,0,0,1]
            ])
            pair_dict = {
                'id': pair_info[0],
                'dataset': dataset,
                'scene': scene,
                'source': pair_info[1],
                'target': pair_info[2],
                'overlap': pair_info[3],
                'init_trans': init_trans,
            }
            pairs.append(pair_dict)
        return pairs

    def _which_dataset(self, scene):
        for dataset, scene_list in self.dataset2scene.items():
            if scene in scene_list:
                return dataset
        # print(f"No such scene: {scene}")
        return -1


if __name__ == '__main__':
    dataset = RegBenchmark(
        root='data',
        gt_path='devel/registration_pairs',
        regtype='global',
        voxel_size=0.1,
        select_dataset='tum'
    )
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
    )
    
    length_list = []
    from tqdm import tqdm
    for src_keypts, _, _, _, _, _  in tqdm(dataloader):
        length_list += [src_keypts.shape[1]]
    print(min(length_list), max(length_list), np.mean(length_list))

# scene    min max mean 
# kaist:  6301 18207 10243.52
# tum  :  147  5130  1524.45
# eth  :  3966 46369 22744.1325