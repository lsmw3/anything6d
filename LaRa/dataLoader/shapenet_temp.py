import os
from PIL import Image
import numpy as np
from glob import glob
import json
import _pickle as cPickle
import cv2
import random
import torch
from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R

import h5py

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

def ixt_to_fov(ixt):
    reso_x = 2 * ixt[0, 2]
    reso_y = 2 * ixt[1, 2]

    focal_x = ixt[0, 0]
    focal_y = ixt[1, 1]

    fov_x = 2 * np.arctan(reso_x / (2 * focal_x))
    fov_y = 2 * np.arctan(reso_y / (2 * focal_y))

    return fov_x, fov_y

class shapenet_template(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(shapenet_template, self).__init__()
        self.cfg = cfg
        self.data_root = cfg.data_root
        self.split = cfg.split
        self.img_size = np.array(cfg.img_size)
        
        instances = glob(os.path.join(self.data_root,'*'))
        
        self.instances = []
        for instance in instances:
            instance_dict = {}
            
            instance_rgb = glob(os.path.join(instance,'rgb', '*.png'))
            instance_rgb.sort()
            
            with open(os.path.join(instance,'scene_gt.json'), 'r') as f:
                instance_exts = json.load(f)
            with open(os.path.join(instance,'scene_camera.json'), 'r') as f:
                instance_ixts = json.load(f)
                
            for i in range(len(instance_rgb)):
                R, T, ixt = np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32), np.eye(3, dtype=np.float32)
                for j in range(3):
                    R[j] = instance_exts[f'{i}'][0]['cam_R_m2c'][j*3:j*3+3]
                    ixt[j] = instance_ixts[f'{i}']['cam_K'][j*3:j*3+3]
                T[:] = instance_exts[f'{i}'][0]['cam_t_m2c']
                
                ext = np.eye(4, dtype=np.float32)
                ext[:3, :3] = R.T
                ext[3, :3] = -R @ T
                
                instance_dict.update(
                    {
                        f'{i}': {
                            'rgb': os.path.join(instance,'rgb', f'{str(i).zfill(6)}.png'),
                            'ext': ext.T.astype(np.float32),
                            'ixt': ixt.astype(np.float32)
                        }
                    }
                )
                
            self.instances.append(instance_dict) # List[Dict[Dict]]
        
        self.n_group = cfg.n_group # 4
        # self.norm_scale = 1000.0
        # self.n_scenes = cfg.n_scenes # 240
        
    def _read_data(self, scene, image_idxs_per_scene: list):
        bg_colors = []
        ixts, exts, w2cs, imgs = [], [], [], []
        for i, image_idx in enumerate(image_idxs_per_scene):
            img_path = scene[f'{image_idx}']['rgb']
            
            # rgb
            # img = cv2.imread(img_path)[:, :, :3]
            # h, w, _ = img.shape
            # H, W = self.img_size
            # start_x = (w - W) // 2
            # start_y = (h - H) // 2
            # cropped_img = img[start_y:start_y + H, start_x:start_x + W]
            # rgb_ = cropped_img[:, :, ::-1] # (416, 416, 3)
            rgb_ = cv2.imread(img_path)[:, :, :3]  # 420x420
            rgb_ = rgb_[:, :, ::-1]
            mask = np.where(np.array(rgb_) < 255, 1, 0)
            
            if self.split != 'train' or i < self.n_group:
                bg_color = np.ones(3).astype(np.float32)
            else:
                bg_color = np.ones(3).astype(np.float32)*random.uniform(0.0, 1.0)
            
            rgb_ = np.array(rgb_).astype(np.float32) / 255.
            rgb_ = (rgb_ * mask + (1 - mask) * bg_color).astype(np.float32)
            
            # TODO modify the ext, pytorch3d -> colmap
            ext = scene[f'{image_idx}']['ext']
            cam_align = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            ext = ext @ cam_align
            # ext = cam_align @ ext
            
            ixt = scene[f'{image_idx}']['ixt']
            # ixt[0, 2] = W // 2
            # ixt[1, 2] = H // 2
            
            w2c = np.linalg.inv(ext).astype(np.float32)
            
            bg_colors.append(bg_color), ixts.append(ixt.astype(np.float32)), exts.append(ext.astype(np.float32)), w2cs.append(w2c), imgs.append(rgb_)
            
        return np.stack(bg_colors), np.stack(ixts), np.stack(exts), np.stack(w2cs), np.stack(imgs)
    
    def get_input(self, scene, img_ids_per_scene):
        bg_colors, tar_ixts, tar_c2ws, tar_w2cs, tar_img = self._read_data(scene, img_ids_per_scene)
        fovx, fovy = ixt_to_fov(tar_ixts[0])

        # align cameras using first view
        # no inverse operation 
        r = np.linalg.norm(tar_c2ws[0,:3,3])
        ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        transform_mats = ref_c2w @ tar_w2cs[:1]
        # tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        # tar_c2ws = transform_mats @ tar_c2ws.copy()
 
        ret = {'fovx':fovx,
               'fovy':fovy,
               }
        H, W = self.img_size

        ret.update({'tar_c2w': tar_c2ws,
                    'tar_w2c': tar_w2cs,
                    'tar_ixt': tar_ixts,
                    'tar_rgb': tar_img,
                    'transform_mats': transform_mats,
                    'bg_color': bg_colors
                    })
        
        # if self.cfg.load_normal:
        #     tar_nrms = tar_nrms @ transform_mats[0,:3,:3].T
        #     ret.update({'tar_nrm': tar_nrms.transpose(1,0,2,3).reshape(H,len(view_id)*W,3)})
        
        near_far = np.array([r-0.8, r+0.8]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'tar_view': 0, 'frame_id': 0}})
        ret['meta'].update({f'tar_h': int(H), f'tar_w': int(W)})

        rays = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        scene = self.instances[index]
        view_ids = range(len(scene))
        selected_view_id = [12, 125, 131, 8, 14, 141, 135, 18]

        if self.split=='train':
            src_view_id = random.sample(selected_view_id, k=self.n_group)
            view_id = src_view_id + random.sample(list(set(view_ids[self.n_group:-self.n_group]) - set(src_view_id)), k=self.n_group)
            # src_view_id = [12, 131, 14, 125]
            # view_id = src_view_id + [56]
        else:
            src_view_id = list(view_ids[:self.n_group])
            view_id = src_view_id + list(view_ids[-self.n_group:])
        
        ret = self.get_input(scene, view_id)
                
        return ret