import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import random
import clip


from .ray_utils import *

clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")


class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0,6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # random sample a subset of images
        subset_len = 30
        self.meta['frames'] = random.sample(self.meta['frames'], subset_len)
        #self.meta['frames'] = self.meta['frames'][:subset_len]


        # ray directions for all pixels, same for all images (same H, W, focal) # TODO: why is this the same for all images?
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample=1.0
        self.all_img_features = []

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]

            # Read transfomation matrix, convert to opencv -> torch, then add to the list of poses
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            # load image path into openCV -> downsample -> convert to torch -> convert to RGB format -> add to list of images
            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            img_ft = clip_preprocess(img).unsqueeze(0).to('cuda')
            with torch.no_grad():
                image_features = clip_model.encode_image(img_ft)
                self.all_img_features += [image_features]
                        
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w) # (4, 800, 800)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB (h*w, 3)
            self.all_rgbs += [img]

            # use pose matrix to get ray origins and directions #TODO: is this correct? What is happening in this step exactly?
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6) #TODO: so we have 1 ray, source to destination, for each pixel in the image?


        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample
