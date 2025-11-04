import json
import sys
from pathlib import Path
from torch import multiprocessing
import matplotlib.pyplot as plt
import random
import numpy as np
import tqdm
from functools import partial
from PIL import Image
import torch
import copy
import os
import cv2
from pathlib import Path
import os
import glob
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj

import argparse
sys.path.append("./submodules/external/Deep3DFaceRecon_pytorch")
from options.test_options import TestOptions
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from util.load_mats import load_lm3d
from models.facerecon_model import FaceReconModel
from models.bfm import ParametricFaceModel
from facenet_pytorch import MTCNN
import PIL


class Deep3DFaceReconstruction_Processor:
    def __init__(self, device, model_name="./submodules/external/Deep3DFaceRecon_pytorch/", epoch=20):
        
        bfm_folder = os.path.join(model_name, "BFM")
        ckpts_dir = os.path.join(model_name, "checkpoints/pretrained")
        

        opt = TestOptions(cmd_line=f"--name={model_name} --epoch={epoch} --bfm_folder={bfm_folder} --checkpoints_dir={ckpts_dir} --use_opengl=False").parse()
        
        model = FaceReconModel(opt)
        model.setup(opt)
        model.device = device
        model.parallelize()
        model.eval()
        detector = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=True,
        )
        lm3d_std = load_lm3d(opt.bfm_folder)
        
        face_model = ParametricFaceModel(bfm_folder=bfm_folder)

        self.opt = opt
        self.model = model
        self.detector = detector
        self.lm3d_std = lm3d_std
        self.face_model = face_model

    def get_head_coeffs(self, img, lmks=None):
        """
       assumes img to be np.ndarray of shape H x W x 3 with RGB entries normalized to 0 ... 255
       """

        lmks = lmks if lmks is not None else self.detect_lmks(img)
        if lmks is None:
            return None
        else:
            h_orig, w_orig = img.shape[:2]
            img_aligned, lmks_aligned, bbx_orig = self.align_img(img, lmks, return_bbx_orig=True)

            if img_aligned is None:
                return dict(id=None, exp=None, tex=None, angle=None, gamma=None, trans=None)
            img_aligned = torch.tensor(img_aligned / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # torch tensor 1 x 3 x H x W; rgb, 0...1
            img_orig = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # torch tensor 1 x 3 x H x W; rgb, 0...1
            lmks_aligned = torch.tensor(lmks_aligned).unsqueeze(0)
            input_data = dict(imgs=img_aligned, lms=lmks_aligned,
                              bbx_orig=bbx_orig, shape_orig=[h_orig, w_orig], imgs_orig=img_orig
                              )

            self.model.set_input(input_data)  # unpack data from data loader
            self.model.test()  # run inference


            head_coeffs = copy.deepcopy(self.model.pred_coeffs_dict)
            for k, v in head_coeffs.items():
                head_coeffs[k] = v.cpu().numpy()

            head_coeffs["bbx_orig"] = np.array(bbx_orig)
            head_coeffs["shape_orig"] = np.array([h_orig, w_orig])

            return head_coeffs

    def align_img(self, img, lmk, return_bbx_orig=False, *args, **kwargs):
        H, W = img.shape[:2]
        lmk = copy.deepcopy(lmk)
        lmk[:, -1] = H - 1 - lmk[:, -1]
        img = PIL.Image.fromarray(img)
        _, img, lmk, _, bbx_orig = align_img(img, lmk, self.lm3d_std, return_bbx_orig=True, *args, **kwargs)
        if img is not None:
            img = np.array(img)
        outs = [img, lmk]
        if return_bbx_orig:
            outs.append(bbx_orig)
        return outs

    def detect_lmks(self, img):
        boxes, probs, points = self.detector.detect(img, landmarks=True)
        if points is None:
            return None
        else:
            return points[0].astype(np.float32)

    def crop_img(self, im: np.ndarray, lm: np.ndarray, blur_pad=False, return_bbx_orig=False):
        im = Image.fromarray(im, "RGB")
        _, H = im.size
        lm = copy.deepcopy(lm)
        lm[:, -1] = H - 1 - lm[:, -1]

        target_size = 1024
        rescale_factor = 300
        center_crop_size = 700
        output_size = 512

        _, im_high, _, _, bbx_orig = align_img(im, lm, self.lm3d_std, target_size=target_size, rescale_factor=rescale_factor, blur_pad=blur_pad, return_bbx_orig=True)
        if im_high is None:
            return None

        left = int(im_high.size[0] / 2 - center_crop_size / 2)
        upper = int(im_high.size[1] / 2 - center_crop_size / 2)
        right = left + center_crop_size
        lower = upper + center_crop_size
        im_cropped = im_high.crop((left, upper, right, lower))
        bbx_orig = [
            bbx_orig[0] + left / target_size * (bbx_orig[2] - bbx_orig[0]),
            bbx_orig[1] + upper / target_size * (bbx_orig[3] - bbx_orig[1]),
            bbx_orig[0] + right / target_size * (bbx_orig[2] - bbx_orig[0]),
            bbx_orig[1] + lower / target_size * (bbx_orig[3] - bbx_orig[1])
        ]

        im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
        im_cropped = np.array(im_cropped)

        if return_bbx_orig:
            return im_cropped, bbx_orig
        else:
            return im_cropped

    def headcoeff2camparams(self, headcoeff):
        angle = headcoeff['angle']
        trans = headcoeff['trans'][0]
        R = self.face_model.compute_rotation(torch.from_numpy(angle))[0].numpy()
        trans[2] += -10
        c = -np.dot(R, trans)
        pose = np.eye(4)
        pose[:3, :3] = R

        c *= 0.27  # normalize camera radius
        c[1] += 0.006  # additional offset used in submission
        c[2] += 0.161  # additional offset used in submission
        pose[0, 3] = c[0]
        pose[1, 3] = c[1]
        pose[2, 3] = c[2]

        focal = 2985.29  # = 1015*1024/224*(300/466.285)#
        pp = 512  # 112
        w = 1024  # 224
        h = 1024  # 224

        count = 0
        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w / 2.0
        K[1][2] = h / 2.0

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        pose = self.fix_poxe_orig(pose)
        K = self.fix_intrinsics(K)

        out = {}
        out["intrinsics"] = K
        out["pose"] = pose

        return out

    def fix_poxe_orig(self, pose: np.ndarray):
        # taken from  eg3d/dataset_preprocessing/ffhq/preprocess_face_cameras.py:fix_pose_orig()
        pose = np.array(pose).copy()
        location = pose[:3, 3]
        radius = np.linalg.norm(location)
        pose[:3, 3] = pose[:3, 3] / radius * 2.7
        return pose

    def fix_intrinsics(self, intrinsics):
        intrinsics = np.array(intrinsics).copy()
        assert intrinsics.shape == (3, 3), intrinsics
        intrinsics[0, 0] = 2985.29 / 700
        intrinsics[1, 1] = 2985.29 / 700
        intrinsics[0, 2] = 1 / 2
        intrinsics[1, 2] = 1 / 2
        assert intrinsics[0, 1] == 0
        assert intrinsics[2, 2] == 1
        assert intrinsics[1, 0] == 0
        assert intrinsics[2, 0] == 0
        assert intrinsics[2, 1] == 0
        return intrinsics



def annotate_dir(deep3dface, img_dir: Path, out_kp_dir: Path, out_bfm_dir: Path, out_normal_dir: Path, out_mesh_dir: Path, out_cam_dir: Path):
    out_kp_dir.mkdir(exist_ok=True, parents=True)
    out_bfm_dir.mkdir(exist_ok=True, parents=True)
    out_mesh_dir.mkdir(exist_ok=True, parents=True)
    out_normal_dir.mkdir(exist_ok=True, parents=True)
    out_cam_dir.mkdir(exist_ok=True, parents=True)
    
    
    in_hair_files = sorted([f for f in img_dir.iterdir() if f.name.lower().endswith(".png") or f.name.lower().endswith(".jpg")])
        
    for img_path in tqdm.tqdm(in_hair_files, leave=False):

        img = Image.open(img_path).convert("RGB")

        img = np.array(img)
        lmks = deep3dface.detect_lmks(img)

        if lmks is None:
            print(f"ERROR WITH FILE {img_path}")
            head_coeffs = dict(id=None, exp=None, tex=None, angle=None, gamma=None, trans=None)
            lmks_68 = np.array([])
            P = np.array([])
        else:
            head_coeffs = deep3dface.get_head_coeffs(img, lmks)
            K = head_coeffs["facemodel_perc_proj"].T  # assuming x:right, y:down, z: view direction, image origin: top left; matches exported mesh
            K[1, 2] = head_coeffs["rasterize_size"][0] - K[1, 2]
            head_coeffs["K"] = K
            P = np.eye(4)[:3]
            P[:3, :3] = K
            head_coeffs["P"] = P
            lmks_68 = deep3dface.model.pred_lm[0].cpu().numpy() / np.array([[img.shape[1], img.shape[0]]])
            lmks_68[:, 1] = 1 - lmks_68[:, 1]

            for k, v in head_coeffs.items():
                head_coeffs[k] = v.tolist()

        out_kp_file = out_kp_dir / (img_path.name.split(".")[0] + ".txt")
        out_bfm_file = out_bfm_dir / (img_path.name.split(".")[0] + ".json")
        out_normal_file = out_normal_dir / (img_path.name.split(".")[0] + ".jpg")
        out_mesh_file = out_mesh_dir / (img_path.name.split(".")[0] + ".obj")
        out_cam_file = out_cam_dir / (img_path.name.split(".")[0] + ".txt")

        pred_normal = (deep3dface.model.pred_normal * .5 + .5) * deep3dface.model.pred_mask
        pred_normal = np.clip(np.round(255. * pred_normal.detach().cpu().permute(0, 2, 3, 1).numpy()[0]), a_min=0, a_max=255).astype(np.uint8)

        np.savetxt(out_kp_file, np.array(lmks_68))

        with open(out_bfm_file, "w") as f:
            json.dump(head_coeffs, f, indent="\t")

        Image.fromarray(pred_normal).save(out_normal_file)

        # saving mesh
        pred_verts = deep3dface.model.pred_vertex[0]

        pred_verts[:, 1] *= -1  # converts camera convention from left-handed system used by eg3d/deep3dfacerecon (camera coordinates: x: right; y: top, z: in view direction)
        # to right-handed coordinate system: (camera coordinates: x: right, y:down, z: in view direction); matches 'K' in exported bfm parameters
        faces = deep3dface.model.facemodel.face_buf
        save_obj(out_mesh_file, pred_verts, faces)
        np.savetxt(out_cam_file, P)


def main(args, device='cuda'):
    
    deep3dface = Deep3DFaceReconstruction_Processor(device)


    all_img_dirs = [
      args.root_path
    ]

    all_img_dirs = [Path(p) for p in all_img_dirs]
    all_idcs = np.arange(len(all_img_dirs))

    njobs = int(os.getenv("CONDOR_WorldSize", 1))
    jobid = int(os.getenv("CONDOR_Process", 0))

    my_idcs = np.array_split(all_idcs, njobs)[jobid]
    for i in tqdm.tqdm(my_idcs, leave=True):
        print('start processing!')
        img_dir = all_img_dirs[i]

        out_kp_dir = img_dir.parent / f"kps{args.save_postfix}"
        out_bfm_dir = img_dir.parent / f"bfm{args.save_postfix}"
        out_normal_dir = img_dir.parent / f"bfm_normals{args.save_postfix}"
        out_mesh_dir = img_dir.parent / f"bfm_meshes{args.save_postfix}"
        out_cam_dir = img_dir.parent / f"bfm_cameras{args.save_postfix}"


        annotate_dir(deep3dface, img_dir, out_kp_dir, out_bfm_dir, out_normal_dir, out_mesh_dir, out_cam_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--root_path', default= './resized_img', type=str)
    parser.add_argument('--save_postfix', default= '', type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)          
        