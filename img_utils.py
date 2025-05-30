import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    src_w = src_width * scale
    src_h = src_height * scale
    rot_rad = np.pi * rot / 180

    def rotate_2d(pt_2d, rot_rad):
        x, y = pt_2d
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)

    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = rotate_2d([0, src_h * 0.5], rot_rad)
    src_rightdir = rotate_2d([src_w * 0.5, 0], rot_rad)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.stack([src_center, src_center + src_downdir, src_center + src_rightdir])
    dst = np.stack([dst_center, dst_center + dst_downdir, dst_center + dst_rightdir])

    return cv2.getAffineTransform(dst if inv else src, src if inv else dst)

def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img.shape[1] - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)
    return img_patch, trans

def get_single_image_crop(image, bbox, scale=1.3):
    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")

    crop_img, _ = generate_patch_image_cv(cvimg=image.copy(), c_x=bbox[0], c_y=bbox[1], bb_width=bbox[2],bb_height=bbox[3], patch_width=224, patch_height=224,  do_flip=False, scale=scale, rot=0
    )
    return convert_cvimg_to_tensor(crop_img)

def convert_cvimg_to_tensor(image):
    transform = get_default_transform()
    return transform(image)

def get_default_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([transforms.ToTensor(), normalize
    ])

def get_single_image_crop_demo(image, bbox, kp_2d, scale=1.2, crop_size=224):
    import cv2
    import numpy as np

    if isinstance(image, str):
        if os.path.isfile(image):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"{image} is not a valid file path.")
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")

    crop_image, trans = generate_patch_image_cv(
        cvimg=image.copy(), c_x=bbox[0], c_y=bbox[1], bb_width=bbox[2], bb_height=bbox[3],  patch_width=crop_size, patch_height=crop_size, do_flip=False, scale=scale, rot=0,
    )

    if kp_2d is not None:
        for n_jt in range(kp_2d.shape[0]):
            kp_2d[n_jt, :2] = trans_point2d(kp_2d[n_jt], trans)

    raw_image = crop_image.copy()
    crop_image = convert_cvimg_to_tensor(crop_image)

    return crop_image, raw_image, kp_2d

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


import os
import cv2
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from img_utils import get_single_image_crop_demo


class Inference(Dataset):
    def __init__(self, image_folder, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)
        self.image_file_names = np.array(self.image_file_names)[frames]
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.scale = scale
        self.crop_size = crop_size
        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False

        self.norm_joints2d = np.zeros_like(self.joints2d)

        if self.has_keypoints:
            bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
            bboxes[:, 2:] = 150. / bboxes[:, 2:]
            self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

            self.image_file_names = self.image_file_names[time_pt1:time_pt2]
            self.joints2d = joints2d[time_pt1:time_pt2]
            self.frames = frames[time_pt1:time_pt2]

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)

        bbox = self.bboxes[idx]

        j2d = self.joints2d[idx] if self.has_keypoints else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=self.scale,
            crop_size=self.crop_size)
        if self.has_keypoints:
            return norm_img, kp_2d
        else:
            return norm_img

