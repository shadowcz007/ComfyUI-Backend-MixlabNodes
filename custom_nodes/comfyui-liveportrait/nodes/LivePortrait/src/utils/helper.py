# coding: utf-8

"""
utility functions and classes to handle feature extraction and model loading
"""

import os
import os.path as osp
import cv2
import torch
from rich.console import Console
from collections import OrderedDict

from LivePortrait.src.modules.spade_generator import SPADEDecoder
from LivePortrait.src.modules.warping_network import WarpingNetwork
from LivePortrait.src.modules.motion_extractor import MotionExtractor
from LivePortrait.src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from LivePortrait.src.modules.stitching_retargeting_network import StitchingRetargetingNetwork
from .rprint import rlog as log


def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(osp.basename(filename))


def is_video(file_path):
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or osp.isdir(file_path):
        return True
    return False

def is_template(file_path):
    if file_path.endswith(".pkl"):
        return True
    return False


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            print(f"Make dir: {d}")
    return d


def squeeze_tensor_to_numpy(tensor):
    out = tensor.data.squeeze(0).cpu().numpy()
    return out


def dct2cuda(dct: dict, device_id: int):
    for key in dct:
        dct[key] = torch.tensor(dct[key]).cuda(device_id)
    return dct


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


def remove_ddp_dumplicate_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new


def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params).cuda(device)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params).cuda(device)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params).cuda(device)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params).cuda(device)
    elif model_type == 'stitching_retargeting_module':
        # Special handling for stitching and retargeting module
        config = model_config['model_params']['stitching_retargeting_module_params']
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
        stitcher = stitcher.cuda(device)
        stitcher.eval()

        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_mouth']))
        retargetor_lip = retargetor_lip.cuda(device)
        retargetor_lip.eval()

        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_eye']))
        retargetor_eye = retargetor_eye.cuda(device)
        retargetor_eye.eval()

        return {
            'stitching': stitcher,
            'lip': retargetor_lip,
            'eye': retargetor_eye
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    model.eval()
    return model


# get coefficients of Eqn. 7
def calculate_transformation(config, s_kp_info, t_0_kp_info, t_i_kp_info, R_s, R_t_0, R_t_i):
    if config.relative:
        new_rotation = (R_t_i @ R_t_0.permute(0, 2, 1)) @ R_s
        new_expression = s_kp_info['exp'] + (t_i_kp_info['exp'] - t_0_kp_info['exp'])
    else:
        new_rotation = R_t_i
        new_expression = t_i_kp_info['exp']
    new_translation = s_kp_info['t'] + (t_i_kp_info['t'] - t_0_kp_info['t'])
    new_translation[..., 2].fill_(0)  # Keep the z-axis unchanged
    new_scale = s_kp_info['scale'] * (t_i_kp_info['scale'] / t_0_kp_info['scale'])
    return new_rotation, new_expression, new_translation, new_scale

def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def resize_to_limit(img, max_dim=1280, n=2):
    h, w = img.shape[:2]
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))
    n = max(n, 1)
    new_h = img.shape[0] - (img.shape[0] % n)
    new_w = img.shape[1] - (img.shape[1] % n)
    if new_h == 0 or new_w == 0:
        return img
    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]
    return img
