from typing import ChainMap
import hydra
import os
import os.path as osp
import multiprocessing
import ray
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import math
import random
import torch
from omegaconf import DictConfig, OmegaConf
from src.utils.ray_utils import ProgressBar, chunks, chunks_balance
from src.utils.metric_utils import aggregate_multi_scene_metrics
from src import DetectorFreeSfM
import argparse

from src.post_optimization.post_optimization import post_optimization
from src.utils.colmap.eval_helper import get_best_colmap_index
from pdb import set_trace as bb
from base_configs import *

# YAML
def load_yaml_to_dict(yaml_file_path):
    """
    
    Args:
        yaml_file_path (str): YAM
    
    Returns:
        dict: YAML
    """
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def get_all_file_paths(root_dir):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.abspath(os.path.join(dirpath, filename))
            file_paths.append(full_path)
    return file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", type=str, required=False, default="/cephfs/jongmin/cvpr2024/test_refinement/SfM_dataset/eth3d_triangulation_dataset/pipes/images")     # should be full absolute path
    parser.add_argument("--colmap_coarse_dir", type=str, required=False, default="/cephfs/jongmin/cvpr2024/test_refinement/SfM_dataset/eth3d_triangulation_dataset/pipes/DetectorFreeSfM_loftr_official_coarse_only__pri_pose/colmap_coarse")     
    parser.add_argument("--refined_colmap_dir", type=str, required=False, default = "/cephfs/jongmin/cvpr2024/test_refinement/SfM_dataset/eth3d_triangulation_dataset/pipes/DetectorFreeSfM_loftr_official_coarse_only__pri_pose/colmap_refined")

    args = parser.parse_args()

    img_list = get_all_file_paths(args.img_folder)
    # img_list.sort()
    best_model_id = get_best_colmap_index(args.colmap_coarse_dir)

    
    post_optimization(
        img_list,
        None,
        match_out_pth=None,
        chunk_size=neural_sfm_configs["NEUSFM_refinement_chunk_size"],
        matcher_model_path=neural_sfm_configs["NEUSFM_fine_match_model_path"],
        matcher_cfg_path=neural_sfm_configs["NEUSFM_fine_match_cfg_path"],
        img_resize=neural_sfm_configs["img_resize"],
        img_preload=False,
        colmap_coarse_dir=osp.join(args.colmap_coarse_dir, best_model_id),
        refined_model_save_dir=args.refined_colmap_dir,
        only_basename_in_colmap=True,
        fine_match_use_ray=False,
        ray_cfg=None,
        colmap_configs=colmap_configs,
        refine_iter_n_times=neural_sfm_configs["refine_iter_n_times"],
        refine_3D_pts_only=neural_sfm_configs["triangulation_mode"],
    )
