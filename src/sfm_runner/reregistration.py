import os
import logging
import subprocess
import os.path as osp

from pathlib import Path
import time
from pdb import set_trace as bb
import pycolmap

def run_image_reregistration(
    deep_sfm_dir, after_refine_dir, colmap_path, image_path="/", colmap_configs=None, verbose=True, use_pycolmap=True
):
    logging.info("Running the bundle adjuster.")

    deep_sfm_model_dir = osp.join(deep_sfm_dir, "model")
    database_path = osp.join(deep_sfm_dir, "database.db")



    if use_pycolmap:
        database = pycolmap.Database(database_path)
        reconstruction = pycolmap.Reconstruction(deep_sfm_model_dir)

        # Incremental mapper 옵션 설정
        mapper_options = pycolmap.IncrementalMapperOptions()

        # 매퍼 인스턴스 생성
        database_cache = pycolmap.DatabaseCache().create(database=database, min_num_matches=0, ignore_watermarks=False, image_names=set(os.listdir(image_path)))
        
        # mapper = pycolmap.IncrementalMapper(database, reconstruction, mapper_options)
        mapper = pycolmap.IncrementalMapper(database_cache)
        mapper.begin_reconstruction(reconstruction)

        try:
            mapper_options.abs_pose_refine_focal_length = False
            mapper_options.abs_pose_refine_extra_params = False

            for image_id in mapper.reconstruction.images.keys() :
                if mapper.reconstruction.images[image_id].registered: # has_pose:  # in pycolmap 3.11
                     continue    
                mapper.register_next_image(mapper_options, image_id)

            
            return True

        except:
            print("Reregister Problem Occurs")
            mapper.reconstruction.write(after_refine_dir)
            return False    

    else:
        cmd = [
            str(colmap_path),
            "image_registrator",
            "--database_path",
            str(database_path),
            "--input_path",
            str(deep_sfm_model_dir),
            "--output_path",
            str(after_refine_dir),
        ]

        if colmap_configs is not None and colmap_configs["no_refine_intrinsics"] is True:
            cmd += [
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
            ]
        
        if 'reregistration' in colmap_configs:
            # Set to lower threshold to registrate more images
            cmd += [
                "--Mapper.abs_pose_max_error",
                str(colmap_configs['reregistration']['abs_pose_max_error']),
                "--Mapper.abs_pose_min_num_inliers",
                str(colmap_configs['reregistration']['abs_pose_min_num_inliers']),
                "--Mapper.abs_pose_min_inlier_ratio",
                str(colmap_configs['reregistration']['abs_pose_min_inlier_ratio']),
                "--Mapper.filter_max_reproj_error",
                str(colmap_configs['reregistration']['filter_max_reproj_error'])
            ]

        if verbose:
            logging.info(' '.join(cmd))
            ret = subprocess.call(cmd)
        else:
            ret_all = subprocess.run(cmd, capture_output=True)
            # with open(osp.join(after_refine_dir, 'reregistration_output.txt'), 'w') as f:
            #     f.write(ret_all.stdout.decode())
            ret = ret_all.returncode

        if ret != 0:
            print("Problem with image registration, existing.")
            exit(ret)


def main(
    deep_sfm_dir,
    after_refine_dir,
    colmap_path="colmap",
    image_path="/",
    colmap_configs=None,
    verbose=True,
    use_pycolmap=True
):
    assert Path(deep_sfm_dir).exists(), deep_sfm_dir

    Path(after_refine_dir).mkdir(parents=True, exist_ok=True)
    run_image_reregistration(
        deep_sfm_dir,
        after_refine_dir,
        colmap_path,
        image_path,
        colmap_configs=colmap_configs,
        verbose=verbose, use_pycolmap=use_pycolmap
    )
