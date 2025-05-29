import os
import logging
import subprocess
import os.path as osp
from shutil import rmtree
import multiprocessing

from pathlib import Path
from pdb import set_trace as bb
import pycolmap 
import pyceres

COLMAP_PATH = os.environ.get("COLMAP_PATH", 'colmap') # 'colmap is default value



def get_image_ids_from_txt(txt_file_path, images):
    image_ids = []
    
    # txt 파일 읽기
    with open(txt_file_path, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    # 각 이미지 이름에 대해 image_id 찾기
    for image_name in image_names:
        found = False
        
        # reconstruction의 모든 이미지를 순회하며 이름 비교
        for image_id, image in images.items():
            if image.name == image_name:
                image_ids.append(image_id)
                found = True
                break
            
    return image_ids

def run_incremental_model_refiner(
    deep_sfm_dir, after_refine_dir, no_filter_pts=False, image_path="/", colmap_configs=None, verbose=True, refine_3D_pts_only=False, filter_threshold=2, use_pba=False, normalize_reconstruction=False, fix_image_poses=False, use_pycolmap=True
):
    logging.info("Running the bundle adjuster.")

    deep_sfm_model_dir = osp.join(deep_sfm_dir, "model")
    database_path = osp.join(deep_sfm_dir, "database.db")
    threshold = filter_threshold
    
    if use_pycolmap:
        try:
            database = pycolmap.Database(database_path)
            reconstruction = pycolmap.Reconstruction(deep_sfm_model_dir)

            # Incremental mapper 옵션 설정
            mapper_options = pycolmap.IncrementalMapperOptions()
            tri_options = pycolmap.IncrementalTriangulatorOptions()
            
            tri_options.complete_max_reproj_error = float(filter_threshold)
            tri_options.merge_max_reproj_error = float(filter_threshold)
            mapper_options.filter_max_reproj_error = float(filter_threshold)

            max_num_refinements = 5
            
            ba_options = pycolmap.BundleAdjustmentOptions()
            # ba_options.solver_options.max_num_iterations = 50

            # 매퍼 인스턴스 생성
            database_cache = pycolmap.DatabaseCache().create(database=database, min_num_matches=0, ignore_watermarks=False, image_names=set(os.listdir(image_path)))
            
            # mapper = pycolmap.IncrementalMapper(database, reconstruction, mapper_options)
            mapper = pycolmap.IncrementalMapper(database_cache)
            mapper.begin_reconstruction(reconstruction)
            # maps = pycolmap.incremental_mapping( database_path=database_path[0],  image_path=image_path[0], output_path = '', input_path=deep_sfm_model_dir,  options=mapper_options  )


            fixed_image_ids = get_image_ids_from_txt(osp.join(deep_sfm_dir, 'fixed_images.txt'), reconstruction.images)
            if len(fixed_image_ids) != 0:
                ba_config = pycolmap.BundleAdjustmentConfig()

                # 모든 등록된 이미지 추가
                for image_id in reconstruction.reg_image_ids():
                    ba_config.add_image(image_id)

                # 특정 이미지들의 pose를 고정
                if refine_3D_pts_only:
                    for image_id in fixed_image_ids:
                        if reconstruction.is_image_registered(image_id):
                            # 전체 pose를 고정 (rotation + translation)
                            ba_config.set_constant_cam_pose(image_id)
                else:
                    if reconstruction.is_image_registered(fixed_image_ids[0]):
                        ba_config.set_constant_cam_pose(fixed_image_ids[0])
                        # ba_config.set_constant_pose(fixed_image_ids[0])  => pycolmap 3.11
                        if reconstruction.is_image_registered(fixed_image_ids[1]):
                            ba_config.set_constant_cam_positions(fixed_image_ids[1], [0])


                # Refine Start
                for _ in range(max_num_refinements):
                    # Bundle adjustment configuration
                    # Bundle adjuster 생성 및 실행


                    # below pycolmap 3.10
                    bundle_adjuster = pycolmap.BundleAdjuster(ba_options, ba_config)
                    # alternative equivalent python-based bundle adjustment (slower):
                    # bundle_adjuster = PyBundleAdjuster(ba_options, ba_config)
                    
                    
                    bundle_adjuster.set_up_problem(mapper.reconstruction, ba_options.create_loss_function())
                    solver_options = bundle_adjuster.set_up_solver_options(
                        bundle_adjuster.problem, ba_options.solver_options
                    )
                    summary = pyceres.SolverSummary()
                    pyceres.solve(solver_options, bundle_adjuster.problem, summary)

                    """
                    # pycolmap 3.11
                    bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_options, ba_config, mapper.reconstruction)
                    success = bundle_adjuster.solve()
                    """



                    num_observations = mapper.reconstruction.compute_num_observations()

                    num_changed_observations = mapper.complete_and_merge_tracks(tri_options)
                    num_changed_observations += mapper.filter_points(mapper_options)

                    changed = (
                        num_changed_observations / num_observations
                        if num_observations > 0
                        else 0
                    )
                    # print("num changed: ", changed)


            else:
                for _ in range(max_num_refinements):
                    mapper.adjust_global_bundle(mapper_options, ba_options)
                    # adjust_global_bundle(mapper, mapper_options, ba_options)
                    # if normalize_reconstruction:
                    #     mapper.reconstruction.normalize()

                    num_observations = mapper.reconstruction.compute_num_observations()
        
                    num_changed_observations = mapper.complete_and_merge_tracks(tri_options)
                    num_changed_observations += mapper.filter_points(mapper_options)

                    changed = (
                        num_changed_observations / num_observations
                        if num_observations > 0
                        else 0
                    )

            os.makedirs(after_refine_dir, exist_ok=True)
            mapper.reconstruction.write(after_refine_dir)
            
            return True

        except:
            print("Wrong_SfM_refine")
            bb()
            return False

    else:
        print("RUN COLMAP BIN")
        cmd = [
            COLMAP_PATH,
            "incremental_model_refiner_no_filter_pts" if no_filter_pts else "incremental_model_refiner",
            "--input_path",
            str(deep_sfm_model_dir),
            "--output_path",
            str(after_refine_dir),
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--Mapper.filter_max_reproj_error",
            str(threshold),
            "--Mapper.tri_merge_max_reproj_error",
            str(threshold),
            "--Mapper.tri_complete_max_reproj_error",
            str(threshold),
            "--Mapper.extract_colors",
            str('1')
        ]

        if use_pba:
            # NOTE: PBA does not allow share intrinsics or fix extrinsics, and only allow SIMPLE_RADIAL camera model
            cmd += [
                "--Mapper.ba_global_use_pba",
                "1"
            ]
        else:
            cmd += [
                "--image_list_path",
                str(osp.join(deep_sfm_dir, 'fixed_images.txt')),
            ]
            pass

        if (colmap_configs is not None and colmap_configs["no_refine_intrinsics"] is True) or refine_3D_pts_only:
            cmd += [
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_extra_params", # Distortion params
                "0",
            ]

        if colmap_configs is not None and 'n_threads' in colmap_configs:
            cmd += ["--Mapper.num_threads", str(min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16))]

        if refine_3D_pts_only:
            if "--image_list_path" not in cmd:
                cmd += [
                    "--image_list_path",
                    str(osp.join(deep_sfm_dir, 'fixed_images.txt')),
                ] # For triangulation, must fix!

            cmd += [
                "--Mapper.fix_existing_images",
                "1",
            ]

        if verbose:
            logging.info(' '.join(cmd))
            ret = subprocess.call(cmd)
        else:
            ret_all = subprocess.run(cmd, capture_output=True)
            # with open(osp.join(after_refine_dir, 'incremental_model_refiner_output.txt'), 'w') as f:
            #     f.write(ret_all.stdout.decode())
            ret = ret_all.returncode

        if ret != 0:
            logging.warning(f"Problem with run_incremental_model_refiner for {deep_sfm_model_dir}, existing.")
            return False
        else:
            return True


def main(
    deep_sfm_dir,
    after_refine_dir,
    no_filter_pts=False,
    image_path="/",
    colmap_configs=None,
    refine_3D_pts_only=False,
    filter_threshold=2,
    use_pba=False,
    verbose=True,
    use_pycolmap=True
):
    assert Path(deep_sfm_dir).exists(), deep_sfm_dir

    if osp.exists(after_refine_dir):
        rmtree(after_refine_dir)
    Path(after_refine_dir).mkdir(parents=True, exist_ok=True)
    success = run_incremental_model_refiner(
            deep_sfm_dir,
            after_refine_dir,
            no_filter_pts,
            image_path,
            colmap_configs=colmap_configs,
            refine_3D_pts_only=refine_3D_pts_only,
            filter_threshold=filter_threshold,
            use_pba=use_pba,
            verbose=verbose, use_pycolmap=use_pycolmap
        )
    return success
