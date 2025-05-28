neural_sfm_configs = {'triangulation_mode': False, 'NEUSFM_enable_post_optimization': False, 'NEUSFM_refinement_chunk_size': 500, \
'NEUSFM_fine_match_model_path': 'save_weight/dim7.ckpt', 'NEUSFM_fine_match_cfg_path': 'hydra_training_configs/experiment/multiview_refinement_matching.yaml', \
'n_images': None, 'img_resize': 1200, 'img_preload': False, 'down_sample_ratio': 1, 'img_pair_strategy': 'edge_index', 'INDEX_num_of_pair': None, 'img_retrival_method': None, 'refine_iter_n_times': 2}

base_colmap_configs = {'ImageReader_camera_mode': 'auto', 'ImageReader_single_camera': False, 'min_model_size': 2, 'no_refine_intrinsics': False, 'n_threads': 16, 'use_pba': False, 'geometry_verify_thr': 15.0, \
'reregistration': {'abs_pose_max_error': 12, 'abs_pose_min_num_inliers': 30, 'abs_pose_min_inlier_ratio': 0.25, 'filter_max_reproj_error': 10}, \
'colmap_mapper_cfgs': {'init_max_error': 10, 'abs_pose_max_error': 12, 'filter_max_reproj_error': 10, 'tri_merge_max_reproj_error': 10, 'tri_complete_max_reproj_error': 10, 'tri_ignore_two_view_tracks': 0}}
