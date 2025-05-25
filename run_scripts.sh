python construct_sfm_data.py --img_folder /NAS2/ksy/IMC2025/train/stairs --matching_pair /NAS2/ksy/IMC2025/train/stairs/edges_518_2.txt
python eval_dataset.py +imc2025=dfsfm.yaml scene_list=[fbk_vineyard___0,fbk_vineyard___1,fbk_vineyard___2,fbk_vineyard___3] neuralsfm.NEUSFM_coarse_matcher=DKM #loftr_official
python export_imc2025_submit.py  --scene_names fbk_vineyard,stairs --matcher DKM #loftr_official
