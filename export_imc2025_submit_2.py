from src.colmap.read_write_model import read_images_binary
from src.utils.colmap.eval_helper import get_best_colmap_index

import os
import numpy as np
import argparse
from collections import defaultdict


from pdb import set_trace as bb

def extract_colmap_pose(model_path):
    images = read_images_binary(model_path)

def comma_separated_list(arg):
    return arg.strip().split(',')


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="construct SfM_dataset folder based on clustering")
    # parser.add_argument('--scene_name', default="stairs")
    parser.add_argument('--scene_names', default=None, type=comma_separated_list)
    parser.add_argument('--model_path', default="./SfM_dataset/IMC2025/") # folder path where images exist, folder name should be scene_name
    parser.add_argument('--matcher', default="loftr_official") # folder path where images exist, folder name should be scene_name

    args = parser.parse_args()
    scene_lists = os.listdir(args.model_path)
    scene_lists.sort()

    if args.scene_names is None:
        grouped = defaultdict(list)

        for item in scene_lists:
            key = item.split('___')[0]
            grouped[key].append(item)

        scene_cluster_list = list(grouped.values())

    else:
        grouped = defaultdict(list)

        for item in scene_lists:
            key = item.split('___')[0]
            if key in args.scene_names:
                grouped[key].append(item)

        scene_cluster_list = list(grouped.values())

    result = []
    for cluster_list in scene_cluster_list:
        for idx, one_scene in enumerate(cluster_list):
            print(one_scene)
            print("{}'s {}th Cluster Info: ".format(one_scene.split("___")[0], str(idx+1)))
            index = get_best_colmap_index(os.path.join(args.model_path, one_scene, "DetectorFreeSfM_{}_coarse_only__scratch_no_intrin".format(args.matcher), "colmap_coarse"))
            if index != str(-1):
                images = read_images_binary(os.path.join(args.model_path, one_scene, "DetectorFreeSfM_{}_coarse_only__scratch_no_intrin".format(args.matcher), "colmap_coarse", str(index), "images.bin"))
                image_files = [f for f in os.listdir(os.path.join(args.model_path, one_scene, "images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                print("{} out of {} images Registered.".format(len(images.keys()), len(image_files)))
                print("=====================================")
                for key, value in images.items():
                    rotmat = qvec2rotmat(value.qvec)
                    result.append("{},cluster{},{},{};{};{};{};{};{};{};{};{};{};{};{}".format(one_scene.split("___")[0], str(idx+1), value.name, \
                        rotmat[0,0], rotmat[0,1], rotmat[0,2], rotmat[1,0], rotmat[1,1], rotmat[1,2], rotmat[2,0], rotmat[2,1], rotmat[2,2],\
                        value.tvec[0], value.tvec[1], value.tvec[2]))
            else:
                print("Cluster SfM reconstruction Fail")
                print("=====================================")
    
    print(result)                    


