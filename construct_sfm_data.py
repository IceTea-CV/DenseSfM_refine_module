import torch
import numpy as np
import os 
import argparse
from pdb import set_trace as bb

def reconstruct_dataset(cluster_list, img_folder, save_folder, scene_name, img_pairs_info):
    
    # os.makedirs(, exist_ok=True)
    
    for idx in range(len(cluster_list)):
        cluster_folder_name = scene_name + "___" + str(idx)
        img_names = cluster_list[idx]


        os.system("rm -r {}".format(str(os.path.join(save_folder, cluster_folder_name))))
        os.makedirs(os.path.join(save_folder, cluster_folder_name), exist_ok=True)
        save_img_folder = os.path.join(save_folder, cluster_folder_name, "images")
        os.makedirs(save_img_folder, exist_ok=True)
        
        

        for img_name in img_names:
            from_path = os.path.join(img_folder, img_name)
            to_path = save_img_folder
            os.system("cp -r {} {}".format(from_path, to_path))

        save_pairs_file = os.path.join(save_folder, cluster_folder_name, "pairs.txt")

        with open(save_pairs_file, 'w') as f:
            for value in img_pairs_info[idx]:
                line = f"{value[0]} {value[1]}\n"
                f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="construct SfM_dataset folder based on clustering")
    # parser.add_argument('--cluster_list', default="./temp.pth")
    parser.add_argument('--img_folder', default="./img_folder") # folder path where images exist, folder name should be scene_name
    parser.add_argument('--matching_pair', default="./edges.txt")
    

    args = parser.parse_args()
    
    save_folder = os.path.join("SfM_dataset", "IMC2025")
    if str(args.img_folder).endswith("/"):
        args.img_folder = args.img_folder[:-1]
    scene_name =  os.path.basename(args.img_folder)
    print(scene_name)
    os.makedirs(save_folder, exist_ok=True)


    """
    cluster_list = list(torch.load(args.cluster_list).values())
    remove_index = []
    for cluster_idx, cluster in enumerate(cluster_list):
        if len(cluster) <= 1:
            remove_index.append(cluster_idx)

    cluster_list = [x for i, x in enumerate(cluster_list) if i not in remove_index]

    num_cluster = len(cluster_list)
    """

    all_img_list = [f for f in os.listdir(args.img_folder) if f.lower().endswith(('.png', '.jpg')) and os.path.isfile(os.path.join(args.img_folder, f))]
    all_img_list.sort()

    
    img_pairs_info = {}
    with open(args.matching_pair, 'r') as file:
        for line in file:
            parts = list(map(int, line.strip().split()))
            key = int(parts[0])
            img1 = all_img_list[int(parts[1])]
            img2 = all_img_list[int(parts[2])]

            if key not in img_pairs_info:
                img_pairs_info[key] = []
            img_pairs_info[key].append([img1, img2])

    num_cluster = len(img_pairs_info.keys())

    cluster_list = []
    for idx in range(num_cluster):
        value_set = {row[0] for row in img_pairs_info[idx]} | {row[1] for row in  img_pairs_info[idx]}
        cluster_list.append(value_set)        

    
    # img_pairs_info = {k: v for k, v in img_pairs_info.items() if k not in remove_index}


    reconstruct_dataset(cluster_list, args.img_folder, save_folder, scene_name, img_pairs_info)

    scene_list = str([scene_name + "___" + str(i) for i in range(num_cluster)])
    scene_list = scene_list.replace("'", "").replace('"', "").replace(" ", "")


    print("python eval_dataset.py +imc2025=dfsfm.yaml scene_list={}".format(str(scene_list)))
    