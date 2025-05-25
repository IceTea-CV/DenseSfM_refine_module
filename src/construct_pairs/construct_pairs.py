from loguru import logger
import ray
import os
from .pairs_exhaustive import exhaustive_all_pairs
from .pairs_from_img_index import pairs_from_index
from pdb import set_trace as bb

@ray.remote(num_cpus=2, num_gpus=0.1, max_calls=1)
def construct_img_pairs_ray_wrapper(*args, **kwargs):
    return construct_img_pairs(*args, **kwargs)



def load_edge(base_path, pair_file):
    pairs = []

    with open(pair_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append(str(os.path.join(base_path, parts[0])) + " " + str(os.path.join(base_path, parts[1])))
    return pairs
    

def construct_img_pairs(img_list, args, strategy='exhaustive', pair_path=None, verbose=True):
    # Construct image pairs:
    logger.info(f'Using {strategy} matching build pairs') if verbose else None
    scene_name = img_list[0].split('/')[-3]
    pair_file = os.path.join('/'.join(img_list[0].split('/')[:3]), "pairs.txt")
    base_path = '/'.join(img_list[0].split('/')[:4])

    if len(img_list) <= 10:
        strategy="exhaustive"

    if strategy == 'exhaustive':
        img_pairs = exhaustive_all_pairs(img_list)
    elif strategy == 'pair_from_index':
        img_pairs = pairs_from_index(img_list, args.INDEX_num_of_pair)
    elif strategy == 'edge_index':
        img_pairs = load_edge(base_path, pair_file)
    else:
        raise NotImplementedError
    
    return img_pairs
