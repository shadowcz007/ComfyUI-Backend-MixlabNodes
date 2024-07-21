import os
from .hydit_v1_1.constants import SAMPLER_FACTORY

import folder_paths

def get_model_dir(m):
    try:
        dir=folder_paths.get_folder_paths(m)[0]
        if os.path.exists(dir):
            return dir
        else:
            return os.path.join(folder_paths.models_dir, m)
    except:
        return os.path.join(folder_paths.models_dir, m)
    
base_path = os.path.dirname(os.path.realpath(__file__))
HUNYUAN_PATH =get_model_dir('hunyuan')
SCHEDULERS_hunyuan = list(SAMPLER_FACTORY.keys())

T5_PATH = get_model_dir('t5')
LORA_PATH = get_model_dir('loras')