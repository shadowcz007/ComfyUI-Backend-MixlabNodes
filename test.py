# 把引用的本地库的路径改为当前目录下 ，以下代码贴至 main.py 修改

import sys
import os

# 去掉环境里的comfyUI路径
comfyui_path = "ComfyUI" # 替换为你要判断和去掉的路径

for path in sys.path:
    file_name = os.path.basename(path)
    if comfyui_path ==file_name:
        print(f"{path} 包含 ComfyUI 路径")
        sys.path.remove(path)
    else:
        print(f"{path} 不包含 ComfyUI 路径")

# 添加当前的ComfyUI路径
current_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(current_directory,'ComfyUI'))

print(sys.path)
# import cuda_malloc
# from nodes import init_custom_nodes
import yaml,folder_paths

def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                print("Adding extra search path", x, full_path)
                folder_paths.add_model_folder_path(x, full_path)

                # print(full_path)

del folder_paths.folder_names_and_paths["custom_nodes"]


extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ComfyUI/extra_model_paths.yaml")
print(extra_model_paths_config_path)
if os.path.isfile(extra_model_paths_config_path):
    load_extra_path_config(extra_model_paths_config_path)

node_paths = folder_paths.get_folder_paths("custom_nodes")
print('#####load_custom_nodes node_paths',node_paths)