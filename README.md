# comfyui-backend-release


1、内置插件的comfyui包：https://github.com/shadowcz007/ComfyUI-Backend-MixlabNodes

2、py环境包，解压后把后缀版本号去掉，放到内置插件的comfyui包里，链接：https://pan.baidu.com/s/1lx12WLxgypTQuvhW1TfNmw?pwd=MAI0 

3、修改 config\extra_model_paths.yaml 里的模型地址



### python环境位置
写到```config\python_path.txt```


基于官方的[ 2.3.1+cu121 v0.0.1 环境 Python 3.11.8](https://github.com/comfyanonymous/ComfyUI/releases/download/v0.0.1/ComfyUI_windows_portable_nvidia.7z)


> 加速推理速度
 [flash-attn 安装方法](https://t.zsxq.com/CMcRp)
 
https://github.com/bdashore3/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3.1cxx11abiFALSE-cp311-cp311-win_amd64.whl


### 模型位置
可以修改```extra_model_paths.yaml```的```base_path```


<!-- ### custom_nodes位置
修改```extra_model_paths.yaml```的 ```other_ui``` 里的```base_path``` -->


## mac 安装
注意python版本3.11，torch版本 ==2.1.2 
完成安装后，记得给权限 ```chmod +x run_mac.sh```
运行```./run_mac.sh```


修改aux的模型地址为comfyui统一配置方式
custom_nodes\comfyui_controlnet_aux\utils.py
```
# 使用comfyui的目录结构
annotator_ckpts_path=folder_paths.get_folder_paths('controlnet_ckpts')[0]

```

IPA用到 insightface ,添加代码，优先使用配置的路径
```
def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)
```

comfyui-reactor-node 修改模型为配置的路径

glob.glob(models_path) ,models_path 需要 models_path = os.path.join(FACE_MODELS_PATH, "*")




