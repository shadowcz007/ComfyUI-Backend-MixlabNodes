# comfyui-backend-release


1、内置插件的comfyui包：https://github.com/shadowcz007/ComfyUI-Backend-MixlabNodes

2、py环境包，解压后把后缀版本号去掉，放到内置插件的comfyui包里，链接：https://pan.baidu.com/s/1lx12WLxgypTQuvhW1TfNmw?pwd=MAI0 

3、修改 config\extra_model_paths.yaml 里的模型地址



### python环境位置
写到```config\python_path.txt```


### 模型位置
可以修改```extra_model_paths.yaml```的```base_path```


<!-- ### custom_nodes位置
修改```extra_model_paths.yaml```的 ```other_ui``` 里的```base_path``` -->


## mac 安装
注意python版本3.11，torch版本 ==2.1.2 
完成安装后，记得给权限 ```chmod +x run_mac.sh```
运行```./run_mac.sh```