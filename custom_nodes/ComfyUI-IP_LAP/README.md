# ComfyUI-IP_LAP
the comfyui custom node of [IP_LAP](https://github.com/Weizhi-Zhong/IP_LAP) to make audio driven videos!
<div>
  <figure>
  <img alt='webpage' src="web.png?raw=true" width="600px"/>
  <figure>
</div>
    
## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
git clone https://github.com/AIFSH/ComfyUI-IP_LAP.git
cd ComfyUI-IP_LAP
pip install -r requirements.txt
```
download [weights](https://www.jianguoyun.com/p/DeXpK34QgZ-EChjI9YcFIAA) or [OneDrive](https://1drv.ms/f/s!Amqu9u09qiUGi7UJIADzCCC9rThkpQ?e=P1jG5N) and put the `*.pth` files in `ComfyUI-IP_LAP/weights`

Load checkpoint from: C:\Users\38957\Documents\GitHub\comfyui-backend-release\custom_nodes\ComfyUI-IP_LAP\weights\landmarkgenerator_checkpoint.pth
Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" to C:\Users\38957/.cache\torch\hub\checkpoints\vgg19-dcbb9e9d.pth
100%|███████████████████████████████████████████████████████████████████████████████| 548M/548M [00:12<00:00, 46.0MB/s]
Perceptual loss:
        Mode: vgg19
Load checkpoint from: C:\Users\38957\Documents\GitHub\comfyui-backend-release\custom_nodes\ComfyUI-IP_LAP\weights\renderer_checkpoint.pth
Downloading: "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" to C:\Users\38957/.cache\torch\hub\checkpoints\s3fd-619a316812.pth
100%|█████████████████████████████████████████████████████████████████████████████| 85.7M/85.7M [00:19<00:00, 4.58MB/s]
Downloading: "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip" to C:\Users\38957/.cache\torch\hub\checkpoints\2DFAN4-cd938726ad.zip



## Windows
There is a portable standalone build for Windows that should work for running on Nvidia GPUs and cuda>=11.8,
click [the link](https://www.bilibili.com/video/BV1qx4y1h7T2) to download
<div>
  <figure>
  <img alt='Wechat' src="1key.jpg?raw=true" width="300px"/>
  <figure>
</div>

## Tutorial
- [Demo](https://www.bilibili.com/video/BV1ht421J7SX)
- [FULL WorkFLOW](https://www.bilibili.com/video/BV1XE421T7ja)

## WeChat Group && Donate
<div>
  <figure>
  <img alt='Wechat' src="wechat.jpg?raw=true" width="300px"/>
  <img alt='donate' src="donate.jpg?raw=true" width="300px"/>
  <figure>
</div>
    
## Thanks
- [NativeSpeakerUI](https://github.com/AIFSH/NativeSpeakerUI)
- [IP_LAP](https://github.com/Weizhi-Zhong/IP_LAP)
