# ComfyUI MuseTalk


# model 
[ComfyUI\models\musetalk\musetalk.json](https://huggingface.co/TMElyralab/MuseTalk/blob/main/musetalk/musetalk.json)
[ComfyUI\models\musetalk\pytorch_model.bin](https://huggingface.co/TMElyralab/MuseTalk/blob/main/musetalk/pytorch_model.bin)

[ComfyUI\models\whisper\tiny.pt](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)


运行工作流还需要另外的模型:
用到controlnet的预处理,主要为了识别人脸范围

ComfyUI\models\controlnet_ckpts\lllyasviel\Annotators\dw-ll_ucoco_384.onnx

ComfyUI\models\controlnet_ckpts\hr16\yolo-nas-fp16\yolo_nas_s_fp16.onnx


Original repo:
https://github.com/kijai/ComfyUI-MuseTalk-KJ

https://github.com/TMElyralab/MuseTalk



