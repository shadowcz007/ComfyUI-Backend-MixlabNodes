
import os
import folder_paths
import torchaudio

class SpeechRecognition:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "upload":("AUDIOINPUTMIX",),   },
                "optional":{
                    "start_by":("INT", {
                        "default": 0, 
                        "min": 0, #Minimum value
                        "max": 2048, #Maximum value
                        "step": 1, #Slider's step
                        "display": "number" # Cosmetic only: display as "number" or "slider"
                    }),
                   
                }
                }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)

    def run(self,upload,start_by):
        return {"ui": {"start_by": [start_by]}, "result": (upload,)}


class SpeechSynthesis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "♾️Mixlab/Audio"

    def run(self, text):
        # print(session_history)
        return {"ui": {"text": text}, "result": (text,)}
    


class AudioPlayNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = ""
        self.compress_level = 4
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
              }, 
                }
    
    RETURN_TYPES = ()
  
    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()

    OUTPUT_NODE = True
  
    def run(self,audio):

        # 判断是否是 Tensor 类型
        is_tensor = not isinstance(audio, dict)
        # print('#判断是否是 Tensor 类型',is_tensor,audio)
        if not is_tensor and 'waveform' in audio and 'sample_rate' in audio:
            # {'waveform': tensor([], size=(1, 1, 0)), 'sample_rate': 44100}
            is_tensor=True

        if is_tensor:
            filename_prefix=""
            # 保存
            filename_prefix += self.prefix_append
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
            results = list()
            
            filename_with_batch_num = filename.replace("%batch_num%", str(1))
            file = f"{filename_with_batch_num}_{counter:05}_.wav"
            
            torchaudio.save(os.path.join(full_output_folder, file), audio['waveform'].squeeze(0), audio["sample_rate"])
            results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
            
        else:
            results=[audio]
                

        # print(audio)
        return {"ui": {"audio":results}}