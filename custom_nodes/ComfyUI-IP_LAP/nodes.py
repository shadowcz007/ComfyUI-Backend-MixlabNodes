import os

import folder_paths
import imageio_ffmpeg as ffmpeg
import subprocess
import platform,torchaudio

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)


from .ip_lap.inference import IP_LAP_infer

input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()
temp_path = folder_paths.get_temp_directory()

model_path=get_model_dir('ip_lap')

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

face_landmarks_detector_path = os.path.join(current_directory,"weights","face_landmarker.task")

class IP_LAP:
    def __init__(self):
        self.ip_lap=None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "audio": ("AUDIO",),
                        "video": ("SCENE_VIDEO",),
                        "T":("INT",{
                            "default": 5,
                        }),
                        "Nl":("INT",{
                            "default": 15,
                        }),
                        "ref_img_N":("INT",{
                            "default": 25,
                        }),
                        "img_size":("INT",{
                            "default": 128,
                        }),
                        "mel_step_size":("INT",{
                            "default": 16,
                        }),
                        "face_det_batch_size":("INT",{
                            "default": 4,
                        }),
                       
                    }
                }

    CATEGORY = "AIFSH_IP_LAP" 

    RETURN_TYPES = ("SCENE_VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "process"

    def process(self, audio, video, T=5,Nl=15,ref_img_N=25,img_size=128,
                mel_step_size=16,face_det_batch_size=4):
        
        if self.ip_lap==None:
            self.ip_lap = IP_LAP_infer(T,
                              Nl,
                              ref_img_N,
                              img_size,
                              mel_step_size,
                              face_det_batch_size,
                              model_path,
                              facemash_model_dir=face_landmarks_detector_path
                              )
        video_name = os.path.basename(video)
        # print(audio)
        out_video_file = os.path.join(out_path, f"ip_lap_{video_name}")

        if 'waveform' in audio and 'sample_rate' in audio and not "audio_path" in audio:
            filename_prefix="ip_lap_"
            
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, 
                temp_path)
            results = list()
            
            filename_with_batch_num = filename.replace("%batch_num%", str(1))
            file = f"{filename_with_batch_num}_{counter:05}_.wav"
            
            audio['audio_path']=os.path.join(full_output_folder, file)

            torchaudio.save(audio['audio_path'], audio['waveform'].squeeze(0), audio["sample_rate"])

        audio_p=audio['audio_path']

        def convert_to_25fps(input_path, output_path):
            # 获取 ffmpeg 可执行文件的路径
            ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

            # 构建 ffmpeg 命令
            command = [
                ffmpeg_exe,
                '-y',
                '-i', input_path,
                '-r', '25',  # 设置帧率为 25 fps
                output_path
            ]

            # 调用 ffmpeg 命令
            # subprocess.run(command)
            subprocess.call(command, shell=platform.system() != 'Windows',stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return output_path
        
        def extract_audio(audio_file, temp_dir):
            print(f'Extracting audio ... from {audio_file}')
            if not audio_file.endswith('.wav'):
                tmp_file = os.path.join(temp_dir, 'temp.wav')
                ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
                command = [ffmpeg_exe, '-y', '-i', audio_file, '-strict', '-2', tmp_file]
                
                subprocess.call(command, shell=platform.system() != 'Windows', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                audio_file = tmp_file

            return audio_file

        video=convert_to_25fps(video,os.path.join(temp_path, f"25fps_{video_name}"))
        audio_p=extract_audio(audio_p,temp_path)

        self.ip_lap(video,audio_p,out_video_file)

        return (out_video_file,)