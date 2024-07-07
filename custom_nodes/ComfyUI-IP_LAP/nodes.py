import os

import folder_paths
import imageio_ffmpeg as ffmpeg
import subprocess
import platform

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



class IP_LAP:

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
        
        ip_lap = IP_LAP_infer(T,Nl,ref_img_N,img_size,mel_step_size,face_det_batch_size,model_path)
        video_name = os.path.basename(video)
        # print(audio)
        out_video_file = os.path.join(out_path, f"ip_lap_{video_name}")
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

        ip_lap(video,audio_p,out_video_file)
        # res_video_file = os.path.join(out_path, f"result_ip_lap_{video_name}")
        # command = f'ffmpeg -y -i {out_video_file} -i {audio} -map 0:0 -map 1:0 -c:a libmp3lame -q:a 1 -q:v 1 -shortest {res_video_file}'
        # subprocess.call(command, shell=platform.system() != 'Windows')
        return (out_video_file,)