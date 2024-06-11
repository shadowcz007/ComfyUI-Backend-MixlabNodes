import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import math
import folder_paths
from contextlib import nullcontext
from tqdm import tqdm

import comfy.latent_formats
import comfy.model_management as mm
from comfy.utils import ProgressBar, unet_to_diffusers, load_torch_file
from comfy.model_base import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x
    
class MuseModelConfig:
    def __init__(self):
        unet_dtype = mm.unet_dtype()
        self.unet_config = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'dtype': unet_dtype, 'in_channels': 8, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 384, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}
        self.latent_format = comfy.latent_formats.SD15
        self.manual_cast_dtype = None
        self.sampling_settings = {}

class UNETLoader_MuseTalk:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "MuseTalk"

    def load_unet(self):
        
        model_path = os.path.join(folder_paths.models_dir,'musetalk')
        
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="TMElyralab/MuseTalk", local_dir=model_path, local_dir_use_symlinks=False)

        unet_weight_path = os.path.join(model_path, "musetalk","pytorch_model.bin") 
        
        sd = load_torch_file(unet_weight_path)
     
        model_config = MuseModelConfig()
        diffusers_keys = unet_to_diffusers(model_config.unet_config)
        
        new_sd = {}
        for k in diffusers_keys:
            if k in sd:
                new_sd[diffusers_keys[k]] = sd.pop(k)

        model = BaseModel(model_config)
        model.diffusion_model.load_state_dict(new_sd, strict=False)
        return (model,)

class muse_talk_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "whisper_features" : ("WHISPERFEAT",),
            "images": ("IMAGE",),
            "masked_images": ("IMAGE",),
            "batch_size": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
            "delay_frame": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",  )
    RETURN_NAMES = ("image",  )
    FUNCTION = "process"
    CATEGORY = "MuseTalk"

    def process(self, model, vae, whisper_features, images, masked_images, batch_size, delay_frame):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = mm.unet_dtype()
        vae_scale_factor = 0.18215
        mm.unload_all_models()
        mm.soft_empty_cache()
        
        images = images.to(dtype).to(device)
        masked_images = masked_images.to(dtype).to(device)      

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            timesteps = torch.tensor([0], device=device)
            vae.first_stage_model.to(device)
            input_latent_list = []
            for image, masked_image in zip(images, masked_images):
                latent = vae.encode(image.unsqueeze(0)).to(dtype).to(device) * vae_scale_factor
                masked_latents = vae.encode(masked_image.unsqueeze(0)).to(dtype).to(device) * vae_scale_factor

                latent_model_input = torch.cat([masked_latents, latent], dim=1)
                input_latent_list.append(latent_model_input)

            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            video_num = len(whisper_features)
            gen = self.datagen(whisper_features, input_latent_list_cycle, batch_size, delay_frame)
            
            total=int(np.ceil(float(video_num)/batch_size))
            
            out_frame_list = []
            
            pbar = ProgressBar(total)
            model.diffusion_model.to(device)
            for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=total)):
        
                tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
                audio_feature_batch = torch.stack(tensor_list).to(device) # torch, B, 5*N,384
                audio_feature_batch = PositionalEncoding(d_model=384)(audio_feature_batch)
              
                pred_latents = model.diffusion_model(latent_batch, timesteps, context=audio_feature_batch)

                pred_latents = (1 / vae_scale_factor) * pred_latents
                decoded = vae.decode(pred_latents)
                
                for frame in decoded:
                    out_frame_list.append(frame)
                pbar.update(1)

            out = torch.stack(out_frame_list, dim=0).float().cpu()
        model.diffusion_model.to(offload_device)
        vae.first_stage_model.to(offload_device)
        return (out,)
    
    def datagen(self, whisper_chunks,vae_encode_latents,batch_size,delay_frame):
        whisper_batch, latent_batch = [], []
        for i, w in enumerate(whisper_chunks):
            idx = (i+delay_frame)%len(vae_encode_latents)
            latent = vae_encode_latents[idx]
            whisper_batch.append(w)
            latent_batch.append(latent)

            if len(latent_batch) >= batch_size:
                whisper_batch = np.asarray(whisper_batch)
                latent_batch = torch.cat(latent_batch, dim=0)
                yield whisper_batch, latent_batch
                whisper_batch, latent_batch = [], []

        # the last batch may smaller than batch size
        if len(latent_batch) > 0:
            whisper_batch = np.asarray(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)

            yield whisper_batch, latent_batch

class vhs_audio_to_audio_tensor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vhs_audio": ("VHS_AUDIO",),
            "target_sample_rate": ("INT", {"default": 16000, "min": 0, "max": 48000}),
            "target_channels": ("INT", {"default": 1, "min": 1, "max": 2}),
             },
    
        }

    RETURN_TYPES = ("VCAUDIOTENSOR", "INT",)
    RETURN_NAMES = ("audio_tensor", "audio_dur",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, vhs_audio, target_sample_rate, target_channels):
        import io
        # Convert the byte stream to a tensor
        audio_bytes = vhs_audio()
        audio_buffer = io.BytesIO(audio_bytes)
        audio_tensor, sample_rate = torchaudio.load(audio_buffer)
        assert audio_tensor.shape[0] in [1, 2], "Audio must be mono or stereo."
        if target_channels == 1:
            audio_tensor = audio_tensor.mean(0, keepdim=True)
        elif target_channels == 2:
            *shape, _, length = audio_tensor.shape
            audio_tensor = audio_tensor.expand(*shape, target_channels, length)
        elif audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.expand(target_channels, -1)
        resampled_audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, target_sample_rate)
        audio_dur = audio_tensor.shape[1] / target_sample_rate
        
        return (resampled_audio_tensor, audio_dur,)

class whisper_to_features:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "audio_tensor" : ("VCAUDIOTENSOR",),
                "fps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("WHISPERFEAT", "INT",)
    RETURN_NAMES = ("whisper_chunks", "frame_count",)
    FUNCTION = "whispertranscribe"
    CATEGORY = "VoiceCraft"

    def whispertranscribe(self, audio_tensor, fps):
        from .musetalk.whisper.model import Whisper, ModelDimensions
        device = mm.get_torch_device()

        model_path = os.path.join(folder_paths.models_dir,'whisper',"tiny.pt")
        
        if not os.path.exists(model_path):
            print(f"Downloading whisper tiny model (72MB) to {model_path}")
            import requests
            url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
            response = requests.get(url)
            if response.status_code == 200:
                with open(model_path, 'wb') as file:
                    file.write(response.content)
            else:
                print(f"Failed to download {url} to {model_path}, status code: {response.status_code}")
        whisper_sd = torch.load(model_path, map_location=device)
        dims = ModelDimensions(**whisper_sd["dims"])
        model = Whisper(dims)
        model.load_state_dict(whisper_sd["model_state_dict"])
        del whisper_sd
        result = model.transcribe(audio_tensor.squeeze(0))
        
        embed_list = []
        for emb in result['segments']:
            encoder_embeddings = emb['encoder_embeddings']
            encoder_embeddings = encoder_embeddings.transpose(0,2,1,3)
            encoder_embeddings = encoder_embeddings.squeeze(0)
            start_idx = int(emb['start'])
            end_idx = int(emb['end'])
            emb_end_idx = int((end_idx - start_idx)/2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        whisper_feature = np.concatenate(embed_list, axis=0)

        audio_feat_length = [2,2]
        whisper_chunks = []
        whisper_idx_multiplier = 50./fps 
        i = 0
        print(f"video in {fps} FPS, audio idx in 50FPS")
        while 1:
            start_idx = int(i * whisper_idx_multiplier)
            selected_feature,selected_idx = self.get_sliced_feature(feature_array= whisper_feature,vid_idx = i,audio_feat_length=audio_feat_length,fps=fps)
            whisper_chunks.append(selected_feature)
            i += 1
            if start_idx>len(whisper_feature):
                break
        print(f"Whisper chunks: {len(whisper_chunks)}")
        return (whisper_chunks, len(whisper_chunks),)
    
    def get_sliced_feature(self,feature_array, vid_idx, audio_feat_length= [2,2],fps = 25):
        """
        Get sliced features based on a given index
        :param feature_array: 
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return: 
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []
        
        center_idx = int(vid_idx*50/fps) 
        left_idx = center_idx-audio_feat_length[0]*2
        right_idx = center_idx + (audio_feat_length[1]+1)*2
        
        for idx in range(left_idx,right_idx):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)
        
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)# 50*384
        return selected_feature,selected_idx
            
NODE_CLASS_MAPPINGS = {
    "whisper_to_features": whisper_to_features,
    "vhs_audio_to_audio_tensor": vhs_audio_to_audio_tensor,
    "muse_talk_sampler": muse_talk_sampler,
    "UNETLoader_MuseTalk": UNETLoader_MuseTalk
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "whisper_to_features": "Whisper To Features",
    "vhs_audio_to_audio_tensor": "VHS Audio To Audio Tensor",
    "muse_talk_sampler": "MuseTalk Sampler",
    "UNETLoader_MuseTalk": "UNETLoader_MuseTalk"
}
