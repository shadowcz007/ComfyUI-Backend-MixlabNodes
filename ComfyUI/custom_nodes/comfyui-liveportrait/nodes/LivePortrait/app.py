# coding: utf-8

"""
The entrance of the gradio
"""

import os
import os.path as osp
import gradio as gr
import tyro
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig
gradio_pipeline = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)
# assets
title_md = "assets/gradio_title.md"
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
data_examples = [
    [osp.join(example_portrait_dir, "s1.jpg"), osp.join(example_video_dir, "d1.mp4"), True, True, True],
    [osp.join(example_portrait_dir, "s2.jpg"), osp.join(example_video_dir, "d2.mp4"), True, True, True],
    [osp.join(example_portrait_dir, "s3.jpg"), osp.join(example_video_dir, "d5.mp4"), True, True, True],
    [osp.join(example_portrait_dir, "s5.jpg"), osp.join(example_video_dir, "d6.mp4"), True, True, True],
    [osp.join(example_portrait_dir, "s7.jpg"), osp.join(example_video_dir, "d7.mp4"), True, True, True],
]
#################### interface logic ####################
# Define components first
eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eye-close ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-close ratio")
output_image = gr.Image(label="The animated image with the given eye-close and lip-close ratio.", type="numpy")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(load_description(title_md))
    gr.Markdown(load_description("assets/gradio_description_upload.md"))
    with gr.Row():
        with gr.Accordion(open=True, label="Reference Portrait"):
            image_input = gr.Image(label="Please upload the reference portrait here.", type="filepath")
        with gr.Accordion(open=True, label="Driving Video"):
            video_input = gr.Video(label="Please upload the driving video here.")
    gr.Markdown(load_description("assets/gradio_description_animation.md"))
    with gr.Row():
        with gr.Accordion(open=True, label="Animation Options"):
            with gr.Row():
                flag_relative_input = gr.Checkbox(value=True, label="relative pose")
                flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                flag_do_crop_input = gr.Checkbox(value=True, label="do crop")
    with gr.Row():
        process_button_animation = gr.Button("🚀 Animate", variant="primary")
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video in the original image space"):
                output_video = gr.Video(label="The animated video after pasted back.")
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video"):
                output_video_concat = gr.Video(label="The animated video and driving video.")
    with gr.Row():
        process_button_reset = gr.ClearButton([image_input, video_input, output_video, output_video_concat], value="🧹 Clear")
    with gr.Row():
        # Examples
        gr.Markdown("## You could choose the examples below ⬇️")
    with gr.Row():
        gr.Examples(
            examples=data_examples,
            inputs=[
                image_input,
                video_input,
                flag_relative_input,
                flag_do_crop_input,
                flag_remap_input
            ],
            examples_per_page=5
        )
    gr.Markdown(load_description("assets/gradio_description_retargeting.md"))
    with gr.Row():
        with gr.Column():
            process_button_close_ratio = gr.Button("🤖 Calculate the eye-close and lip-close ratio")
            process_button_retargeting = gr.Button("🚗 Retargeting", variant="primary")
            process_button_reset_retargeting = gr.ClearButton([output_image, eye_retargeting_slider, lip_retargeting_slider], value="🧹 Clear")
        # with gr.Column():
            eye_retargeting_slider.render()
            lip_retargeting_slider.render()
        with gr.Column():
            with gr.Accordion(open=True, label="Eye and lip Retargeting Result"):
                output_image.render()
    # binding functions for buttons
    process_button_close_ratio.click(
        fn=gradio_pipeline.prepare_retargeting,
        inputs=image_input,
        outputs=[eye_retargeting_slider, lip_retargeting_slider],
        show_progress=True
    )
    process_button_retargeting.click(
        fn=gradio_pipeline.execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider],
        outputs=output_image,
        show_progress=True
    )
    process_button_animation.click(
        fn=gradio_pipeline.execute_video,
        inputs=[
            image_input,
            video_input,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input
        ],
        outputs=[output_video, output_video_concat],
        show_progress=True
    )
    process_button_reset.click()
    process_button_reset_retargeting
##########################################################

demo.launch(
    server_name=args.server_name,
    server_port=args.server_port,
    share=args.share,
)
