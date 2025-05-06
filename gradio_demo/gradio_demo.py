# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import random

import gradio as gr
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline


help_text = """
SuperEdit (InstructPix2Pix-SD1.5)
"""

model_id = "limingcv/SuperEdit_InstructP2P_SD15_BaseInstructDiffusion"  # Our model
# model_id = "timbrooks/instruct-pix2pix" # Default InstructPix2Pix model

# 添加示例数据
examples = [
    [
        "eval/imgs/24.jpeg",
        "Add a pair of black sunglasses for the dog",   # editing instruction
        50,                       # steps
        42,                       # seed
        10,                      # text_cfg_scale
        1.5,                      # image_cfg_scale
        1024,                     # resolution
    ],
    [
        "eval/imgs/24.jpeg",
        "Change the background to a sandy beach with the ocean in the distance.",   # editing instruction
        50,                       # steps
        42,                       # seed
        10,                      # text_cfg_scale
        1.5,                      # image_cfg_scale
        1024,                     # resolution
    ],
    [
        "eval/imgs/25.jpeg",
        "Change the image style to a watercolor painting",   # editing instruction
        50,                       # steps
        42,                       # seed
        10,                      # text_cfg_scale
        1.5,                      # image_cfg_scale
        1024,                     # resolution
    ],
    [
        "eval/imgs/23.jpeg",
        "Remove the collar from the dog's neck",   # editing instruction
        50,                       # steps
        42,                       # seed
        10,                      # text_cfg_scale
        1.5,                      # image_cfg_scale
        1024,                     # resolution
    ],
    [
        "eval/imgs/10.jpeg",
        "Transform the global scene to a winter setting with snow covering the houses, trees, and boat.",   # editing instruction
        50,                       # steps
        42,                       # seed
        10,                      # text_cfg_scale
        1.5,                      # image_cfg_scale
        1024,                     # resolution
    ],
]

def main():
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")

    def generate(
        input_image: Image.Image,
        instruction: str,
        steps: int,
        seed: int,
        text_cfg_scale: float,
        image_cfg_scale: float,
        resolution: int,
    ):
        # 确保数值参数都是正确的类型
        steps = int(steps)
        seed = int(seed)
        text_cfg_scale = float(text_cfg_scale)
        image_cfg_scale = float(image_cfg_scale)
        resolution = int(resolution)

        if instruction == "":
            return [input_image, seed]

        size = input_image.size
        input_image = input_image.resize((int(resolution * size[0] / size[1]), resolution))

        generator = torch.manual_seed(seed)
        edited_image = pipe(
            instruction, image=input_image,
            guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
            num_inference_steps=steps, generator=generator,
        ).images[0]

        edited_image = edited_image.resize((int(resolution * size[0] / size[1]), resolution))

        return [seed, text_cfg_scale, image_cfg_scale, edited_image]

    def reset():
        return [50, 42, 7.5, 1.5, 1024, None]

    with gr.Blocks() as demo:
        gr.HTML("""<h1 style="font-weight: 900; margin-bottom: 7px;">
   SuperEdit (InstructPix2Pix-SD1.5)
</h1>
<p>The image will be resized to the specified resolution on the shortest side while maintaining the original aspect ratio. The default sampling steps is 50.
<br/>
<p/>""")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                generate_button = gr.Button("Generate")
            with gr.Column(scale=1, min_width=100):
                reset_button = gr.Button("Reset")
            with gr.Column(scale=3):
                instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil", interactive=True)
            edited_image = gr.Image(label=f"Edited Image", type="pil", interactive=False)

        with gr.Row():
            steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Steps")
            seed = gr.Slider(minimum=0, maximum=100000, value=42, step=1, label="Seed")
            text_cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="Text CFG")
            image_cfg_scale = gr.Slider(minimum=0.5, maximum=2.0, value=1.5, step=0.1, label="Image CFG")
            resolution = gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="Resolution")

        gr.Markdown(help_text)

        # 添加示例功能
        gr.Examples(
            examples=examples,
            inputs=[
                input_image,
                instruction,
                steps,
                seed,
                text_cfg_scale,
                image_cfg_scale,
                resolution,
            ],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
            fn=generate,
            run_on_click=True,
            cache_examples=True,
        )

        generate_button.click(
            fn=generate,
            inputs=[
                input_image,
                instruction,
                steps,
                seed,
                text_cfg_scale,
                image_cfg_scale,
                resolution,
            ],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
        )
        reset_button.click(
            fn=reset,
            inputs=[],
            outputs=[steps, seed, text_cfg_scale, image_cfg_scale, resolution, edited_image],
        )

    demo.queue()
    demo.launch(share=True)


if __name__ == "__main__":
    main()