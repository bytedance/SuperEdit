<div align="center">
 👋 Hi, everyone!
    <br>
    We are <b>ByteDance Intelligent Creation team.</b>
</div>


# [ICCV 2025] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing
<p align="center">
  <a href="https://liming-ai.github.io/SuperEdit">
    <img src="https://img.shields.io/badge/SuperEdit-Project Page-yellow">
  </a>
  <a href="https://arxiv.org/abs/2505.02370">
    <img src="https://img.shields.io/badge/SuperEdit-Tech Report-red">
  </a>
  <a href="https://huggingface.co/datasets/limingcv/SuperEdit-40K">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-SuperEdit--40K-red">
  <a href="./LICENSE-Apache">
    <img src="https://img.shields.io/badge/License-Apache-blue">
  </a>
</p>

We are extremely delighted to release SuperEdit. SuperEdit achieves state-of-the-art image editing performance by improving the supervision quality. Our method does not require extra VLM modules or pre-training tasks used in previous work, offering a more direct and efficient way to provide better supervision signals, and providing a novel, simple, and effective solution for instruction-based image editing. Refer to [Project Website](https://liming-ai.github.io/SuperEdit/) for a quick review.

# Getting started
1. Prepare environment
    ```bash
    bash prepare_env.sh
    ```
2. Dataset
The dataset should be automatically downloaded when you run the training code. If you want to preview it, please refer to [SuperEdit-40K](https://huggingface.co/datasets/limingcv/SuperEdit-40K) on HuggingFace.

3. Training (8x 80G A100 by default)
    ```bash
    bash superedit/instruct_pix2pix/train_sd15.sh
    ```
4. Evaluation
    ```bash
    bash superedit/instruct_pix2pix/eval_sd15.sh
    ```
5. Gradio Demo
    ```bash
    python3 gradio_demo/gradio_demo.py
    ```

# What's New in SuperEdit?
## Data is the Key to Instruction-based Image Editing
![](https://liming-ai.github.io/SuperEdit/static/images/motivation.png)
Unlike existing efforts that attempt to (a) scale up edited images with noisy supervision;(b) introduce massive VLMs into editing model architecture;(c) perform additional pre-training tasks;(d) we focus on improving the effectiveness of supervision signals, which is the fundamental issue of image editing.


## Diffusion Timestep-Prior Makes the Unified Rectification Guideline
![](https://liming-ai.github.io/SuperEdit/static/images/guideline.png)
We find that different timesteps play distinct roles in image generation for text-to-image diffusion models, regardless of the editing instructions. Specifically, diffusion models focus on (a) global layout in the early stages, (b) local object attributes in the mid stages, (c) image details in the late stages, and the (c) image style across all stages of sampling. This finding inspires us to guide VLMs based on these four generation attributes, establishing a unified rectification method for various editing instructions.

## Editing Instruction Rectification & Training Pipeline
![](https://liming-ai.github.io/SuperEdit/static/images/rectification.png)
(a) Existing work primarily uses LLMs and diffusion models to automatically generate edited images. However, current diffusion models often fail to accurately follow text prompts while maintaining the input image's layout, resulting in mismatches between the original-edited image pairs and the editing instructions. (b) We perform instruction rectification (Step 3) based on the images constructed in Steps 1 and 2. We show VLMs can understand the differences between the images, enabling them to rectify editing instructions to be better aligned with original-edited image pairs.

![](https://liming-ai.github.io/SuperEdit/static/images/training_pipeline.png)
(a) Based on the rectified editing instruction and original-edited image pair, we utilize the Vision-Language Models (VLM) to generate various image-related wrong instructions. These involve random substitutions of quantities, spatial locations, and objects within the rectified editing instructions according to the original-edited images context; (b) During each training iteration, we randomly select one wrong instruction \(c_{neg}^T\) and input it along with the rectified instruction \(c_{pos}^T\) into the editing model to obtain predicted noises. The goal is to make the rectified instruction's predicted noise \(\epsilon_{pos}\) closer to the sampled training diffusion noise \(\epsilon\), and ensure the noise from incorrect instructions \(\epsilon_{neg}\) is farther.

## Better Performance with GPT-4o and Human Evaluation
![](https://liming-ai.github.io/SuperEdit/static/images/comparison.png)
Compared with existing methods, our SuperEdit achieves better editing results with less training data and model sizes, both in Real-Edit Benchmark with GPT-4o (Table 1) and human evaluation (Figure 7).

# License
This project is licensed under LICENSE-Apache. See the `LICENSE-Apache` flie for details.

# Citation
If you find SuperEdit useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex
@Article{SuperEdit,
      title={SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing},
      author={Ming Li, Xin Gu, Fan Chen, Xiaoying Xing, Longyin Wen, Chen Chen, Sijie Zhu},
      year={2025},
      eprint={2505.02370},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.02370},
}
@inproceedings{MultiReward,
  title={Multi-Reward as Condition for Instruction-based Image Editing},
  author={Gu, Xin and Li, Ming and Zhang, Libo and Chen, Fan and Wen, Longyin and Luo, Tiejian and Zhu, Sijie},
  booktitle={The Thirteenth International Conference on Learning Representations}
  year={2025},
}
```