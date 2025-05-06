<div align="center">
 👋 Hi, everyone!
    <br>
    We are <b>ByteDance Intelligent Creation team.</b>
</div>


# SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing
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

We are extremely delighted to release SuperEdit. SuperEdit achieves state-of-the-art image editing performance by improving the supervision quality. Our method does not require extra VLM modules or pre-training tasks used in previous work, offering a more direct and efficient way to provide better supervision signals, and providing a novel, simple, and effective solution for instruction-based image editing.

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

# License
This project is licensed under LICENSE-CC-BY-NC. See the `LICENSE-CC-BY-NC` flie for details.

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