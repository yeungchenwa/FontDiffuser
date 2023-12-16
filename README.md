<div align=center>

# FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning

</div>

[HERE LOGO].  

<p align="center">
    <a href='111'><img src='https://img.shields.io/badge/Arxiv-2312.98527-red'>
    <a href='https://github.com/yeungchenwa/FontDiffuser'><img src='https://img.shields.io/badge/Code-aka.ms/fontdiffuser-yellow'>
    <!-- </a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TextDiffuser-blue)](https://huggingface.co/spaces/JingyeChen22/TextDiffuser) -->
    <a href=''><img src='https://img.shields.io/badge/GoogleColab-link-purple'>    
</p>


<p align="center">
   <strong><a href="#🔥-Model-Zoo">🔥 Model Zoo </a></strong> •
   <strong><a href="#Installation">Installation </a></strong> •
   <strong><a href="#Training">Training</a></strong> •
   <strong><a href="#Sampling">Sampling</a></strong> •
   <strong><a href="#Run-WebUI">Run WebUI</a></strong>   
</p>

## 🌟 Highlights
[Here GIF]
+ We propose **FontDiffuser**, which is capable to generate unseen characters and styles, and it can be extended to the cross-lingual generation, such as Chinese to Korean.
+ **FontDiffuser** excels in generating complex character and handling large style variation. And it achieves state-of-the-art performance. 
+ We release the 💻[Gradio Demo]() in Hugging Face.  

## 📅 News
- **2023.12.16**: The gradio app script is realeased.
- **2023.12.10**: 🔥 Release source code with phase 1 training and sampling.
- **2023.12.09**: 🎉 Our paper is accepted by AAAI2024.

## 🔥 Model Zoo
| **Model**                                    | **chekcpoint** | **status** |
|----------------------------------------------|----------------|------------|
| **FontDiffuer**                              | [GoogleDrive]() / [BaiduYun]() / [OneDrive]() | Released  |
| **SCR**                                      | [GoogleDrive]() / [BaiduYun]() / [OneDrive]() | Coming Soon           |
| **FontDiffuer (trained by a large dataset)** | [GoogleDrive]() / [BaiduYun]() / [OneDrive]() | May Be Coming |

## 🚧 TODO List
- [x] Add phase 1 training and sampling script.
- [x] Add WebUI demo.
- [ ] Push demo to Hugging Face.
- [ ] Add phase 2 training script and checkpoint.
- [ ] Add the pre-training of SCR module.

## 🛠️ Installation
### Prerequisites(Recommended)
- Linux
- Python 3.9
- Pytorch 1.13.1
- CUDA 11.7

### Environment Setup
Clone this repo:
```bash
git clone https://github.com/yeungchenwa/FontDiffuser.git
```

**Step 0**: Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1**: Create a conda environment and activate it.
```bash
conda create -n fontdiffuser python=3.9 -y
conda activate fontdiffuser
```

**Step 2**: Install related version Pytorch following [here](https://pytorch.org/get-started/previous-versions/).
```bash
# Suggested
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

**Step 3**: Install the required packages.
```bash
pip install -r requirements.txt
```

## 🏋️ Training

## 📺 Sampling
coming soon ...

## 📱 Run WebUI
coming soon ...

## 🌄 Gallery
coming sonn ...

## ⭐ Star Rising
[![Star Rising](https://api.star-history.com/svg?repos=yeungchenwa/FontDiffuser&type=Timeline)](https://star-history.com/#yeungchenwa/FontDiffuser&Timeline)
