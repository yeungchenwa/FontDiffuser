<div align=center>

# FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning

</div>

![FontDiffuser_LOGO](figures/logo.png)  

<p align="center">
    <a href='111'><img src='https://img.shields.io/badge/Arxiv-2312.98527-red'>
    <a href='https://github.com/yeungchenwa/FontDiffuser'><img src='https://img.shields.io/badge/Code-aka.ms/fontdiffuser-yellow'>
    <!-- </a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TextDiffuser-blue)](https://huggingface.co/spaces/JingyeChen22/TextDiffuser) -->
    <a href=''><img src='https://img.shields.io/badge/GoogleColab-link-purple'>    
</p>


<p align="center">
   <strong><a href="#🔥-model-zoo">🔥 Model Zoo </a></strong> •
   <strong><a href="#🛠️-installation">🛠️ Installation </a></strong> •
   <strong><a href="#🏋️-training">🏋️ Training</a></strong> •
   <strong><a href="#📺-sampling">📺 Sampling</a></strong> •
   <strong><a href="#📱-run-webui">📱 Run WebUI</a></strong>   
</p>

## 🌟 Highlights
![Vis_1](figures/vis_1.png)
+ We propose **FontDiffuser**, which is capable to generate unseen characters and styles, and it can be extended to the cross-lingual generation, such as Chinese to Korean.
+ **FontDiffuser** excels in generating complex character and handling large style variation. And it achieves state-of-the-art performance. 
+ We release the 💻[Gradio Demo]() in Hugging Face.  

## 📅 News
- **2023.12.16**: Our demo is combined with InstructPix2Pix and ControlNet.  
- **2023.12.16**: The gradio app demo is realeased.  
<img src="figures/gradio_fontdiffuer.png" width="40%" height="auto">
- **2023.12.10**: 🔥 Release source code with phase 1 training and sampling.    
- **2023.12.09**: 🎉 Our paper is accepted by AAAI2024.

## 🔥 Model Zoo
| **Model**                                    | **chekcpoint** | **status** |
|----------------------------------------------|----------------|------------|
| **FontDiffuer**                              | [GoogleDrive](https://drive.google.com/drive/folders/12hfuZ9MQvXqcteNuz7JQ2B_mUcTr-5jZ?usp=drive_link) / [BaiduYun:gexg](https://pan.baidu.com/s/19t1B7le8x8L2yFGaOvyyBQ) | Released  |
| **SCR**                                      | - | Coming Soon           |
| **FontDiffuer (trained by a large dataset)** | - | May Be Coming |

## 🚧 TODO List
- [x] Add phase 1 training and sampling script.
- [x] Add WebUI demo.
- [ ] Push demo to Hugging Face.
- [ ] Add phase 2 training script and checkpoint.
- [ ] Add the pre-training of SCR module.

## 🛠️ Installation
### Prerequisites (Recommended)
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
### Data Contruction
The training data files tree should be (The data examples are listed in directory `data_examples/train/`):
```
├──data_examples
│   └── train
│       ├── ContentImage
│       │   ├── char0.png
│       │   ├── char1.png
│       │   ├── char2.png
│       │   └── ...
│       └── TargetImage.png
│           ├── style0
│           │     ├──style0+char0.png
│           │     ├──style0+char1.png
│           │     └── ...
│           ├── style1
│           │     ├──style1+char0.png
│           │     ├──style1+char1.png
│           │     └── ...
│           ├── style2
│           │     ├──style2+char0.png
│           │     ├──style2+char1.png
│           │     └── ...
│           └── ...
```
### Training - Phase 1
```bash
sh train_phase_1.sh
```
- `data_root`

### Training - Phase 2
```bash
Coming Soon...
```

## 📺 Sampling
### Step 1 => Prepare the checkpoint   
Option (1) Download the checkpoint following:
or (2) Put your checkpoint to the 

### Step 2 => Run the script  
(1) Sampling image from content image.  
```bash
sh script/sample_content_image.sh
```
(2) Sampling image from content character.  
**Note** Maybe you need a ttf file that contains numerous Chinese characters, you can download it from here [BaiduYun:wrth](https://pan.baidu.com/s/1LhcXG4tPcso9BLaUzU6KtQ).
```bash
sh script/sample_content_character.sh
```

## 📱 Run WebUI
### (1) Sampling by FontDiffuser
```bash
gradio gradio_app.py
```
**Example**:   
<p align="center">
<img src="figures/gradio_fontdiffuer.png" width="40%" height="auto">
</p>

### (2) Sampling by FontDiffuser and Rendering by ControlNet
```bash
gradio gradio_app_controlnet.py
```

## 🌄 Gallery
coming sonn ...

## ⭐ Star Rising
[![Star Rising](https://api.star-history.com/svg?repos=yeungchenwa/FontDiffuser&type=Timeline)](https://star-history.com/#yeungchenwa/FontDiffuser&Timeline)
