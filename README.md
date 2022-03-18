# Single Image Dehazing With Consistent Contrast-Assisted Reconstruction Loss

## Network Architecture
![Network Architecture](./img/framework.png)

## Dependencies
- Python >= 3.6
- PyTorch >= 1.0
- NVIDIA GPU+CUDA

## Introduction
- **main.py** and **test.py** are the entry codes for training and testing, respectively.
- **data_utils.py** is used to load the training and testing data.
- **metrics.py** contains image quality evaluation metrics, i.e., PSNR and SSIM.
- **option.py** contains all the options.
- **utils.py** provides all the required utilities.
- **./models/model.py** defines the configurations of our proposed model.
- **./losses/ContrastLoss.py** defines the contrastive loss.

## Datasets
Create folder "data" and download the datasets into "data" folder. The datasets are available via the following links.
- [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/)
- [I-Haze](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/)
- [O-Haze](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/)
- [Dense-Haze](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/)

##### Note: The above datasets can be used directly in our method without the specific pre-processing.

## Pre-trained Models
- [Google Drive](https://drive.google.com/drive/folders/19Ot3OG8MYyuUDXI7gn3sRaE-gWpGQV7o?usp=sharing)

## Quick Start
### Train
```
python main.py
```
### Test
```
python test.py
```

## Performance
<details>
  <summary>Table 1: Quantitative comparisons with SOTA methods on SOTS-Indoor and SOTS-Outdoor synthetic datasets.</summary>
  <img src="./img/SOTS.png"/>
</details>

<details>
  <summary>Table 2: Quantitative comparisons with SOTA methods on I-Haze, O-Haze and Dense-Haze real-world datasets.</summary>
  <img src="./img/I-O-Dense.png"/>
</details>

<details>
  <summary>Table 3: Ablation study of the proposed method with different components on SOTS-Indoor and Dense-Haze datasets.</summary>
  <img src="./img/loss.png"/>
</details>

<details>
  <summary>Table 4: Parameter analysis on ITS dataset.</summary>
  <img src="./img/param.png"/>
</details>

<details>
  <summary>Figure 2: Visual comparisons on SOTS-Indoor and SOTS-Outdoor datasets for different methods.</summary>
  <img src="./img/SOTS_img.png"/>
</details>

<details>
  <summary>Figure 3: The dehazed results of different methods on the image with different hazy intensities.</summary>
  <img src="./img/haze_level.png"/>
</details>




