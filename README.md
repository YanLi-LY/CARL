# Single Image DehazingWith Consistent Contrast-Assisted Reconstruction Loss

## Network Architecture
![Network Architecture](https://github.com/YanLi-LY/images/blob/main/framework.png)

## Dependencies
- Python>=3.6
- PyTorch>=1.0
- NVIDIA GPU+CUDA

## Datasets
- [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/)
- [I-Haze](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/)
- [O-Haze](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/)
- [Dense-Haze](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/)
##### Note: The above datasets can be used directly in our method without specific pre-processing.

## Pre-trained Models
- [Google Drive](https://drive.google.com/drive/folders/19Ot3OG8MYyuUDXI7gn3sRaE-gWpGQV7o?usp=sharing)

## Usage
### Train
```
python main.py
```
### Test
```
python test.py
```

## Performance

