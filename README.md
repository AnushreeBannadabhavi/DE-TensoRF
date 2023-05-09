# DE-TensoRF

DE-TensoRF - a data-efficient implementation of [TensoRF] (https://apchenstu.github.io/TensoRF/). This is a course project for [CPSC533R: Computer Graphics and Computer Vision](https://www.cs.ubc.ca/~rhodin/2022_2023_CPSC_533R/).

Proposed three techniques to achieve data-efficiency: symmetry, semantic conditioning, and semantic loss. Detailed report can be found [here](report.pdf)

This work is based on [TensoRF] (https://apchenstu.github.io/TensoRF/). The original code can be found [here](https://github.com/apchenstu/TensoRF).

## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.10.1 

Install environment:
```
conda create -n TensoRF python=3.8
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```


## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
* [Forward-facing](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)



## To train DE-TensoRF:

```
python train.py --config configs/lego.txt
```

### Contributors
- [Anushree Bannadabhavi](https://github.com/AnushreeBannadabhavi)
- [Aditya Chinchure](https://github.com/aditya10)
