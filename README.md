&nbsp;

<div align="center">
<p align="center"> <img src="fig/logo.png" width="100px"> </p>

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2311.10959)
[![Youtube](https://img.shields.io/badge/Youtube-video-179bd3)](https://www.youtube.com/watch?v=oVVUaBY61eo)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/structure-aware-sparse-view-x-ray-3d/novel-view-synthesis-on-x3d)](https://paperswithcode.com/sota/novel-view-synthesis-on-x3d?p=structure-aware-sparse-view-x-ray-3d)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/structure-aware-sparse-view-x-ray-3d/low-dose-x-ray-ct-reconstruction-on-x3d)](https://paperswithcode.com/sota/low-dose-x-ray-ct-reconstruction-on-x3d?p=structure-aware-sparse-view-x-ray-3d)

<h2> A Toolbox for Sparse-View X-ray 3D Reconstruction </h2> 



<img src="3d_demo/backpack.gif" style="height:260px" /> <img src="3d_demo/box.gif" style="height:260px" />  <img src="3d_demo/bonsai.gif" style="height:260px" /> 



<img src="3d_demo/foot.gif" style="height:230px" />  &emsp; &emsp; <img src="3d_demo/teapot.gif" style="height:230px" /> <img src="3d_demo/engine.gif" style="height:230px" />


</div>


&nbsp;

### News
- **2024.06.02 :** Data, code, models, and training logs have been realeased. Feel free to use them :)
- **2024.02.26 :** Our paper has been accepted by CVPR 2024. Code and pre-trained models will be released to the public before the start date of CVPR 2024 (2024.06.19). Stay tuned! :tada: :confetti_ball:
- **2023.11.21 :** The benchmark of X3D at the [paper-with-code website](https://paperswithcode.com/dataset/x3d) has been set up. You are welcome to make a comparison. 🚀
- **2023.11.21 :** Our paper is on [arxiv](https://arxiv.org/abs/2311.10959) now. We will develop this repo into a baseline for X-ray novel view synthesis and CT reconstruction. All code, models, data, and training logs will be released. 💫

### Performance

<details close>
<summary><b>Novel View Synthesis</b></summary>

![results1](/fig/nvs_1.png)

![results2](/fig/nvs_2.png)

</details>


<details close>
<summary><b>CT Reconstruction</b></summary>

![results3](/fig/ct_1.png)

![results4](/fig/ct_2.png)

</details>


This repo will support 8 state-of-the-art algorithms including 5 NeRF-based methods, 2 optimization-based methods, and 1 analytical method for sparse-view X-ray 3D reconstruction.

<details open>
<summary><b>Supported algorithms:</b></summary>

* [x] [SAX-NeRF](https://arxiv.org/abs/2311.10959) (CVPR 2024)
* [x] [TensoRF](https://arxiv.org/abs/2203.09517) (ECCV 2022)
* [x] [NAF](https://arxiv.org/abs/2209.14540) (MICCAI 2022)
* [x] [NeAT](https://arxiv.org/abs/2202.02171) (ACM TOG 2022)
* [x] [NeRF](https://arxiv.org/abs/2003.08934) (ECCV 2020)
* [x] [InTomo](https://openaccess.thecvf.com/content/ICCV2021/papers/Zang_IntraTomo_Self-Supervised_Learning-Based_Tomography_via_Sinogram_Synthesis_and_Prediction_ICCV_2021_paper.pdf) (ICCV 2021)
* [x] [SART](https://engineering.purdue.edu/RVL/Publications/SART_84.pdf) (Ultrasonic imaging 1984)
* [x] [ASD-POCS](https://www.researchgate.net/profile/Emil-Sidky/publication/23169511_Image_reconstruction_in_circular_cone-beam_computed_tomography_by_constrained_total-variation_minimization/links/0c96052408b0814590000000/Image-reconstruction-in-circular-cone-beam-computed-tomography-by-constrained-total-variation-minimization.pdf) (Physics in Medicine & Biology 2008)
* [x] [FDK](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-1-6-612&id=996) (Josa a 1984)

</details>

&nbsp;

## 1. Create Environment:

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment.

``` sh
# Create environment
conda create -n sax_nerf python=3.9
conda activate sax_nerf

# Install pytorch (hash encoder requires CUDA v11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other packages
pip install -r requirements.txt
```

We suggest you install TIGRE toolbox (2.3 version) for executing traditional CT reconstruction methods and synthesize your own CT data if you plan to do so. Please note that TIGRE v2.5 might stuck when CT resolution is large.
``` sh
# Download TIGRE
wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip
unzip v2.3.zip
rm v2.3.zip

# Install TIGRE
pip install cython==0.29.25
pip install numpy==1.21.6
cd TIGRE-2.3/Python/
python setup.py develop
```

&nbsp;

## 2. Prepare Dataset:

Download our processed datasets from [Google drive](https://drive.google.com/drive/folders/1SlneuSGkhk0nvwPjxxnpBCO59XhjGGJX?usp=sharing) or Baidu disk. Then put the downloaded datasets into the folder `data/` as

```sh
  |--data
      |--chest_50.pickle
      |--abdomen_50.pickle
      |--aneurism_50.pickle
      |--backpack_50.pickle
      |--bonsai_50.pickle
      |--box_50.pickle
      |--carp_50.pickle
      |--engine_50.pickle
      |--foot_50.pickle
      |--head_50.pickle
      |--leg_50.pickle
      |--pancreas_50.pickle
      |--pelvis_50.pickle
      |--teapot_50.pickle
      |--jaw_50.pickle
```

&nbsp;

## 3. Testing:

You can directly download our pre-trained models from [Google drive](https://drive.google.com/drive/folders/1wlDrZQRbQENcfW1Pjrr1gasFQ8v6znHV?usp=sharing) or Baidu disk. Then put the downloaded models into the folder `pretrained/` and run

```sh
# SAX-NeRF
python test.py --method Lineformer --category chest --config config/Lineformer/chest_50.yaml --weights pretrained/chest.tar --output_path output 

# FDK
python3 eval_traditional.py --algorithm fdk --category chest --config config/FDK/chest_50.yaml

# SART
python3 eval_traditional.py --algorithm sart --category chest --config config/SART/chest_50.yaml

# ASD_POCS
python3 eval_traditional.py --algorithm asd_pocs --category chest --config config/ASD_POCS/chest_50.yaml
```

&nbsp;

## 4. Training:

We provide the training logs on all scenes for your convenience to debug. Please download the training logs from [Google dive](https://drive.google.com/drive/folders/123WISBBc3rjfKqZ1EGK0-2sW5TY5dkLI?usp=sharing) of Baidu disk.

```sh
# SAX-NeRF
python train_mlg.py --config config/Lineformer/chest_50.yaml

# NeRF
python train.py --config config/nerf/chest_50.yaml

# Intratomo
python train.py --config config/intratomo/chest_50.yaml

# NAF
python train.py --config config/naf/chest_50.yaml

# TensoRF
python train.py --config config/tensorf/chest_50.yaml
```

You can use [this repo](https://github.com/darglein/NeAT) to run NeAT. Remember to reprocess the data first.

&nbsp;

## 5. Visualization

To render a cool demo, we provide visualization code in the folder `3D_vis`

```sh
cd 3D_vis
python 3D_vis_backpack.py
python 3D_vis_backpack_gif.py
```

&nbsp;

## 6. Generate Your Own Data
We also provide code for data generation in the folder `dataGenerator`. Firstly, you need to install the [TIGRE](https://github.com/CERN/TIGRE) toolbox. To give a quick start, we provide a raw data for your debugging. Please down load the raw data from [Google dive](https://drive.google.com/drive/folders/1i3BhyftggTj1SqW6Ibl5tWTWD0VLc7ex?usp=sharing) or Baidu disk and then put it into the folder `dataGenerator/raw_data`. Run

```sh
cd dataGenerator
python data_vis_backpack.py
cd ..
python generateData_backpack.py
```

&nbsp;

## 7. Citation
If this repo helps you, please consider citing our works:


```sh
@inproceedings{sax_nerf,
  title={Structure-Aware Sparse-View X-ray 3D Reconstruction},
  author={Yuanhao Cai and Jiahao Wang and Alan Yuille and Zongwei Zhou and Angtian Wang},
  booktitle={CVPR},
  year={2024}
}
```
