&nbsp;

<div align="center">
<p align="center"> <img src="fig/logo.png" width="100px"> </p>

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2311.10959)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/structure-aware-sparse-view-x-ray-3d/novel-view-synthesis-on-x3d)](https://paperswithcode.com/sota/novel-view-synthesis-on-x3d?p=structure-aware-sparse-view-x-ray-3d)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/structure-aware-sparse-view-x-ray-3d/low-dose-x-ray-ct-reconstruction-on-x3d)](https://paperswithcode.com/sota/low-dose-x-ray-ct-reconstruction-on-x3d?p=structure-aware-sparse-view-x-ray-3d)

<h2> A Toolbox for Sparse-View X-ray 3D Reconstruction </h2> 



<img src="3d_demo/backpack.gif" style="height:260px" /> <img src="3d_demo/box.gif" style="height:260px" />  <img src="3d_demo/bonsai.gif" style="height:260px" /> 



<img src="3d_demo/foot.gif" style="height:230px" />  &emsp; &emsp; <img src="3d_demo/teapot.gif" style="height:230px" /> <img src="3d_demo/engine.gif" style="height:230px" />


</div>


&nbsp;

### News
- **2023.11.21 :** The benchmark of X3D at the [paper-with-code website](https://paperswithcode.com/dataset/x3d) has been set up. You are welcome to make a comparison.
- **2023.11.21 :** Our paper is on [arxiv](https://arxiv.org/abs/2311.10959) now. We will develop this repo into a baseline for X-ray novel view synthesis and CT reconstruction. Code, models, and data will be released. 💫

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

* [ ] [SAX-NeRF](https://arxiv.org/abs/2311.10959) (Arxiv 2023)
* [ ] [TensoRF](https://arxiv.org/abs/2203.09517) (ECCV 2022)
* [ ] [NAF](https://arxiv.org/abs/2209.14540) (MICCAI 2022)
* [ ] [NeAT](https://arxiv.org/abs/2202.02171) (ACM TOG 2022)
* [ ] [NeRF](https://arxiv.org/abs/2003.08934) (ECCV 2020)
* [ ] [InTomo](https://openaccess.thecvf.com/content/ICCV2021/papers/Zang_IntraTomo_Self-Supervised_Learning-Based_Tomography_via_Sinogram_Synthesis_and_Prediction_ICCV_2021_paper.pdf) (ICCV 2021)
* [ ] [SART](https://engineering.purdue.edu/RVL/Publications/SART_84.pdf) (Ultrasonic imaging 1984)
* [ ] [ASD-POCS](https://www.researchgate.net/profile/Emil-Sidky/publication/23169511_Image_reconstruction_in_circular_cone-beam_computed_tomography_by_constrained_total-variation_minimization/links/0c96052408b0814590000000/Image-reconstruction-in-circular-cone-beam-computed-tomography-by-constrained-total-variation-minimization.pdf) (Physics in Medicine & Biology 2008)
* [ ] [FDK](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-1-6-612&id=996) (Josa a 1984)


</details>
