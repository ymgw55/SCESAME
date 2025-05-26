# SCESAME

![assets/scesame.png](assets/scesame.png)

> [Zero-Shot Edge Detection with SCESAME: Spectral Clustering-based Ensemble for Segment Anything Model Estimation](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/html/Yamagiwa_Zero-Shot_Edge_Detection_With_SCESAME_Spectral_Clustering-Based_Ensemble_for_Segment_WACVW_2024_paper.html)                 
> [Hiroaki Yamagiwa](https://ymgw55.github.io/), Yusuke Takase, Hiroyuki Kambe, [Ryosuke Nakamoto](https://www.let.media.kyoto-u.ac.jp/en/member/ryosuke-nakamoto/)                
> *WACV 2024 Workshop*

This paper proposes a novel zero-shot edge detection with SCESAME, which stands for Spectral Clustering-based Ensemble for Segment Anything Model Estimation, based on the recently proposed Segment Anything Model (SAM) [1]. SAM is a foundation model for segmentation tasks, and one of the interesting applications of SAM is Automatic Mask Generation (AMG), which generates zero-shot segmentation masks of an entire image. AMG can be applied to edge detection, but suffers from the problem of overdetecting edges. Edge detection with SCESAME overcomes this problem by three steps: (1) eliminating small generated masks, (2) combining masks by spectral clustering, taking into account mask positions and overlaps, and (3) removing artifacts after edge detection. 

---

## Docker

This repository is intended to be run in a Docker environment. If you are not familiar with Docker, please install the packages listed in [requirements.txt](requirements.txt).

### Docker build

Create a Docker image as follows:

```bash
$ bash script/docker/build.sh
```

### Docker run

Run the Docker container by passing the GPU ID as an argument:
```bash
$ bash script/docker/run.sh 0
```

---

## Data

### BSDS500
download BSDS500 [2] dataset from [official site](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).

If you cannot download it, the following mirror repositories may be helpful.
- https://github.com/BIDS/BSDS500

Then prepare the following directory structure:

```bash
data/BSDS500/
    ├── groundTruth
    │   └── test
    │       ├── 100007.mat
    │       ├── 100039.mat
    │       ...
    │       
    └── images
        ├── test
        │   ├── 100007.jpg
        │   ├── 100039.jpg
        │   ...
        │
        ├── train
        └── val
```

### NYUDv2

download NYUDv2 [3] test dataset from [EDTER](https://github.com/MengyangPu/EDTER).
Then prepare the following directory structure:

```bash
data/NYUDv2/
    ├── groundTruth
    │   └── test
    │       ├── img_5001.mat
    │       ├── img_5002.mat
    │       ...
    │       
    └── images
        ├── test
        │   ├── img_5001.png
        │   ├── img_5002.png
        │   ...
        │
        ├── train
        └── val
```

---

## Model

Create a directory to download the model as follows:

```bash
mkdir model
```

### SAM

Download the SAM model as follows:

```bash
wget -P model https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Edge-NMS

In the original paper [1], Canny edge NMS [4] was used for edge NMS.
However, in our environment, it did not produce the edges reported in the paper. 
Therefore, we temporarily used OpenCV's Structured Forests [5] model for edge NMS.

Download the Structured Forests model as follows:

```bash
wget -P model https://cdn.rawgit.com/opencv/opencv_extra/3.3.0/testdata/cv/ximgproc/model.yml.gz
```

---

## Prediction

Predict edges as follows:

```bash
python scesame_pipeline.py --t 3 --c 2 --p 5 --dataset BSDS500
```

Here
- `--t`: The variable $t$ for Top Mask Selection (TMS).
- `--c`: The variable $c$ for Spectral Clustering (SC).
- `--p`: The variable $p$ for Boundary Zero Padding (BZP).
- `--dataset`: Choose `BSDS500` or `NYUDv2`.


If $t$, $c$, or $p$ is not given, the setting is ablated.
In particular, if all of $t$, $c$, and $p$ are not given, it corresponds to AMG.

Additionally, the kernel size of the Gaussian blur for edge detection and the $\tau$ used for the similarity matrix in SC can be specified by `--kernel_size` and `--tau`.

---

## Evaluation

We use [py-bsds500](https://github.com/Britefury/py-bsds500/tree/master) for edge detection. Some bugs have been fixed and ported to the `py-bsds500` directory.
Compile the extension module with:
```bash
cd py-bsds500
python setup.py build_ext --inplace
```

Then evaluate ODS, OIS, and AP as follows:

```bash
python evaluate_parallel.py ../data/BSDS500 ../output/BSDS500/pred/t3_c2_p5_ks3_tau0.5/ test --max_dist 0.0075
python evaluate_parallel.py ../data/NYUDv2 ../output/NYUDv2/pred/t3_c2_p5_ks3_tau0.5/ test --max_dist 0.011
```
Note that following previous works, the localization tolerance is set to 0.0075 for BSDS500 and 0.011 for NYUDv2.

---

## References

### Code

We used publicly available repositories. We are especially grateful for the following repositories. Thank you.

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [py-bsds500](https://github.com/Britefury/py-bsds500)
- [segment-anything-edge-detection](https://github.com/ymgw55/segment-anything-edge-detection)
- [opencv_contrib](https://github.com/opencv/opencv_contrib)
- [plot-edge-pr-curves](https://github.com/MCG-NKU/plot-edge-pr-curves)
- [EDTER](https://github.com/MengyangPu/EDTER)

### Paper

[1] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick. Segment Anything. ICCV 2023.

[2] Pablo Arbelaez, Michael Maire, Charless C. Fowlkes, and Jitendra Malik. Contour detection and hierarchical image segmentation. IEEE Trans. Pattern Anal. Mach. Intell 2011.

[3] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from RGBD images. ECCV 2012.

[4] John F. Canny. A computational approach to edge detection. IEEE Trans. Pattern Anal. Mach. Intell 1986.

[5] Piotr Dollar and C. Lawrence Zitnick. Fast edge detection using structured forests. IEEE Trans. Pattern Anal. Mach. Intell 2015.

---

## Citation
If you find our code or model useful in your research, please cite our paper:
```
@InProceedings{Yamagiwa_2024_WACV,
    author    = {Yamagiwa, Hiroaki and Takase, Yusuke and Kambe, Hiroyuki and Nakamoto, Ryosuke},
    title     = {Zero-Shot Edge Detection With SCESAME: Spectral Clustering-Based Ensemble for Segment Anything Model Estimation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2024},
    pages     = {541-551}
}
```