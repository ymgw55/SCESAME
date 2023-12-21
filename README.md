# SCESAME

![assets/scesame.png](assets/scesame.png)

> [Zero-Shot Edge Detection with SCESAME: Spectral Clustering-based Ensemble for Segment Anything Model Estimation](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/html/Yamagiwa_Zero-Shot_Edge_Detection_With_SCESAME_Spectral_Clustering-Based_Ensemble_for_Segment_WACVW_2024_paper.html)                 
> [Hiroaki Yamagiwa](https://ymgw55.github.io/), Yusuke Takase, Hiroyuki Kambe, [Ryosuke Nakamoto](https://www.let.media.kyoto-u.ac.jp/en/member/ryosuke-nakamoto/)                
> *WACV 2024 Workshop*

This paper proposes a novel zero-shot edge detection with SCESAME, which stands for Spectral Clustering-based Ensemble for Segment Anything Model Estimation, based on the recently proposed Segment Anything Model (SAM). SAM is a foundation model for segmentation tasks, and one of the interesting applications of SAM is Automatic Mask Generation (AMG), which generates zero-shot segmentation masks of an entire image. AMG can be applied to edge detection, but suffers from the problem of overdetecting edges. Edge detection with SCESAME overcomes this problem by three steps: (1) eliminating small generated masks, (2) combining masks by spectral clustering, taking into account mask positions and overlaps, and (3) removing artifacts after edge detection. 

# Code
The source code is being organized and will be available soon. 


# Citation
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