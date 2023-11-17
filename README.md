# SCESAME

![assets/scesame.png](assets/scesame.png)

> [Zero-Shot Edge Detection with SCESAME: Spectral Clustering-based Ensemble for Segment Anything Model Estimation](https://arxiv.org/abs/2308.13779)                 
> [Hiroaki Yamagiwa](https://ymgw55.github.io/), Yusuke Takase, Hiroyuki Kambe, [Ryosuke Nakamoto](https://www.let.media.kyoto-u.ac.jp/en/member/ryosuke-nakamoto/)                
> *WACV 2024 Workshop*

This paper proposes a novel zero-shot edge detection with SCESAME, which stands for Spectral Clustering-based Ensemble for Segment Anything Model Estimation, based on the recently proposed Segment Anything Model (SAM). SAM is a foundation model for segmentation tasks, and one of the interesting applications of SAM is Automatic Mask Generation (AMG), which generates zero-shot segmentation masks of an entire image. AMG can be applied to edge detection, but suffers from the problem of overdetecting edges. Edge detection with SCESAME overcomes this problem by three steps: (1) eliminating small generated masks, (2) combining masks by spectral clustering, taking into account mask positions and overlaps, and (3) removing artifacts after edge detection. 

# Code
The source code is being organized and will be available soon. 


# Citation
If you find our code or model useful in your research, please cite our paper:
```
@misc{yamagiwa2023zeroshot,
      title={Zero-Shot Edge Detection with SCESAME: Spectral Clustering-based Ensemble for Segment Anything Model Estimation}, 
      author={Hiroaki Yamagiwa and Yusuke Takase and Hiroyuki Kambe and Ryosuke Nakamoto},
      year={2023},
      eprint={2308.13779},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```