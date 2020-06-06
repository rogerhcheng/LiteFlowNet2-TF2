# LiteFlowNet2-TF2


This is my TensorFlow 2 implementation of LiteFlowNet2 [1] (an improved version of the original LiteFlowNet [2]).

I used this implementation of the original LiteFlowNet [3] as my starting point: https://github.com/keeper121/liteflownet-tf2  
That implementation was inspired by this LiteFlowNet [4] PyTorch implemetation: https://github.com/sniklaus/pytorch-liteflownet  
The original Caffe implementation of LiteFlowNet2 is here: https://github.com/twhui/LiteFlowNet2

Please cite the paper accordingly and make sure to adhere to the licensing terms of the authors.
Should you be making use of this particular implementation, please acknowledge it appropriately [5].

<a href="https://arxiv.org/abs/1903.07414" rel="Paper"><img src="http://www.arxiv-sanity.com/static/thumbs/1903.07414v3.pdf.jpg" alt="Paper" width="100%"></a>


## Setup
The correlation layer is using the tensorflow_addons package which requires TF 2. I tested with these combinations of packages installed via pip:

```
pip install --user tensorflow==2.1.0
pip install --user tensorflow_addons==0.7.0
```
and
```
pip install --user tensorflow==2.2.0
pip install --user tensorflow_addons==0.10.0
```

The included weights were converted from the original Caffe implementation [1]. Note that the Sintel and Kitti fine-tuned models have slightly different model architectures.


## How to run
To run it on a pair of images, execute the following command to run the Sintel model:
```
python eval.py --img1=./images/first.png --img2=./images/second.png --use_Sintel=True --display_flow=True --img_out=out.png
```
And this command for the Kitti model:
```
python eval.py --img1=./images/first.png --img2=./images/second.png --use_Sintel=False --display_flow=True --img_out=out.png
```

As you can see below, the results are very close to, but differ a little from the original Caffe implementation. The author of the original LiteFlowNet TF implementation believes it is due to a slightly different feature warping implementation than in the original work.

<p align="center"><img src="images/compare.gif?raw=true" alt="Comparison"></p>


## License
Original materials are provided for research purposes only, and commercial use requires consent of the original author. Please see https://github.com/twhui/LiteFlowNet2#license-and-citation for more information.


## References
```
[1]  @inproceedings{hui20liteflownet2,
         author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
         title = {A {L}ightweight {O}ptical {F}low {CNN} - {R}evisiting {D}ata {F}idelity and {R}egularization},
         booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
         year = {2020},
         url = {http://mmlab.ie.cuhk.edu.hk/projects/LiteFlowNet/} 
     }
```
```
[2]  @inproceedings{Hui_CVPR_2018,
         author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
         title = {{LiteFlowNet}: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018},
         url = {http://mmlab.ie.cuhk.edu.hk/projects/LiteFlowNet/} 
     }
```
```
[3]  @misc{liteflownet-tf2,
         author = {Vladimir Mikhelev},
         title = {{LiteFlowNet} inference realization with {Tensorflow 2}},
         year = {2020},
         url = {https://github.com/keeper121/liteflownet-tf2}
    }
```
```
[4]  @misc{pytorch-liteflownet,
         author = {Simon Niklaus},
         title = {A Reimplementation of {LiteFlowNet} Using {PyTorch}},
         year = {2019},
         url = {https://github.com/sniklaus/pytorch-liteflownet}
    }
```
```
[5]  @misc{LiteFlowNet2-TF2,
         author = {Roger Cheng},
         title = {{LiteFlowNet2} implementation with {TensorFlow 2}},
         year = {2020},
         url = {https://github.com/rogerhcheng/LiteFlowNet2-TF2}
    }
```
