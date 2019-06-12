# ArchSegmentation

This is my architecture for semantic segmentation.The proposed model EAFPN adds more attention on the boundary of objects during training.My experiments show that this model is able to improve the segmentation result and suitable for man-made objects with straight boundary.The advantage of EAFPN is it adds no extra computation during inferencing.You may reference to the result of my model to see the difference.

**Note:** This repo is  copy of EAFPN(without model implementation,the original private repo will be released after my paper released)

#### results

result_with_edge is the output of my proposed model EAFPN, result_without_edge is the output of original FPN."edgexx" means edges detected by net work in different stage and egde4 is the comfusion of each stage. Both model use ResNet-101 as backbone. EAFPN adds no extra computation in inferencing phase

![](https://raw.githubusercontent.com/Citygity/ArchSegmentation/master/images/00004997.png)
![](https://raw.githubusercontent.com/Citygity/ArchSegmentation/master/images/44.png)


#### environment

- pytorch 1.0.0
- tensorflow
- tensorboard
- tensorboardX
- opencv

#### performance

**RoadSign (Non open dataset)**

tested on 2988 1920\*1020 images.Our FPN model was first trained on cityscape, then finetuned on our own data collected on Beijing-Shanghai Expressway contains 24000+ 1920*1080 annotated images.

| Methods | Backbone  | TrainSet | EvalSet | Mean Accuracy | Mean IoU | back ground| roadsign |
| ------- | --------- | -------- | ------- | ------------- | -------- | ---------- | -------- |
| FPN     | resnet-50 | *train*  | *val*   | 91.37%        | 87.93%   |  99.85%    | 75.75%   |   
| FPN     | resnet-18 | *train*  | *val*   | 89.57%        | 84.16%   |  99.795%   | 68.525%  |
| EAFPN   | resnet-50 | *train*  | *val*   | 95.35%        | 93.10%   |  99.92%    | 86.285%  |
| EAFPN   | resnet-18 | *tran*   | *val*   | 95.11%        | 92.30%   |            |          |

**Inria Aerial Image dataset:**

The training set contains 180 color image tiles of size 5000×5000, covering a surface of 1500 m × 1500 m each (at a 30 cm resolution).  The test set contains the same amount of tiles as the training set (but the reference data is not disclosed).  

I crop the training data to 500*500 for training, and 1000 500\*500 images for validate.All experiments's results are listed as below.

| Methods                        | Backbone    | TrainSet | EvalSet | Mean Accuracy | Mean IoU(msf) |
| ------------------------------ | ----------- | -------- | ------- | ------------- | ------------- |
| FPN                            | ResNet-101  | *train*  | *val*   | 89.60%        | 83.55%        |
| EAFPN                          | ResNet-101  | *train*  | *val*   | 92.84%        | 87.60%        |
| EAFPN                          | ResNext 101 | *train*  | *val*   | 92.31%        | 87.14%        |
| EFPN(edge detection with ASPP) | ResNet101   | *train*  | *val*   | 93.40%        | 88.32%        |
| stcked FPN*                    | ResNet-101  | *train*  | *val*   | 96.77%        | 89.80%        |

FPN is the original model of FPN, EAFPN detects boundary to strengthen the segmentation result.

EAFPN(edge detection with ASPP) uses ASPP module to capture multi scale feature map, this model is training. Stacked FPN is a validating experiment to validate if edge detection is helpful to improve the segmentation result.
