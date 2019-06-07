# ArchSegmentation

This is my architecture for semantic segmentation, you may reference the guide part to get start with this project.The proposed model  EAFPN adds more attention on the boundary of objects.My experiments show that this model is able to improve the segmentation result and suitable for man-made objects with straight boundary.You may reference to the inference result of my model to see the difference.

**Note:**This repo is  copy of EAFPN(without model implementation,the original private repo will be released after my paper released)

#### results





#### environment

- pytorch 1.0.0
- tensorflow
- tensorboard
- tensorboardX
- opencv

#### performance

**RoadSign (Non open dataset)**

| Methods | Backbone  | TrainSet | EvalSet | Mean Accuracy | Mean IoU |
| ------- | --------- | -------- | ------- | ------------- | -------- |
| FPN     | resnet-50 | *train*  | *val*   |               | %        |
| EAFPN   | resnet-50 | *train*  | *val*   | 95.285%       | 93.046%  |
|         |           |          |         |               |          |

**Inria Aerial Image dataset:**

The training set contains 180 color image tiles of size 5000×5000, covering a surface of 1500 m × 1500 m each (at a 30 cm resolution).  The test set contains the same amount of tiles as the training set (but the reference data is not disclosed).  

I crop the training data to 500*500 for training(), and 1000 500\*500 images for validate.All experiments's results are listed as below.

| Methods | Backbone    | TrainSet | EvalSet | Mean Accuracy | Mean IoU(msf) |
| ------- | ----------- | -------- | ------- | ------------- | ------------- |
| FPN     | resnet-101  | *train*  | *val*   | 89.60%        | 83.55%        |
| EAFPN   | resnet-101  | *train*  | *val*   |               |               |
| EAFPN   | ResNext 101 | *train*  | val     | 91.92%        | 86.91%        |
|         |             |          |         |               |               |

