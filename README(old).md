# Patch-sampled contrastive learning for dense prediction pretraining in metallographic images
This is the source code of "Patch-sampled contrastive learning for dense prediction pretraining in metallographic images". 

### Introduction:

The shapes, sizes, and distributions of microstructures determine the mechanical properties of alloy products, necessitating identification of microstructures in metallographic images during the manufacturing process. Previous studies demonstrated the success of deep learning in analyzing such images. However, existing methods do not fully consider the high cost of annotations, which limits the application of deep learning-based methods. Considering the popularity of self-supervised learning for natural images, further research is required to design a microstructure-specific pretraining framework. To this end, we propose a novel patch-sampled contrastive learning (PSCL) method, considering the characteristics of metallographic image segmentation tasks. First, image- and patch-level contrastive learning frameworks are designed to help the model effectively capture visual features. The former captures global features, whereas the latter captures local features in greater detail. Second, a multiscale strategy is introduced to improve the adaptability of the subsequent microstructure segmentation. Finally, considering that patches may concurrently contain multiple microstructures, a sampling method based on feature similarity was designed to capture the discriminable features of different categories. In experiments, after model fine-tuning of only one annotated image, the proposed method achieved a Dice score of 0.6296 which is higher than those of existing self-supervised learning methods with identical model structure, proving the effectiveness of the method in microstructure segmentation.


### Train and Evaluation
1. Clone this repository to local.

2. Download our Aluminum alloy microstructure image dataset from the link we provided https://pan.baidu.com/s/18fBqMlGDj1s3bQGm41yycg?pwd=845d.

3. Extract t[requirements.txt](requirements.txt)he compressed file to the dataset folder.

4. Execute the training file train.py until the termination condition is reached (By default, the pre-training and fine-tuning stages are executed sequentially).
   (Option[requirements.txt](requirements.txt)) Adjust hyperparameters "global_local_ratio" to control the weights of image- and patch-level contrastive learning
   (Option) Select different supervised samples to perform multiple repeated experiments by setting "multi_round" to true.

During and after training, the predictions and checkpoints are saved and the "log_xxx" is constructed for recording losses and performances.

### Dataset
The dataset is a series of preheating-treatment microstructural images with different holding times were obtained using a metallographic microscope. 
In total, 198 metallographic images of size 376 ×376 were collected in this study. 
Among them, there were 66 images with holding temperatures of 15/30 min, 60 with holding temperatures of 45/60 min, and 72 with holding temperatures of 90/120 min. 
**This dataset is derived from production practice and is publicly available for academic exchange purposes. It cannot be used for commercial purposes**

### Results on Metallographic Dataset
| Method                 | Dice(Mean) | Dice(Std) | ACC(Mean) | ACC(Std) | 
|:-----------------------|:----------:|:---------:|:---------:|:--------:| 
| Baseline               |   0.5134   |  0.0609   |  0.9167   |  0.0373  |
| MoCo                   |   0.5696   |  0.0683   |  0.9448   |  0.0181  | 
| SimCLR                 |   0.5712   |  0.0301   |  0.9329   |  0.0368  | 
| BYOL                   |   0.5437   |  0.0743   |  0.9339   |  0.0232  | 
| DenseCL                |   0.5622   |  0.0377   |  0.9356   |  0.0227  | 
| PSCL-MoCo (proposed)   |   0.6284   |  0.0316   |  0.9546   |  0.0108  | 
| PSCL-SimCLR (proposed) |   0.6296   |  0.0522   |  0.9574   |  0.0086  | 

### Note
When performing contrastive learning pre-training, 
supervised samples in extreme cases 
(with only one sample and imbalanced extreme categories) may affect the generality of the method. 
Future work focuses on improving the robustness of methods

### Final models
This is the pre-trained model and log file in our paper. We used this model for fine-turning and evaluation. You can download by:
https://pan.baidu.com/s/19cvVEM4Bz-i4gHZrnQA5tw?pwd=bxm8 code: bxm8.


### References
[1] <a href="https://github.com/facebookresearch/moco">MoCo: Improved baselines with momentum contrastive learning.</a>

[2] <a href="https://github.com/google-research/simclr">SimCLR: A simple framework for contrastive learning of visual representations.</a>

[3] <a href="https://github.com/WXinlong/DenseCL">DenseCL: Dense Contrastive Learning for Self-Supervised Visual Pre-Training.</a>

