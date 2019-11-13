# Employing Attention Based Learning for Medical Image Segmentation

This repository contains my MSc Machine Learning thesis work titled 'Employing Attention Based Learning for Medical Image Segmentation'. 

## U-Net and Attention U-Net 

This thesis concerns work on [U-Net](https://arxiv.org/abs/1505.04597) and [Attention U-Net](https://arxiv.org/abs/1804.03999) as applied in 2D and 3D lung segmentation tasks.

### 2D Lung Segmentation

U-Net and 2 variants of the Attention U-Net are applied in digital X-Ray images for lung segmentation. For the latter, we apply visual attention by either using a single gating signal for all the attention gates extracted by the coarsest convolution block of the left/downsampling path of the original U-Net, or by using several gating signals extracted from the preceding block of each upsampling block on the right path of the original U-Net. For all cases, k-fold cross-validation is applied.

Representative results: 

|Model                         | k-fold  | DSC % |
|:-----------------------------|:--------|:------| 
| U-Net                        | 5       | 97.71 |        
|                              | 10      | 97.78 |        
|                              | 20      | 97.17 |        
| Single-Gating Attention U-Net| 5       | 97.71 |        
|                              | 10      | 97.77 |        
| Multi-Gating Attention U-Net | 5       | 97.84 |        
|                              | 10      | 97.79 |        

### 3D Lung Segmentation

Using volumetric CT scan data from the [LUNA16](https://luna16.grand-challenge.org) challenge, we perform important preprocessing steps (more information can be found on the report) and apply a 3D U-Net and a 3D Multi-Gating Attention U-Net for lung segmentation. k-fold cross validation is also performed at this case.  

Representative results: 

|Model               | k-fold  | DSC % |
|:-------------------|:--------|:------| 
| 3D U-Net           | 5       | 94.42 |        
|                    | 10      | 94.45 |        
| 3D Attention U-Net | 5       | 94.61 |        
|                    | 10      | 94.56 |        


## Training 

There are separate files (```train.py``` and ```trainLuna.py```) for the 2D and 3D models and data respectively.

You can also use configuration files where you can manipulate a few hyperparameters for training, and can manipulate the code to add your own. 
  
Train a 2D U-Net:
```
python train.py --config configs/UNet2D.json
```

Train a 2D single-gated U-Net:
```
python train.py --config configs/AttentionUnetSingleLUNA16.json
```

Train a 2D multi-gated U-Net:
```
python train.py --config configs/AttentionUnetMulti2D.json
```

Train a 3D U-Net with 5-fold cross-validation:

```
python trainLUNA16.py --config configs/UnetLUNA16_5fold.json
```

Train a 3D Attention U-Net with 10-fold cross-validation:

```
python trainLUNA16.py --config configs/AttentionUnetMultiLUNA16.json
```

## Thesis Report 

More details are included in my [Thesis](https://www.dropbox.com/s/a1w4665466xi13r/KTH%20Master%20Thesis%20Report%20Alexandros%20Ferles.pdf?dl=0) report.
