# 3D-DeepBox 

This project is unofficial implementation of 3D Bounding Box Estimation Using Deep Learning and Geometry using tensorflow 2.0
- [3D Bounding Box Estimation Using Deep Learning and Geometry](#3d-deepbox)

  * [Problem Statement](#problem-statement)
  * [Research Paper Summary](#research-paper-summary)
  * [Data](#data)
  * [Environment Setup](#environment-setup)
  * [Implementation](#implementation)
  * [Results](#results)
  * [Test Results](#test-results)
  * [References](#references)
  
  
## Problem Statement

This objective of this project is to estimate 3d bounding box and pose using single image. This project will focus on 3d bbox estimation of these classes :

 1. Car
 2. Truck
 3. Van
 4. Tram
 5. Pedestrian
 6. Cyclist
 
## Research Paper Summary 


## Data

The model has been trained on [KITTI](http://www.cvlibs.net/datasets/kitti/) Dataset . The dataset can be downloaded from the official website. 
The dataset should be stored in following structure :
```
--kitti
  |__training
          |__image_2          
          |__label_2          
  |__validation
          |__image_2
          |__label_2
 ```
          
## Environment Setup

To setup the environment download the [anaconda](https://www.anaconda.com/) . 
Clone this repo.

```
git clone https://github.com/vinayver198/3D-DeepBox.git
cd 3D-DeepBox
conda env create -f environment.yml
```
## Implementation

## Results
## Test Results
## References

1. [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/pdf/1612.00496v2.pdf)
