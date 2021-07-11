## Table of Contents


<!-- MarkdownTOC depth=4 -->
- [Prerequisites](#prerequisites)
- [Usage Guide](#usage-guide)
- [To Do Left](#to-do-left)
 
## Prerequisites
- Create python3.6 environment and install the required packages

- Installation Guide
  ```bash
  pip install -r requirements.txt
  ```
  
 ## Usage Guide

### Training
This is the training stage of the project.
**Steps**
1. First download xception weights using below command:
```bash
  wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5 -P pretrained_weights/
```
2. Then move folder containing images and their label csv files to `data` directory

**Usage:**
```bash
python train.py --data_dir data/ \
                --ckpt_dir ckpt/ \
                 --imagenet_weights pretrained_weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5 \
                 --test_split 0.1 \
                 --random_state 30 \
                 --num_epochs 3000 \
                 --batch_size 64 \
                 --lr 0.0001 \
                 --stoploss 0.6 \
                 --restore_model=True
```
### Testing
This module is used to test the performance of the trained model on test dataset generated above.

**Usage:**
```bash
python train.py --data_dir data/ \
                --ckpt_dir ckpt/ \
                 --imagenet_weights pretrained_weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5 \
                 --batch_size 16
```
### Custom Testing
This module is used to test the performance of trained model on any general dataset which can be of different distribution also.
**Steps**
1. move all the images to `custom_data/test_images` directory and after model get run `output.csv` will be generated in `custom_data` directory

**Usage:**
```bash
python train.py --test_data_dir custom_data/test_images/ \
                --output_dir custom_data/ \
                --ckpt_dir ckpt/
```
  
## To Do Left
1. Adding proper callbacks to make training process efficient.
2. Since it is a single neural net for all the 3 independent classes so model will take a lot time to converge although implemented loss function works good but takes time. Some more advanced loss function can be used.
