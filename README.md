# BDCKD-Unlocking-the-Power-of-Brownian-Distance-Covariance-in-Knowledge-Distillation

## Introduction


This is the official code repository for the ICASSP 2025 research paper---BDCKD Unlocking the Power of Brownian Distance Covariance in Knowledge Distillation.
It contains the implementation of the algorithms and methods described in the paper.

## BDCKD

#### Ablation of BDCKD on CIFAR100: We set ResNet32x4 as the teacher and ShuffleNetV2 as the student.

| Method           | Inter         | Intra         | ACC (\%)  | $\Delta$ KD (\%)  | $\Delta$ Stu (\%)  |
| -------------- | ------------- | ------------- | ---------------------- | --------------------------- | ---------------------------- |
| Student        | -             | -             | 71.82                  | -                           | -                            |
| KD (Hinton et al.) | ✓         | -             | 74.45                  | -                           | +2.63                        |
| BDCKD | -       | ✓             | 76.34                  | +1.89                       | +4.52                        |
|                | ✓             | -             | 77.23                  | +2.78                       | +5.41                        |
|                | ✓             | ✓             | **78.05**              | **+3.60**                   | **+6.23**                    |

#### Comparison of various distillation methods' accuracy across same teacher-student architectures.

| Teacher         | WRN-40-2 | RN-32x4 | Vgg13 | RN-110 |
| --------------- | -------- | ------ | ---- | ----- |
| ACC                | 75.61    | 79.42  | 74.64| 74.31  |
| Student         | WRN-40-1 | RN-8x4 | Vgg8 | RN-20 |
| ACC                 | 71.98    | 72.50  | 70.36| 69.06 | 
| FitNet          | 72.24    | 73.50  | 71.02| 68.99  |
| CRD (ICLR 2020) | 74.14    | 75.51  | 73.94| 71.16  |
| RKD (CVPR 2019) | 72.22    | 75.51  | 71.48| 69.25  |
| OFD (ICCV 2019) | 74.33    | 74.95  | 73.95| 71.29  |
| KD (NIPS 2014)  | 73.54    | 73.33  | 72.98| 70.67  |
| DKD (CVPR 2022) | 74.81    | 76.32  | 74.42| 71.06  |
| DIST (NeurIPS 2022) | 74.73 | 76.31  | 73.63| 71.19  |
| CTKD (AAAI 2023) | 73.93   | 73.39  | 73.52| 70.99  |
| LSKD (CVPR 2024) | 74.37   | 76.62  | 74.36| 71.48  |
| BDCKD (Ours)    | **74.98**| **77.25** | **74.73** | **71.61** |
| $\Delta$ KD $\uparrow$ | **+1.44** | **+3.92** | **+1.75** | **+0.94** |

#### Comparison of various distillation methods' accuracy across different teacher-student architectures.
| Teacher | WRN-40-2 | RN-50 | RN-32x4 | RN-32x4 |
| --------------- | -------- | ----- | ------- | ------- |
| ACC                | 75.61    | 79.34 | 79.42   | 79.42   |
| Student         | SN-V1    | MN-V2 | SN-V1   | SN-V2   |
| ACC                | 70.50    | 64.60 | 70.50   | 71.82   |
| FitNet          | 73.73    | 63.16 | 73.54   | 73.54   |
| CRD (ICLR 2020) | 76.05    | 69.11 | 75.11   | 75.65   |
| RKD (CVPR 2019) | 72.21    | 64.43 | 73.21   | 73.21   |
| OFD (ICCV 2019) | 75.85    | 69.04 | 76.82   | 76.82   |
| KD (NIPS 2014)  | 74.83    | 67.35 | 74.07   | 74.45   |
| DKD (CVPR 2022) | 76.70    | 70.35 | 76.45   | 77.07   |
| DIST (NeurIPS 2022) | 76.12 | 68.66 | 76.34   | 77.35   |
| CTKD (AAAI 2023) | 75.78   | 68.47 | 74.48   | 75.31   |
| LSKD (CVPR 2024) | 76.33   | 69.02 | 75.33   | 75.56   |
| BDCKD (Ours)    | **76.86**| **70.72** | **77.20** | **78.05** |
| $\Delta$ KD $\uparrow$ | **+2.03** | **+3.37** | **+3.13** | **+3.60** |

#### Top1\&Top5 Accuracy on Tiny-ImageNet Dataset

| Teacher/Student | ResNet50/ResNet18 | ResNet34/MobileNetV2 |
| --------------- | ----------------- | -------------------- |
| Accuracy        | Top-1  | Top-5     | Top-1  | Top-5         |
| Teacher Acc     | 67.36  | 85.86     | 66.23  | 85.63         |
| Student Acc     | 65.13  | 84.76     | 56.32  | 80.64         |
| KD (NIPS 2014)  | 67.19  | 86.35     | 56.69  | 80.59         |
| DKD (CVPR 2022) | 67.44  | 86.54     | 61.48  | 83.39         |
| LSKD (CVPR 2024)| 67.56  | 86.88     | 59.24  | 81.79         |
| BDCKD (Ours)    | **68.51** | **87.27** | **63.21** | **84.45** |
| $\Delta$ KD $\uparrow$ | **+1.32** | **+0.92** | **+6.52** | **+1.86** |

## Getting started

1. Evaluation

- You can evaluate the performance of our models or models trained by yourself.


  ```bash
  # evaluate 
  python tools/eval.py --model <model_name> --ckpt <ckpt_path> --dataset <cifar100/tiny_imagenet> 
  ```


2. Training on CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  # For instance, our BDC method.
  python3 tools/train.py --cfg configs/cifar100/bdckd/res32x4_res8x4.yaml
  ```

3. Training on Tiny-ImageNet

- Download the dataset at <http://cs231n.stanford.edu/tiny-imagenet-200.zip> and put them to `./data/tiny-imagenet-200`

  ```bash
  # for instance, our BDC method.
  python3 tools/train.py --cfg configs/tiny_imagenet/bdc/r34_mv2.yaml
  ```

## Acknowledgement


- Thanks for mdistiller and FeiLong. We build this library based on the [mdistiller codebase](https://github.com/megvii-research/mdistiller) and the [FeiLong codebase](https://github.com/Fei-Long121/DeepBDC) 