Below is an enhanced, "fancier" version of your README. It emphasizes the core contributions and adds some stylistic elements for a more engaging presentation.

---

# ✨ DeCo-Diff ✨
**A PyTorch Implementation for Multi-Class Unsupervised Anomaly Detection**

This repository hosts the official PyTorch implementation for our CVPR 2025 paper:  
**"Correcting Deviations from Normality: A Reformulated Diffusion Model for Unsupervised Anomaly Detection"**.

---

## 🎨 Approach


![DeCo-Diff](./assets/DeCo-for-UAD.png)

---

## 🚀 Getting Started

### 🛠️ Environment Setup

We utilize **Python 3.11** for all experiments. To install the necessary packages, simply run:

```bash
pip3 install -r requirements.txt
```

### 📁 Datasets

Download the datasets below and organize them as shown:
- [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)

The expected file structure (default for MVTec-AD) is as follows:
```
├── class1
│   ├── ground_truth
│   │   ├── defect1
│   │   └── defect2
│   ├── test
│   │   ├── defect1
│   │   ├── defect2
│   │   └── good
│   └── train
│       └── good
├── class2
...
```

---

## 🏋️ Training

Train our model using the following command. This command sets up the RLR training with various options tailored to your dataset and desired augmentations:

```bash
torchrun evaluation_DeCo_Diff.py \
            --dataset mvtec \
            --model UNet_L \
            --mask-random-ratio True \
            --object-category all  \
            --image-size 288 \
            --center-size 256 \
            --center-crop True \
            --augmentation True \
            --ckpt-every 20 
```

---

## 🧪 Testing

Once the model is trained, test its performance using the command below:

```bash
python evaluation_DeCo_Diff.py \
            --dataset mvtec \
            --model UNet_L \
            --object-category all  \
            --image-size 288 \
            --center-size 256 \
            --center-crop True \
            --augmentation True 
```

---

## 📸 Sample Results

Below are some sample outputs showcasing the performance of DeCo-Diff on real data:

![DeCo-Diff Samples](./assets/Samples.png)

---
