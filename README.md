# Deep Learning Image Denoising

A structured exploration and benchmark implementation of Deep Learning architectures for **Image Denoising**, ranging from classical feedforward CNN baselines to state-of-the-art Vision Transformers and activation-free architectures. 

This repository explores model evolution across both **Synthetic Gaussian Noise (AWGN)** and **Real-World Camera Sensor Noise (sRGB & RAW domain)**.

---

## Benchmarks & Model Evaluation

Below is a detailed benchmark comparison across standard evaluation datasets (metrics reported in **PSNR (dB) / SSIM**).

### 1. Synthetic Gaussian Noise Benchmarks ($\sigma = 25$ & $\sigma = 50$)
*Evaluates spatial reasoning, edge preservation, and structural recovery under Additive White Gaussian Noise (AWGN).*

| Network Architecture | Paradigm | CBSD68 ($\sigma=25$) | CBSD68 ($\sigma=50$) | Kodak24 ($\sigma=25$) | Kodak24 ($\sigma=50$) | Urban100 ($\sigma=25$) | Urban100 ($\sigma=50$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DnCNN** | Residual CNN | 31.24 / 0.883 | 27.95 / 0.791 | 32.14 / 0.888 | 28.95 / 0.803 | 30.81 / 0.879 | 27.59 / 0.782 |
| **CBDNet** | Dual-Stage CNN | 31.15 / 0.880 | 27.80 / 0.785 | 32.08 / 0.882 | 28.80 / 0.795 | 30.40 / 0.865 | 27.20 / 0.771 |
| **SwinIR** | Vision Transformer | 31.78 / 0.898 | 28.56 / 0.822 | 32.89 / 0.903 | 29.79 / 0.832 | 32.90 / 0.923 | 29.82 / 0.868 |
| **Restormer** | Transposed ViT | **31.79 / 0.899** | **28.60 / 0.825** | **33.04 / 0.906** | **30.01 / 0.838** | **32.96 / 0.925** | **30.02 / 0.874** |
| **NAFNet** | Activation-Free | 31.70 / 0.895 | 28.52 / 0.820 | 32.80 / 0.901 | 29.70 / 0.830 | 32.40 / 0.915 | 29.65 / 0.860 |

---

### 2. Real-World Sensor Noise Benchmarks (sRGB / RAW)
*Evaluates real camera sensor noise handling (Poisson-Gaussian distributions, read/shot noise, ISO variance).*

| Network Architecture | Paradigm | SIDD (sRGB) PSNR / SSIM | DND (sRGB) PSNR / SSIM | Key Innovation / Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| **DnCNN** | Residual CNN | 23.66 / 0.583 | 32.43 / 0.790 | Baseline reference (Fails on real sensor noise). |
| **CBDNet** | Dual-Stage CNN | 30.78 / 0.801 | 38.06 / 0.942 | Models non-uniform real noise variances. |
| **SwinIR** | Vision Transformer | 40.05 / 0.961 | 39.91 / 0.961 | Shifted-window self-attention for fine textures. |
| **Restormer** | Transposed ViT | 40.02 / 0.960 | 40.03 / 0.956 | Cross-covariance attention ($O(N)$ spatial complexity). |
| **NAFNet** | Activation-Free | **40.30 / 0.962** | **40.27 / 0.957** | **SOTA Efficiency**: Non-linear activation-free block. |

---

## Usage Instructions

This project features a modular factory-registry framework. You can train different models or introduce a new architecture without altering the core training infrastructure.

### Train Baseline Model
```bash
python train.py --model baseline_cnn --train_dir ./data/train --val_dir ./data/val --epochs 10
```
### Train DnCNN Model
```bash
python train.py --model dncnn --train_dir ./data/train --val_dir ./data/val --epochs 20
```
### Run Inference
```bash
python infer.py --model dncnn --weights best_dncnn.pth --input_image noisy_sample.png --output_image clean_sample.png
```