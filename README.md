# Deep Learning Image Denoising

A structured exploration and benchmark implementation of Deep Learning architectures for **Image Denoising**, ranging from classical feedforward CNN baselines to state-of-the-art Vision Transformers and activation-free architectures[cite: 10]. 

This repository explores model evolution across both **Synthetic Gaussian Noise (AWGN)** and **Real-World Camera Sensor Noise (sRGB & RAW domain)**[cite: 10].

## Usage Instructions

This repository features automatic hardware discovery supporting **Intel XPU**, **NVIDIA CUDA**, and **CPU** backends seamlessly[cite: 10].

### Train DnCNN Model on Intel XPU
```bash
python train.py --config configs/config.yaml
python train.py --config configs/config.yaml --device xpu
python train.py --config configs/config.yaml --epochs 2
```

### Run Inference
```bash
python infer.py --model baseline_cnn --weights checkpoints/final_baseline_cnn.pth --input_image ../../SD68-dataset/noisy5/val/noisy/0001.png --output_image test_result_0001.png
python infer.py --model dncnn --weights checkpoints/best_dncnn.pth --input_image noisy_sample.png --output_image clean_sample.png --device xpu
python infer.py --model dncnn --weights checkpoints/best_dncnn.pth --input_image noisy_sample.png --output_image clean_sample.png --device cpu
```

---

## Experiment Log

### 1. Baseline CNN (Simple Autoencoder): L1 vs. L2 Loss
*   **Model:** `base_cnn` (Simple Autoencoder)
*   **Hyperparameters:** 20 Epochs, Batch Size 16
*   **Observation:** When evaluating the direct-mapping autoencoder, using an L1 Loss penalty resulted in noticeable color desaturation in the denoised image outputs. Switching the loss function to L2 (MSE) loss mitigated the severity of this regression-to-the-mean issue, yielding better color preservation in the final RGB predictions.

