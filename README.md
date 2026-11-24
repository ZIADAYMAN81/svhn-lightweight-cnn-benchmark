# SVHN Digit Classification: Lightweight CNN Benchmark

> Deep Learning coursework, The British University in Egypt, 2026. Team of 4. **This repository contains my part: the lightweight-architecture track.** Teammates covered VGG/ResNet/DenseNet, custom CNN/EfficientNet/hybrid models and AlexNet/Inception/FPN.

How much accuracy do you give up when you choose a model small enough for edge devices? I trained **9 variants** of three lightweight CNN families on the **Street View House Numbers (SVHN)** dataset and compared accuracy against training time.

## Setup
- **Data:** SVHN via `torchvision.datasets.SVHN(download=True)`, with an 80/20 train/validation split and the official test split kept separate.
- **Preprocessing:** resize to 64×64, random rotation and colour jitter.
- **Loss:** class-weighted cross-entropy, because the digit classes are imbalanced.
- **Training:** Adam, lr 1e-3, batch size 128, 3 epochs per model.
- **Transfer learning:** ImageNet weights, compared with training from scratch.

## Results

| Model variant | Best val. accuracy | Total time (s) | Time / epoch (s) |
|---|---|---|---|
| **MobileNetV3-Large** | **0.9327** | 9006 | 3002 |
| MobileNetV3-Small | 0.9274 | 2920 | 973 |
| ShuffleNetV2 ×1.0 | 0.9191 | **373** | **124** |
| SqueezeNet 1.0 | 0.9133 | 566 | 189 |
| MobileNetV3-Small (no pretraining) | 0.9067 | 2865 | 955 |
| ShuffleNetV2 ×0.5 | 0.8932 | 251 | 84 |
| SqueezeNet 1.1 + dropout | 0.8924 | 369 | 123 |
| ShuffleNetV2 ×0.5 (no pretraining) | 0.7280 | 248 | 83 |
| SqueezeNet 1.1 | 0.0664 | 354 | 118 |

Raw numbers: [`results/svhn_model_comparison.csv`](results/svhn_model_comparison.csv)

## Findings
- **Best accuracy:** MobileNetV3-Large (93.3%).
- **Best trade-off:** ShuffleNetV2 ×1.0 reached 91.9%, only 1.4 pp lower, with about **24× less training time**.
- **Pretraining matters most for the smallest models.** ShuffleNetV2 ×0.5 fell from 89.3% to 72.8% without it, while MobileNetV3-Small dropped only 2 pp.
- **A failed run, explained.** Plain SqueezeNet 1.1 collapsed to 6.6%, which is chance level for 10 classes. That points to training instability (dead ReLUs or a bad learning rate for this architecture). Adding dropout recovered 89.2%.

## Limitations
These are validation numbers after only 3 epochs, a limit set by compute. Longer schedules, learning-rate tuning and final test-set reporting would be the next steps.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook notebooks/svhn_lightweight_cnns.ipynb   # downloads SVHN automatically
```

## Tech stack
Python · PyTorch · torchvision · pandas · matplotlib · seaborn
