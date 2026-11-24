# SVHN Digit Classification: Lightweight CNN Benchmark

> Deep Learning coursework, The British University in Egypt, 2026.

How much accuracy do you give up when you choose a model small enough for edge devices? I trained **9 variants** of three lightweight CNN families on the **Street View House Numbers (SVHN)** dataset. Each was compared on validation accuracy, **held-out test accuracy** and training time.

## Setup
- **Data:** SVHN (`torchvision.datasets.SVHN`): 73,257 training images split 80/20 into train/validation, plus the official **26,032-image test split**, held out for final evaluation.
- **Preprocessing:** resize to 64×64, random rotation (±10°) and colour jitter on the training set.
- **Loss:** class-weighted cross-entropy, because the digit classes are imbalanced.
- **Training:** Adam, lr 1e-3, batch size 128, 3 epochs per model, CPU only. The best-validation checkpoint is the one evaluated on the test set.
- **Transfer learning:** ImageNet weights, compared with training from scratch.

## Results
All 9 experiments were trained in one top-to-bottom run of the notebook.

| Model variant | Val. accuracy | **Test accuracy** | Training time |
|---|---|---|---|
| **ShuffleNetV2 ×1.0** | **0.918** | **0.935** | 6.0 min |
| MobileNetV3-Large | 0.916 | 0.925 | 15.4 min |
| SqueezeNet 1.0 | 0.897 | 0.917 | 9.1 min |
| ShuffleNetV2 ×0.5 | 0.887 | 0.907 | 3.9 min |
| MobileNetV3-Small | 0.897 | 0.904 | 6.3 min |
| MobileNetV3-Small (no pretraining) | 0.877 | 0.885 | 6.0 min |
| SqueezeNet 1.1 + dropout | 0.858 | 0.876 | 5.8 min |
| ShuffleNetV2 ×0.5 (no pretraining) | 0.699 | 0.715 | 3.9 min |
| SqueezeNet 1.1 | 0.079 | 0.076 | 6.0 min |

Raw numbers: [`results/svhn_model_comparison.csv`](results/svhn_model_comparison.csv)

## Findings
- **Best model: ShuffleNetV2 ×1.0, 93.5% test accuracy.** It beat the much larger MobileNetV3-Large (92.5%) while training about **2.6× faster**.
- **Pretraining matters most for the smallest models.** ShuffleNetV2 ×0.5 fell from 90.7% to 71.5% test accuracy without ImageNet weights, while MobileNetV3-Small dropped only about 2 points.
- **A failed configuration, reproduced and explained.** Plain SqueezeNet 1.1 collapses to chance level (~7.6%, 10 classes) in both the original and the re-run, with its loss stuck at ln(10) ≈ 2.303. That points to dead activations in its final convolutional classifier at this learning rate. Adding dropout recovered 87.6%.
- Test accuracy is slightly *higher* than validation for every model, because augmentation is applied while validating on the training distribution, while the test split is un-augmented.

## Limitations and next steps
Three epochs on CPU is a short budget. Longer schedules with learning-rate decay, multiple seeds (to report mean ± std) and a lower learning rate for SqueezeNet 1.1 would be the next steps.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook notebooks/svhn_lightweight_cnns.ipynb   # downloads SVHN automatically if not present
```
The full run took about 66 minutes on an 8-thread laptop CPU.

## Tech stack
Python · PyTorch · torchvision · pandas · matplotlib · seaborn
