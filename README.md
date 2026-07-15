# MNIST CNN vs RNN Comparison

Handwritten digit classification (MNIST) using a CNN and an RNN in PyTorch, comparing their performance.

## Dataset
MNIST — 60,000 train / 10,000 test grayscale images, 28×28, digits 0–9.

## Models
- **CNN**: 3 conv blocks (Conv2D → ReLU → MaxPool), channels 1→32→64→128, then FC layers → 10 classes.
- **RNN**: Each image treated as 28 timesteps × 28 features; `nn.RNN` (hidden size 128) → FC → 10 classes.

Both trained 5 epochs, Adam (lr=0.001), CrossEntropyLoss.

## Results

| Model | Test Accuracy |
|-------|----------------|
| CNN   | **99.20%**     |
| RNN   | **93.09%**     |

CNN converged faster and outperformed RNN across all digit classes — spatial convolution is a much better fit for image data than treating rows as a sequence. RNN struggled most on digits 5, 6, and 8.

## Setup
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
jupyter lab
```

## Structure
```
mnist-cnn-rnn-comparison/
├── Handwritten-digit-class-CNN_RNN.ipynb
├── README.md
```

## Conclusion
CNN outperforms RNN for handwritten digit classification.

## License
MIT
