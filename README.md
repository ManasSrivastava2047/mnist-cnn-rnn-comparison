# MNIST CNN vs RNN Comparison

Handwritten digit classification on the MNIST dataset using a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN), built in PyTorch, with a performance comparison between the two.

## Problem Statement

Handwritten digit recognition is a core computer vision problem, used in postal automation, cheque processing, and document digitization. This project trains a CNN and an RNN to classify MNIST digits (0–9) and compares how well each architecture handles image data.

## Dataset
- **MNIST** — 60,000 training images, 10,000 test images
- Grayscale, 28×28 pixels, 10 classes
- Loaded via `torchvision.datasets.MNIST`

## Models

**CNN** — 3 conv blocks (Conv2D → ReLU → MaxPool), channels 1→32→64→128, followed by fully connected layers → 10 output classes. Learns spatial features directly from the 2D image.

**RNN** — Each image is treated as a sequence of 28 rows (timesteps), each row a 28-dim feature vector, fed through `nn.RNN` (hidden size 128) → FC layer → 10 output classes.

Both trained for 5 epochs, Adam optimizer (lr=0.001), CrossEntropyLoss.

## Results

| Model | Test Accuracy | Precision | Recall | F1-score |
|-------|----------------|-----------|--------|----------|
| CNN   | **99.20%**     | 0.99      | 0.99   | 0.99     |
| RNN   | **93.09%**     | 0.93      | 0.93   | 0.93     |

**Training accuracy by epoch:**

| Epoch | CNN     | RNN     |
|-------|---------|---------|
| 1     | 94.42%  | 68.81%  |
| 2     | 98.61%  | 84.98%  |
| 3     | 99.02%  | 91.20%  |
| 4     | 99.13%  | 93.35%  |
| 5     | 99.40%  | 94.44%  |

## Key Takeaways
- CNN significantly outperforms RNN (99.20% vs 93.09%), since convolution exploits the 2D spatial structure of images directly, while RNN's row-by-row framing is a less natural fit.
- RNN struggled most with digits 5, 6, and 8 (lowest precision); CNN stayed consistent (~0.99) across all classes.
- **Conclusion: CNN performs better than RNN for handwritten digit classification.**

## Setup

```bash
git clone https://github.com/<your-username>/mnist-cnn-rnn-comparison.git
cd mnist-cnn-rnn-comparison
pip install torch torchvision numpy matplotlib seaborn scikit-learn
jupyter lab
```

## Project Structure

```
mnist-cnn-rnn-comparison/
├── Handwritten-digit-class-CNN_RNN.ipynb   # Data loading, models, training, evaluation
├── README.md
```
