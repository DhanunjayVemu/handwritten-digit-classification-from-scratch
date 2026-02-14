
# Handwritten Digit Classification using Neural Network (From Scratch)

This project implements a **Fully Connected Neural Network from scratch using NumPy** to classify handwritten digits (0–9).

No deep learning frameworks (TensorFlow / PyTorch) were used — everything including **forward propagation, backpropagation** was implemented manually.

---

## Project Overview

* Dataset: MNIST-style handwritten digits
* Total samples: 42,000
* Image size: 28 × 28 pixels (784 features)
* Classes: 10 (Digits 0–9)
* Frameworks used:

  * NumPy
  * Pandas
  * Matplotlib

---

## Neural Network Architecture

Input Layer: 784 neurons
Hidden Layer: 10 neurons (ReLU activation)
Output Layer: 10 neurons (Softmax activation)

```
784 → 10 → 10
```

---

## Features Implemented

* Forward Propagation  
* Backward Propagation  
* ReLU Activation  
* Softmax Activation  
* Cross Entropy Loss  
* Gradient Descent  
* Accuracy Tracking  
* Cost vs Iterations Plot  
* Confusion Matrix  
* Sample Prediction Visualization  

---

## Training Details

* Learning Rate: 0.10 (Gradient Descent)
* Iterations: 500  
* Normalization: Pixel values scaled to [0,1]  
* One-hot encoding used for labels  

---

## Cost Function

Cross Entropy Loss for multi-class classification:

```
J = -(1/m) Σ y log(a)
```

---

## Results

* Cost decreases steadily over iterations.  
* Accuracy increases and stabilizes.  
* Strong diagonal dominance in confusion matrix.  
* Model generalizes well on development set.  

---

## Optimization Methods

Standard parameter update:
```
W = W - α dW
```
### Stochastic Gradient Descent (Mini-Batch SGD)

Instead of computing gradients using the entire dataset (Standardd Gradient Descent), SGD updates parameters using small mini-batches of data. This introduces controlled stochastic noise, which often improves convergence speed and generalization.

Parameter update rule:

For each mini-batch:  
    Compute gradients using batch data  
    W = W - α dW

---

## Project Structure

```
.
├── train.csv
├── MNIST_NN_scratch_SGD.ipynb 
├── README.md
```

---

## How to Run

1. Clone the repository:

```
git clone https://github.com/DhanunjayVemu/handwritten-digit-classification-from-scratch.git
```

2. Install dependencies:

```
pip install numpy pandas matplotlib
```

3. Open the Jupyter Notebook:

Then open:  
MNIST_NN_scratch_SGD.ipynb  
and 
Run cells sequentially

---

## What I Learned
 
* Mathematical understanding of backpropagation  
* How gradients flow through layers  
* Why softmax + cross-entropy works well  
* Effect of learning rate on convergence  
* Difference between standard Gradient Descent(GD) and Stoichastic GD  

---

---

## Author

**Vemu Venkat Sai Dhanunjay Sharma**  
B.Tech Student  
Passionate about Machine Learning & Systems Programming

---

