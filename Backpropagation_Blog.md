---
title: 'How Machines Learn from Mistakes: A Deep Dive into Backpropagation'

---

# How Machines Learn from Mistakes: A Deep Dive into Backpropagation

## Introduction

Imagine teaching a child to recognize animals. Initially, they make mistakes, but with corrective feedback, they gradually improve. This learning-from-errors process closely mirrors how neural networks learn. At the heart of this process lies **backpropagation** — the mathematical engine that adjusts weights in neural networks to minimize prediction errors.

In this article, we'll unravel the mathematical routine of backpropagation in simple terms. We will break down the steps, use pseudocode and visual representations, present real-world use cases, and explain the limitations, all while keeping new learners in mind.

---

## Neural Network Architecture Overview

Before diving into backpropagation, it's important to understand the structure of a basic feedforward neural network:

- **Input Layer**: Receives features from the dataset.
- **Hidden Layer(s)**: Transforms data using weights and activation functions.
- **Output Layer**: Produces the final prediction.

Each neuron computes a weighted sum of its inputs and applies an activation function to generate its output. The sigmoid function is often used:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Each neuron's operation can be described as:

$$
z = w \cdot x + b \quad \text{and} \quad a = \sigma(z)
$$

Where:
- $x$: input vector  
- $w$: weights  
- $b$: bias  
- $z$: linear combination  
- $a$: activated output  

---

### Visual Overview

<p align="center">
  <img src="https://hackmd.io/_uploads/rkmweug1gl.png" alt="Neural Network Diagram" width="500"/>
</p>


This diagram includes:
- A simple 3-layer neural network
- Forward pass data flow
- Backward gradient arrows

---

## Step-by-Step Breakdown of Backpropagation

### 1. Forward Pass

- Compute $z = w \cdot x + b$
- Compute $a = \sigma(z)$
- Output $\hat{y} = a$ at the final layer

---

### 2. Loss Function

Use the **Mean Squared Error (MSE)**:

$$
L = \frac{1}{2}(y_{\text{true}} - y_{\text{pred}})^2
$$

---

### 3. Gradient Descent

To minimize the loss, update the weights using:

$$
w := w - \eta \frac{\partial L}{\partial w}
$$

Where $\eta$ is the learning rate.

---

### 4. Backpropagation via Chain Rule

Apply the chain rule:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

Breakdown:
- $\frac{\partial L}{\partial a} = -(y - a)$  
- $\frac{\partial a}{\partial z} = a(1 - a)$  
- $\frac{\partial z}{\partial w} = x$

So:

$$
\frac{\partial L}{\partial w} = -(y - a) \cdot a(1 - a) \cdot x
$$

---

### 5. Weight Update Rule

$$
w := w + \eta (y - a) \cdot a(1 - a) \cdot x
$$

---
## Worked-Out Numerical Example

Suppose:

- Input $x = 0.5$
- Initial weight $w = 0.4$
- Bias $b = 0.1$
- True output $y_{\text{true}} = 1$
- Activation function: Sigmoid

---

### 1. Forward Pass

First, compute $z$:

$$
z = (w \times x) + b = (0.4 \times 0.5) + 0.1 = 0.3
$$

Now apply the sigmoid activation:

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-0.3}} \approx 0.5744
$$

Thus, the network predicts $\hat{y} = 0.5744$.

---

### 2. Loss Calculation

Using Mean Squared Error (MSE):

$$
L = \frac{1}{2}(y_{\text{true}} - y_{\text{pred}})^2 = \frac{1}{2}(1 - 0.5744)^2 \approx 0.0905
$$

---

### 3. Backward Pass (Gradient Calculation)

First, compute each partial derivative:

- Derivative of loss with respect to activation:

$$
\frac{\partial L}{\partial a} = -(y_{\text{true}} - a) = -(1 - 0.5744) = -0.4256
$$

- Derivative of activation with respect to $z$ (for sigmoid):

$$
\frac{\partial a}{\partial z} = a(1 - a) = 0.5744 \times (1 - 0.5744) = 0.2445
$$

- Derivative of $z$ with respect to weight $w$:

$$
\frac{\partial z}{\partial w} = x = 0.5
$$

Now applying the chain rule:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \times \frac{\partial a}{\partial z} \times \frac{\partial z}{\partial w}
$$

Substituting the values:

$$
\frac{\partial L}{\partial w} = (-0.4256) \times 0.2445 \times 0.5 \approx -0.0520
$$

---

### 4. Weight Update

Using a learning rate $\eta = 0.1$, update the weight:

$$
w := w - \eta \times \frac{\partial L}{\partial w}
$$

Thus:

$$
w = 0.4 + 0.1 \times 0.0520 = 0.4052
$$

**Final updated weight:**

$$
w \approx 0.4052
$$

---

## Full Pseudocode for Backpropagation

```text
1. Initialize weights and biases randomly
2. For each epoch:
   a. For each training sample:
      i. Perform forward pass to get predictions
      ii. Calculate loss using the loss function
      iii. Backward pass:
           - Compute gradient of loss w.r.t each weight
           - Use chain rule to propagate error backward
      iv. Update weights and biases using gradient descent
3. Repeat until convergence or max epochs

```


## Real-World Use Case: MNIST Digit Recognition

The **MNIST dataset** contains 28x28 grayscale images of handwritten digits (0–9). A neural network trained on this dataset can identify which digit appears in each image.

Using backpropagation:

- The model starts with random weights.
- For each image, it predicts a digit and compares it to the correct one.
- Based on the error, it adjusts the weights using backpropagation.
- Repeating this across thousands of examples leads to highly accurate digit classification.

---

## Types of Data Backpropagation Works With

Backpropagation works with:

- **Numerical data** (tabular, continuous variables)  
- **Images** (e.g., MNIST, CIFAR-10)  
- **Text** (after converting to embeddings or vectors)  
- **Time series** (using RNNs or LSTMs)

All data must be **numerically encoded** before being used — that means converting raw input into vectors of numbers that a network can process.

---

## What Changes Based on the Type of Data?

- **Activation Function**:  
  - Use $\text{sigmoid}$ for binary classification  
  - Use $\text{softmax}$ for multi-class classification

- **Loss Function**:  
  - Use $\text{MSE}$ (Mean Squared Error) for regression  
  - Use $\text{Cross-Entropy}$ for classification tasks

- **Network Architecture**:  
  - Use **CNNs** (Convolutional Neural Networks) for image data  
  - Use **RNNs** (Recurrent Neural Networks or LSTMs) for sequential/time series data  
  - Use **MLPs** (Multi-layer Perceptrons) for structured/tabular data

- **Preprocessing Techniques**:  
  - Tokenization (for text)  
  - Normalization or standardization (for numerical features)  
  - Embedding generation (e.g., word2vec, GloVe)  
  - Padding or sequence trimming (for time series)

---

## Strengths and Weaknesses

### Strengths:
- Efficient gradient calculation via the **chain rule**  
- Works with a wide variety of data types  
- Enables training of **deep** and **complex** neural networks  
- Universally applicable across different architectures (MLPs, CNNs, RNNs)

### Weaknesses:
- Can suffer from **vanishing or exploding gradients** in deep networks  
- May converge to **local minima** or saddle points  
- Requires **differentiable activation functions**  
- **Computationally expensive** for deep or large-scale models


## Limitations of Backpropagation

Even though it’s powerful, backpropagation comes with some limitations:

- **Local Minima**: It may converge to a suboptimal solution instead of the best one.
- **Vanishing Gradients**: Gradients become very small in deep networks, slowing learning.
- **Computational Cost**: Training deep networks requires large datasets and GPU acceleration.
- **Sensitivity to Hyperparameters**: Performance depends heavily on learning rate, weight initialization, and batch size.

---

## Glossary

- **Activation Function**: A non-linear transformation applied to neuron outputs (e.g., sigmoid, ReLU)
- **Backpropagation**: Algorithm for computing gradients to update weights
- **Chain Rule**: Rule in calculus used to differentiate composed functions
- **Epoch**: One complete pass through the training dataset
- **Gradient Descent**: Optimization technique to minimize loss
- **Loss Function**: A function that measures how wrong the prediction is

---

## Conclusion

Backpropagation is the foundation of learning in neural networks. By computing how much each weight contributes to the prediction error, and adjusting accordingly, neural networks gradually improve their performance.

Mastering backpropagation means you understand how deep learning models learn. Whether you're working on image recognition or text classification, this algorithm is your behind-the-scenes hero.



## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016).  
  *Deep Learning.* MIT Press.  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

- Nielsen, M.  
  *Neural Networks and Deep Learning: A free online book.*  
  [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)  
  Chapter 2: How the backpropagation algorithm works

- Scikit-learn Documentation – Neural Network Models  
  [https://scikit-learn.org/stable/modules/neural_networks_supervised.html](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

- HMKCode Blog: Backpropagation Step-by-Step  
  [https://hmkcode.com/ai/backpropagation-step-by-step/](https://hmkcode.com/ai/backpropagation-step-by-step/)

- Ng, A. (DeepLearning.AI)  
  *Deep Learning Specialization – Course 1: Neural Networks and Deep Learning.*  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning)

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).  
  *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.  
  [https://ieeexplore.ieee.org/document/726791](https://ieeexplore.ieee.org/document/726791)



