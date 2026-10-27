# Deep Learning Laboratory & Neural Network Implementations

> **Practical implementations of Convolutional Neural Networks (CNNs), forward propagation from scratch, optimization algorithms (Adam, RMSprop, SGD), gradient descent, and regularization techniques.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-orange)](https://tensorflow.org)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)

---

## 1. Laboratory Overview

The **Deep Learning Laboratory** provides a systematic, code-first exploration of fundamental and intermediate deep learning concepts. Rather than treating neural networks as black boxes, this repository demonstrates:
1. Low-level mathematical implementations (matrix forward propagation, manual gradient descent).
2. Convolutional Neural Networks for computer vision (digit classification, feature map pooling).
3. Comparative optimization analysis (benchmarking SGD, Momentum, RMSprop, and Adam).
4. Generalization strategies (mitigating overfitting via Dropout and weight decay).
5. Loss function formulations and AI ethics analysis.

---

## 2. Projects & Implementations

| Project File | Focus Area | Key Architectural & Algorithmic Concepts |
| :--- | :--- | :--- |
| `Project11_Handwritten_Digit_Classification_CNN.ipynb` | Computer Vision (CNN) | Convolutional layers, Max-Pooling, Flatten, Softmax classification on MNIST |
| `Project12_Understanding_Convolution_and_Pooling.ipynb` | Feature Extraction | Kernel filters, feature map transformations, receptive fields, stride/padding |
| `Project6_Neural_Network_Forward_Propagation.ipynb` | Neural Foundations | Matrix weight multiplications, bias vectors, ReLU/Sigmoid activation functions |
| `Project_8_Gradient_Descent_Implementation_.ipynb` | Optimization Theory | Batch vs Stochastic Gradient Descent (SGD), learning rate sensitivity, loss curves |
| `Project9_Optimizer_Comparison_Using_a_Neural_Network.ipynb` | Optimizer Benchmark | Empirical comparison of SGD, Momentum, RMSprop, and Adam convergence speeds |
| `Project_10_Overfitting_Underfitting.ipynb` | Model Regularization | Bias-variance tradeoff, Dropout layers, L2 kernel regularization, Early Stopping |
| `Project_5_DL.ipynb` | Multilayer Perceptrons | Multi-layer feedforward architecture design, backpropagation, and training loops |
| `Assignment_7__Loss_Function_Implementation.ipynb` | Loss Formulations | Mathematical implementation of MSE, Binary Cross-Entropy, Categorical Cross-Entropy |
| `Assignment 4 _Ethical Considerations (DL).pdf` | AI Ethics & Fairness | Research report on algorithmic bias, data privacy, transparency, and accountability |

---

## 3. Technology Stack

- **Languages & Frameworks:** Python, TensorFlow / Keras, PyTorch
- **Data & Computation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## 4. Repository Structure

```
Deep_Learning/
├── Project11_Handwritten_Digit_Classification_CNN.ipynb
├── Project12_Understanding_Convolution_and_Pooling.ipynb
├── Project6_Neural_Network_Forward_Propagation.ipynb
├── Project_8_Gradient_Descent_Implementation_.ipynb
├── Project9_Optimizer_Comparison_Using_a_Neural_Network.ipynb
├── Project_10_Overfitting_Underfitting.ipynb
├── Project_5_DL.ipynb
├── Assignment_7__Loss_Function_Implementation.ipynb
├── Assignment 4 _Ethical Considerations (DL).pdf
├── AI_in_Our_Daily_Life 01.pptx
└── README.md
```

---

## 5. Setup & Execution

```bash
# 1. Clone repository
git clone https://github.com/keerthi19hub/Deep_Learning.git
cd Deep_Learning

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/macOS

# 3. Dependencies
pip install tensorflow torch numpy pandas matplotlib seaborn jupyter

# 4. Launch Jupyter
jupyter notebook
```

---

## 6. Key Learnings & Empirical Observations

- **CNN Feature Hierarchies:** Early convolutional layers extract low-level spatial primitives (edges, gradients), while deeper layers assemble class-specific semantic patterns.
- **Optimizer Performance:** Adaptive moment estimation (Adam) consistently achieves faster convergence and stable loss reduction across sparse and dense gradients compared to standard SGD.
- **Regularization Impact:** Incorporating 20-30% Dropout combined with early stopping effectively suppresses training/validation divergence in multi-layer dense networks.

---

## 7. Author & Contribution

- **Keerthi N** (*Alliance University, BCA*) — Authored all deep learning notebooks, neural network implementations, optimizer benchmarking, and technical documentation.

---

## 8. License

Developed for academic coursework and portfolio demonstration at Alliance University. Released under the [MIT License](https://opensource.org/licenses/MIT).
