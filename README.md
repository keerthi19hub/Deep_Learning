# 🧠 Deep Learning Architectures — Neural Networks, Computer Vision & Representation Learning

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=311042,6B21A8,9333EA&height=220&section=header&text=Deep%20Learning%20Architectures&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Neural%20Networks%2C%20Computer%20Vision%20%26%20Representation%20Learning&descSize=18&descAlignY=60&descAlign=50" width="100%" alt="Deep Learning Architectures Banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%2F%20TensorFlow-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Domain-Computer%20Vision%20%26%20CNNs-6B21A8?style=for-the-badge" alt="Computer Vision" />
  <img src="https://img.shields.io/badge/Tool-Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Scope-Coursework%20%26%20Lab%20Portfolio-9333EA?style=for-the-badge" alt="Coursework" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 📌 Table of Contents

- [📖 Executive Summary](#-executive-summary)
- [🔬 Laboratory Curriculum & Implemented Projects](#-laboratory-curriculum--implemented-projects)
- [✨ Key Technical Concepts Explored](#-key-technical-concepts-explored)
- [🏗️ Deep Learning Experimental Pipeline](#️-deep-learning-experimental-pipeline)
- [🛠️ Tech Stack Matrix](#️-tech-stack-matrix)
- [⚙️ Reproducible Quickstart & Setup](#️-reproducible-quickstart--setup)
- [📁 Project Structure](#-project-structure)
- [👩‍💻 Author & Academic Context](#-author--academic-context)

---

## 📖 Executive Summary

The **Deep Learning Laboratory & Neural Network Systems** repository is an academic and empirical portfolio demonstrating the mathematical foundations and applied implementations of modern deep architectures. Spanning hand-crafted forward and backpropagation routines, loss landscape analysis, convolutional neural networks (CNNs), optimizer comparisons (Adam vs. SGD vs. RMSProp), and regularization mechanics, this laboratory bridges theoretical computational linear algebra with practical PyTorch and TensorFlow implementations.

---

## 🔬 Laboratory Curriculum & Implemented Projects

| Project / Notebook | Focus Area | Key Concepts Demonstrated |
| :--- | :--- | :--- |
| **Project 11** | **CNN Handwritten Digit Classification** | Multi-layer CNN on MNIST, Conv2D, MaxPool, Dropout, Softmax |
| **Project 12** | **Convolution & Pooling Mechanics** | Spatial kernel operations, stride arithmetic, feature map downsampling |
| **Project 6** | **Neural Network Forward Propagation** | Manual matrix multiplication, non-linear activation functions (ReLU, Sigmoid) |
| **Project 8** | **Gradient Descent Implementation** | Batch, Mini-batch, and Stochastic Gradient Descent (SGD) convergence |
| **Project 9** | **Optimizer Comparison** | Convergence rate benchmarks across Adam, RMSProp, AdaGrad, and Momentum |
| **Project 10** | **Overfitting & Regularization** | L1/L2 weight decay, dropout probability tuning, validation loss curves |
| **Assignment 7** | **Loss Function Implementations** | Mathematical implementation of Categorical Cross-Entropy, Binary Cross-Entropy, MSE |
| **Ethics Doc** | **Ethical Considerations in DL** | Algorithmic fairness, model explainability (XAI), privacy implications |

---

## ✨ Key Technical Concepts Explored

- **👁️ Spatial Feature Extraction:** Convolutional filter kernels detecting edges, textures, and compositional hierarchies in visual data.
- **📉 Optimization Dynamics:** Empirical comparison of adaptive learning rate algorithms (Adam) demonstrating accelerated escape from saddle points relative to vanilla SGD.
- **🛡️ Regularization Diagnostics:** Resolving high-variance regimes (overfitting) via dropout and early stopping monitored through validation loss telemetry.

---

## 🏗️ Deep Learning Experimental Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Deep Learning Experimental Framework                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Mathematical Layer: Vectorized Forward Pass & Activation Functions          │
│ $$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \quad A^{[l]} = g(Z^{[l]})$$       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Spatial Architecture: Convolutional Feature Extraction                      │
│ [Input Image] ──► [Conv2D (ReLU)] ──► [MaxPool2D] ──► [Flatten] ──► [Dense] │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Optimization & Loss Layer: Backpropagation & Convergence Analysis           │
│ - Gradient descent updates across Adam, SGD, and RMSProp                    │
│ - Loss telemetry and decision boundary visualizations                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack Matrix

| Domain | Technology | Implementation Details |
| :--- | :--- | :--- |
| **Core Libraries** | PyTorch, TensorFlow / Keras | GPU-accelerated tensor computations, autodiff, model layers |
| **Scientific Computing** | NumPy, SciPy | Vectorized matrix operations, manual gradient implementations |
| **Visual Analytics** | Matplotlib, Seaborn | Loss and accuracy curves, feature map heatmaps, confusion matrices |
| **Environment** | Jupyter Notebook | Interactive laboratory execution and visual documentation |

---

## ⚙️ Reproducible Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/keerthi19hub/Deep_Learning.git
cd Deep_Learning
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate  # Linux/macOS: source venv/bin/activate

pip install torch torchvision tensorflow numpy matplotlib seaborn jupyter
```

### 3. Run Notebooks
```bash
jupyter notebook
```

---

## 📁 Project Structure

```text
Deep_Learning/
├── Project11_Handwritten_Digit_Classification_CNN.ipynb
├── Project12_Understanding_Convolution_and_Pooling.ipynb
├── Project6_Neural_Network_Forward_Propagation.ipynb
├── Project8_Gradient_Descent_Implementation_.ipynb
├── Project9_Optimizer_Comparison_Using_a_Neural_Network.ipynb
├── Project_10_Overfitting_Underfitting.ipynb
├── Project_5_DL.ipynb
├── Assignment_7__Loss_Function_Implementation.ipynb
├── Assignment 4 _Ethical Considerations (DL).pdf
├── DL Assignment 02.pdf
├── DL_Assignment 3.docx
├── AI_in_Our_Daily_Life 01.pptx
└── README.md                                           # Production technical documentation
```

---

## 👩‍💻 Author & Academic Context

**Keerthi N**  
- **GitHub:** [@keerthi19hub](https://github.com/keerthi19hub)  
- **Portfolio:** [keerthi19hub.github.io](https://keerthi19hub.github.io)  
- **Course:** Deep Learning · Alliance University, Bengaluru  
- **Role:** Designed neural network architectures, hyperparameter benchmarks, and authored technical laboratory reports.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=311042,6B21A8,9333EA&height=100&section=footer" width="100%" alt="Footer Banner" />
</p>
