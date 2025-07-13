# 📝 Handwriting Classification with TensorFlow & CNN (MNIST Dataset)

This project showcases a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify handwritten digits from the MNIST dataset. It's an ideal deep learning exercise for exploring essential concepts like image preprocessing, model design, training, and performance visualization.

---

## 🚀 Project Overview

- **Dataset:** MNIST (70,000 grayscale images of handwritten digits 0–9)
- **Input Shape:** 28×28 pixels, grayscale (1 channel)
- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow & Keras
- **Task:** Predict the digit in each image (0–9)

---

## 📦 Repository Contents

- `Handwriting Classification Model (MNIST).ipynb` — Complete Jupyter notebook (code, outputs, and explanations)
- `README.md` — Project description and usage guide
- `requirements.txt` — Python package dependencies

---

## 🧠 Core Concepts Demonstrated

- Image preprocessing for deep learning
- Building a CNN with Keras
- Using ReLU and softmax activation functions
- Multi-class classification with `categorical_crossentropy` loss
- Evaluation via accuracy, loss curves, and confusion matrix
- Visualizing training progress and prediction errors

---

## 🛠️ Technologies Used

| Tool/Library   | Purpose                      |
|----------------|------------------------------|
| Jupyter Notebook | Coding Enviroment          |
| Python         | Programming Language         |
| TensorFlow     | Deep Learning Framework      |
| Keras          | Model Building & Training    |
| Matplotlib     | Data Visualization           |
| Seaborn        | Statistical Plotting         |
| NumPy          | Numerical Computation        |
| Pandas         | Data Manipulation            |

---

## 🏗️ Model Architecture & Summary

The CNN architecture used in this project:

- **Input:** 28x28x1 grayscale image
- **Conv2D Layer 1:** 32 filters, (5x5), ReLU, 'same' padding
- **Conv2D Layer 2:** 32 filters, (5x5), ReLU, 'same' padding
- **MaxPooling2D:** Downsampling
- **Dropout:** 25% rate
- **Conv2D Layer 3:** 64 filters, (3x3), ReLU, 'same' padding
- **Conv2D Layer 4:** 64 filters, (3x3), ReLU, 'same' padding
- **MaxPooling2D:** Downsampling
- **Dropout:** 25% rate
- **Flatten:** Converts 2D feature maps to 1D vector
- **Dense Layer:** 128 units, ReLU
- **Dropout:** 50% rate
- **Output Layer:** 10 units, softmax (digit classes 0–9)

**Model Summary:**
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Parameters   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        832       
 conv2d_1 (Conv2D)           (None, 28, 28, 32)        25632     
 max_pooling2d (MaxPooling2D)(None, 14, 14, 32)        0         
 dropout (Dropout)           (None, 14, 14, 32)        0         
 conv2d_2 (Conv2D)           (None, 14, 14, 64)        18496     
 conv2d_3 (Conv2D)           (None, 14, 14, 64)        36928     
 max_pooling2d_1 (MaxPooling2)(None, 7, 7, 64)         0         
 dropout_1 (Dropout)         (None, 7, 7, 64)          0         
 flatten (Flatten)           (None, 3136)              0         
 dense (Dense)               (None, 128)               401536    
 dropout_2 (Dropout)         (None, 128)               0         
 dense_1 (Dense)             (None, 10)                1290      
=================================================================
Total params: 484,714
Trainable params: 484,714
Non-trainable params: 0
_________________________________________________________________
```

---

## 📊 Evaluation Metrics

- Training & validation accuracy
- Training & validation loss curves
- Confusion matrix
- Sample prediction outputs

---

## ⚡ How to Use This Project

Open the notebook `Handwriting Classification Model (MNIST).ipynb` in Jupyter Notebook, and run each cell sequentially. All instructions, code, and outputs are self-contained within the notebook.

---

## 💡 Tips

- Use a Python environment with TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and Pandas installed.
- The notebook includes visualizations for training curves and confusion matrix for easy performance analysis.

## 📖 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
