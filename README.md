# 🛡️ Health Insurance Cost Prediction

An end-to-end machine learning project that predicts annual health insurance costs using a PyTorch-based deep learning regression model with an interactive Gradio web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Overview

This project implements a deep learning solution for predicting health insurance costs based on demographic and health-related features. The model is trained to estimate annual insurance premiums, providing valuable insights for both insurance providers and potential customers.

## ✨ Features

- **Deep Learning Model**: Custom PyTorch neural network optimized for regression
- **Feature Engineering**: Comprehensive preprocessing including categorical encoding and feature scaling
- **Interactive UI**: User-friendly Gradio interface for real-time predictions
- **Production Ready**: Clean, modular code suitable for deployment
- **Real-time Inference**: Instant cost predictions with formatted output

---

## 🧠 Model Architecture

The model uses a fully connected neural network with the following architecture:

| Layer | Neurons | Activation | Purpose |
|-------|---------|------------|---------|
| Input | 6 | - | Encoded features |
| Hidden 1 | 48 | ReLU | Feature extraction |
| Hidden 2 | 128 | ReLU | Deep feature learning |
| Output | 1 | Linear | Cost prediction |

**Key Specifications:**
- **Framework**: PyTorch
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (used during training)
- **Input Features**: Age, Gender, BMI, Children, Smoking Status, Region
- **Output**: Annual insurance cost in USD

---

## 💻 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.8+ | Core programming language |
| **Deep Learning** | PyTorch | Model development and inference |
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Preprocessing** | Scikit-learn | Label encoding and feature scaling |
| **Web Interface** | Gradio | Interactive UI and deployment |
| **Version Control** | Git | Source code management |

---

## 🌐 Live Demo

**Try it now:** [Health Insurance Cost Predictor](https://huggingface.co/spaces/ksukheja/health-insurance-cost-predictor) 🚀

Experience the model instantly without any setup! The application is deployed on Hugging Face Spaces.

### How to Use:
1. Adjust the input features (Age, BMI, Gender, etc.)
2. Click "Predict" to get your insurance cost estimate
3. Results are displayed in USD with formatted currency

> 💡 **Note**: Predictions are estimates based on the trained model and should not be considered official insurance quotes.

---

## 💻 Run Locally (Optional)

Want to run the project on your machine or contribute to development?

<details>
<summary><b>Click to expand local setup instructions</b></summary>

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/komal-sukheja/health-insurance-cost-prediction.git
cd health-insurance-cost-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will launch at `http://127.0.0.1:7860`

---


## 📁 Project Structure
```
health-insurance-cost-predictor/
│
├── app.py                          # Gradio app with UI components and interaction logic
├── predict.py                      # Preprocessing pipeline and model inference code
├── model.py                        # PyTorch neural network model definition
├── requirements.txt                # Project dependencies
│
├── weights/
│   └── insurance_model.pth         # Trained model weights file
│
├── encoders/
│   ├── labelencoders.pkl           # Pickled label encoders for categorical features
│   └── scaler.pkl                  # Pickled scaler for numerical feature normalization
│
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Contact

**Komal Sukheja**

[![GitHub](https://img.shields.io/badge/GitHub-komal--sukheja-181717?style=flat&logo=github)](https://github.com/komal-sukheja)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/komal-sukheja/)

Email: komalsukheja2001@gmail.com
---

<div align="center">
  
**⭐ Star this repo if you find it helpful! ⭐**

</div>

<div align="center">
  <strong>⭐ If you find this project helpful, please consider giving it a star! ⭐</strong>
</div>
