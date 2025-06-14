
# Training CNN/DNN Models for Loan Approval Prediction with Multi Modal Input and Explainability

This project implements and compares two deep learning models—a DNN and a CNN—for loan approval prediction using both structured financial data and unstructured textual descriptions. It also applies SHAP to interpret the models’ decisions globally and locally.

---

##  Project Structure

```
LoanApprovalProject/
├── Colab Notebooks Code/                   # Colab notebooks for training, SHAP, and analysis
├── Saved Models/                    # Saved model weights, preprocessors, SHAP outputs
├── Data/                    # Raw, cleaned, and sampled dataset file
├── Figures/                 # All model output figures used in report/poster
├── Report/                  # LaTeX source + final compiled report
├── Presentation Poster/     # Final poster PDF
└── README.md                # Project overview and execution instructions
```

---

##  Colab Notebooks Code (Colab Notebooks Code Folder)

| Notebook                     | Description |
|-----------------------------|-------------|
| `1_Preprocessing.ipynb`     | Loads and cleans LendingClub dataset, creates balanced 20k sample |
| `2_DNN_Model.ipynb`         | Trains DNN model using tabular + text inputs |
| `3_CNN_Model.ipynb`         | Trains CNN model with Conv1D layers for tabular features |
| `4_SHAP_Explainability.ipynb` | Generates global + local SHAP explanations |
| `5_Comparative_Analysis.ipynb` | Compares DNN and CNN on metrics, training time, SHAP |

---

##  Saved Models (Saved Models Folder)

- `*.h5`: Trained model weights  
- `*.joblib`: Preprocessing pipelines  
- `*.npy` & `.pkl`: SHAP explanations and feature mappings

---

##  Visual Figures (Figures Folder)

Includes:
- Training accuracy/loss curves (DNN & CNN)
- Confusion matrices
- ROC curves
- Global SHAP bar plots (Top 20)
- SHAP force plots (local explanations)
- ROC AUC comparison

---

##  Report & Poster

- `Report/main.pdf`: Full LaTeX scientific report with experiments and results
- `Presentation Poster/Poster.pdf`: Final summary poster 

---

##  Dataset

This project uses the **Lending Club Loan Dataset** from Kaggle:  
 https://www.kaggle.com/datasets/wordsforthewise/lending-club

You can find the following version in the `/Data/` folder:
- `loan_data_sampled.csv` (final 20k sample used for training)

---

##  Requirements

Install with:
```bash
pip install tensorflow transformers shap joblib scikit-learn pandas matplotlib numpy
```

Tested with:
- Python 3.9+
- TensorFlow 2.12
- Transformers 4.36
- SHAP 0.44

---

## ▶ How to Run

1. Start with `1_Preprocessing.ipynb` to generate the sampled dataset  
2. Train models using `2_DNN_Model.ipynb` and `3_CNN_Model.ipynb`  
3. Run `4_SHAP_Explainability.ipynb` to generate SHAP plots  
4. Compare results using `5_Comparative_Analysis.ipynb`

---

##  Author

**Project Title**: Training CNN/DNN Models for Loan Approval Prediction with Multi Modal Input and Explainability
**Name**: Diya Hatem Mosbah Sounoqrot – 2286340  
**Course**: AIN3002 – Deep Learning – Spring 2025  
**University**: Bahcesehir University  
**Submission**: Final Project

---
