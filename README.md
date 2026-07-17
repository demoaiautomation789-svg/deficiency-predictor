# 🩺 Vitamin & Mineral Deficiency Disease Predictor

An end-to-end machine learning project that predicts nutritional deficiency
diseases (**Anemia, Scurvy, Rickets/Osteomalacia, Night Blindness**, or a
"No Deficiency Pattern") from demographic, lifestyle, dietary, symptom, and
lab data, with an interactive Streamlit web app.

## Live demo

🔗 [https://deficiency-predictor-built-by-shahzain.streamlit.app](https://deficiency-predictor-built-by-shahzain.streamlit.app)

No installation needed — open the link in any browser, fill in the form, and get an instant prediction.

## Overview

Nutritional deficiencies (like Anemia, Scurvy, or Rickets) often go
undiagnosed until symptoms become severe. This project uses a
supervised machine learning model trained on 4,000 patient records to
predict the most likely deficiency disease based on easily available
information — no invasive tests required, though lab values can be
included for a more precise result.

This is a **multiclass classification problem** with 5 possible
outcomes:

| Class | Description |
|---|---|
| Healthy | No clear deficiency pattern detected |
| Anemia | Low iron / Vitamin B12, low hemoglobin |
| Scurvy | Severe Vitamin C deficiency |
| Rickets / Osteomalacia | Vitamin D / calcium deficiency, weak bones |
| Night Blindness | Vitamin A deficiency |

---

## 📊 Results

Model: **Random Forest Classifier** (300 trees, `class_weight="balanced"` to handle class imbalance)

| Metric | Score |
|---|---|
| **Accuracy** | **99.6%** |
| **Macro F1-score** | **99.4%** |

**Per-class performance (on held-out 20% test set, 800 samples):**

| Disease | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Anemia | 1.00 | 0.99 | 0.99 | 249 |
| Healthy | 1.00 | 1.00 | 1.00 | 302 |
| Night Blindness | 0.96 | 1.00 | 0.98 | 24 |
| Rickets / Osteomalacia | 1.00 | 1.00 | 1.00 | 206 |
| Scurvy | 1.00 | 1.00 | 1.00 | 19 |

Even the rare classes (Night Blindness: 24 samples, Scurvy: 19 samples
in the test set) are classified with high precision and recall, thanks
to balanced class weighting during training.

Full training logs, the confusion matrix, and feature importance
analysis are available in
[`notebooks/Nutrition.ipynb`](notebooks/Nutrition.ipynb).

---

## 📁 Project Structure

```
.
├── data/
│   └── vitamin_deficiency_disease_dataset_20260123.csv   # 4,000 patient records
├── notebooks/
│   └── Nutrition.ipynb            # full training notebook (Google Colab)
├── app.py                         # Streamlit web application
├── best_model.pkl                 # trained Random Forest classifier
├── encoder.pkl                    # LabelEncoder (class index -> disease name)
├── feature_columns.pkl            # exact feature column order used in training
├── requirements.txt                # pinned dependencies
├── .gitignore
└── README.md
```

## 🔬 Methodology

1. **Cleaning:** dropped the redundant `symptoms_list` text column
   (duplicate of the individual `has_*` binary flags); filled missing
   `alcohol_consumption` values with a neutral `Not_Specified` category
   rather than guessing with mean/mode.
2. **Encoding:** one-hot encoding (`pd.get_dummies`) for nominal
   categorical input features; label encoding for the target column.
3. **Class imbalance:** addressed with `class_weight="balanced"` in the
   Random Forest — not with feature scaling, since scaling and class
   imbalance are unrelated problems, and tree-based models don't
   require feature scaling in the first place.
4. **Train/test split:** 80/20 stratified split to preserve class
   proportions in both sets.
5. **Evaluation:** accuracy, macro F1-score, full classification
   report, and confusion matrix — chosen over plain accuracy because
   the dataset is imbalanced.
6. **Model persistence:** the trained model, label encoder, and feature
   column order are each saved separately with `joblib` so the
   Streamlit app can load and use them without retraining.

---

## 🖥️ App Features

- **Collapsible sections** (Basic Information, Lifestyle & Diet,
  Symptoms, Lab Values) to keep the form compact and easy to navigate.
- **Automatic BMI calculation** from height and weight — no need to
  know or calculate your own BMI.
- **Low / Normal / High dropdowns** for lab values instead of raw
  numbers, each with a hover tooltip explaining what it means in plain
  language.
- **Responsive layout** — fields sit side-by-side on desktop/laptop and
  automatically stack vertically on mobile.
- **Color-coded probability breakdown** (interactive Plotly chart)
  showing the model's confidence across all 5 possible outcomes, with
  hover tooltips explaining what each condition actually is.

---

## 🚀 Getting Started

### Run it locally

```bash
git clone https://github.com/demoaiautomation789-svg/deficiency-predictor.git
cd deficiency-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`. The
pre-trained model files (`best_model.pkl`, `encoder.pkl`,
`feature_columns.pkl`) are already included, so no retraining is
required to run the app.

### Retrain the model yourself

Open `notebooks/Nutrition.ipynb` in Google Colab, run all cells, and
download the three `.pkl` files it generates — then replace the
existing ones in this repo.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Model training | Python, pandas, scikit-learn (Random Forest) |
| Model persistence | joblib |
| Web app / UI | Streamlit |
| Visualization | Plotly |
| Training environment | Google Colab |
| Deployment | Streamlit Community Cloud |

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. It is
trained on a sample dataset and is **not a certified medical diagnostic
tool**. Predictions should never replace professional medical advice —
always consult a qualified healthcare provider for real health
concerns.

---

## 📄 License

MIT
