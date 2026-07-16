# 🩺 Vitamin & Mineral Deficiency Disease Predictor

An end-to-end machine learning project that predicts nutritional deficiency
diseases (**Anemia, Scurvy, Rickets/Osteomalacia, Night Blindness**, or a
"No Deficiency Pattern") from demographic, lifestyle, dietary, symptom, and
lab data, with an interactive Streamlit web app.

## Live demo

🔗 **[Try it live here]([https://deficiency-predictor-built-by-shahzain.streamlit.app])**

## Project structure

```
.
├── data/
│   └── vitamin_deficiency_disease_dataset.csv   # 4,000 patient records
├── notebooks/
│   └── Nutrition_annotated.ipynb                # training notebook (Google Colab)
├── app.py                                       # Streamlit web app
├── best_model.pkl                               # trained Random Forest classifier
├── encoder.pkl                                  # LabelEncoder (0-4 -> disease name)
├── feature_columns.pkl                          # exact column order used in training
├── requirements.txt
└── README.md
```

## Problem type

Multiclass classification over 5 classes (Healthy, Anemia, Rickets/
Osteomalacia, Night Blindness, Scurvy), trained on 4,000 patient records
combining demographics, lifestyle, diet, lab values, and symptoms.

## Approach

- **Preprocessing:** dropped the redundant free-text `symptoms_list`
  column (the same information already exists as individual binary
  flags), filled missing `alcohol_consumption` values with a
  `Not_Specified` category, and one-hot encoded all categorical input
  columns. The target column was label-encoded.
- **Class imbalance:** the dataset is imbalanced (Scurvy and Night
  Blindness are rare classes). This was handled with
  `class_weight="balanced"` in the model, and evaluation focused on
  **macro F1** and the full classification report rather than accuracy
  alone.
- **Model:** Random Forest Classifier (300 trees, balanced class
  weights). No feature scaling was applied since tree-based models
  split on thresholds, not distances, so scaling has no effect on them.
- **Result:** ~98-99% accuracy and macro F1 on a held-out 20% test
  split. See `notebooks/Nutrition_annotated.ipynb` for the full training
  process, evaluation metrics, and confusion matrix.

## Getting started (running locally)

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the app**

   ```bash
   streamlit run app.py
   ```

3. Open the local URL Streamlit prints (usually `http://localhost:8501`).

The model files (`best_model.pkl`, `encoder.pkl`, `feature_columns.pkl`)
are already included in this repo, so the app runs immediately without
retraining. To retrain the model yourself, open and run
`notebooks/Nutrition_annotated.ipynb` in Google Colab, then replace the
three `.pkl` files with the newly downloaded ones.

## App features

- **Collapsible sections** (Basic Information, Lifestyle & Diet,
  Symptoms, Lab Values) to keep the form compact.
- **Automatic BMI calculation** from height and weight (no manual BMI
  entry needed).
- **Low / Normal / High dropdowns** for lab values instead of raw
  numbers, each with a hover tooltip explaining what it means.
- Responsive layout — fields sit side-by-side on desktop/laptop and
  automatically stack vertically on mobile.
- Shows the predicted disease with a confidence score, a full
  color-coded probability breakdown across all classes (Plotly chart),
  and hover tooltips explaining what each disease actually is.

## Disclaimer

This project is for educational and portfolio purposes only. It is
trained on a sample dataset and **is not a medical diagnostic tool**.
Always consult a qualified healthcare professional for real health
concerns.

## License

MIT
