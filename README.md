# 🌾 Ensemble-Based Crop Recommendation System

> An end-to-end machine learning system that predicts the optimal crop based on
> soil nutrients and climate conditions using a soft-voting ensemble of Random Forest,
> Extra Trees, and Gradient Boosting classifiers.

---

## 📊 Results

| Model              | Accuracy | Precision | Recall | F1     |
|--------------------|----------|-----------|--------|--------|
| Random Forest      | 98.79%   | 98.97%    | 98.79% | 98.80% |
| Extra Trees        | 98.94%   | 99.10%    | 98.94% | 98.94% |
| Gradient Boosting  | 98.03%   | 98.26%    | 98.03% | 98.05% |
| **Soft Voting Ensemble** | **99.24%** | **99.35%** | **99.24%** | **99.25%** |

Cross-validation (10-fold, full dataset):  
**Ensemble: 99.64% ± 0.18%**

---

## 🗂 Project Structure

```
crop_recommender/
│
├── crop_recommendation.csv     # Dataset (2200 samples, 22 crops)
│
├── train_pipeline.py           # ← Run this first to train & save the model
├── app.py                      # ← Then run this to start the Flask API
│
├── src/
│   ├── data_loader.py          # Load + EDA summary
│   ├── preprocessing.py        # Encoding, scaling, split, imbalance check
│   └── model_trainer.py        # Tuning, CV, ensemble, evaluation, saving
│
├── models/                     # Auto-created after training
│   ├── ensemble_model.joblib   # Saved soft-voting ensemble
│   ├── scaler.joblib           # Fitted StandardScaler
│   └── label_encoder.joblib    # LabelEncoder for 22 crops
│
├── templates/
│   └── index.html              # Frontend UI
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `xgboost` is unavailable (offline/restricted environment),
> the pipeline automatically falls back to `GradientBoostingClassifier`
> which is included in scikit-learn and achieves equivalent performance.

### 2. Train the model

```bash
python train_pipeline.py
```

This will:
- Load and EDA the dataset
- Stratified 70/30 train/test split
- Scale features with StandardScaler
- Check class imbalance (SMOTE if needed)
- Tune each model with RandomizedSearchCV (15 iter × 5-fold CV)
- Build soft-voting ensemble
- Run 10-fold cross-validation
- Print full evaluation metrics & classification report
- Save `ensemble_model.joblib`, `scaler.joblib`, `label_encoder.joblib`

### 3. Start the Flask API

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🌐 API Reference

### `POST /predict`

**Request body (JSON):**

```json
{
  "N":           90,
  "P":           42,
  "K":           43,
  "temperature": 20.8,
  "humidity":    82.0,
  "ph":          6.5,
  "rainfall":    202.9
}
```

**Response:**

```json
{
  "predicted_crop": "rice",
  "confidence":     0.9821,
  "confidence_pct": "98.2%",
  "emoji":          "🌾",
  "top_3": [
    { "crop": "rice",   "probability": 0.9821, "emoji": "🌾" },
    { "crop": "jute",   "probability": 0.0112, "emoji": "🌱" },
    { "crop": "cotton", "probability": 0.0067, "emoji": "🌿" }
  ],
  "input_received": { "N": 90, "P": 42, ... }
}
```

### `GET /health`
Returns API status and number of supported crops.

### `GET /crops`
Returns list of all 22 supported crops.

---

## 🧠 ML Architecture

### Dataset
- **Source:** Kaggle Crop Recommendation Dataset
- **Samples:** 2,200 (100 per crop, perfectly balanced)
- **Features:** N, P, K, Temperature, Humidity, pH, Rainfall
- **Classes:** 22 crops

### Pipeline Steps

1. **Data Loading** — validation, EDA
2. **Label Encoding** — LabelEncoder for 22 string classes
3. **Train/Test Split** — 70/30, stratified
4. **Feature Scaling** — StandardScaler
5. **Imbalance Check** — SMOTE applied only if ratio > 2.0 (not needed here)
6. **Hyperparameter Tuning** — RandomizedSearchCV (15 iter × 5-fold CV per model)
7. **Cross-Validation** — 10-fold stratified on full dataset
8. **Soft Voting Ensemble** — probability averaging across 3 tuned models
9. **Evaluation** — Accuracy, Precision, Recall, F1, Classification Report

### Ensemble Members

| Model                 | Type        | Key Params                           |
|-----------------------|-------------|--------------------------------------|
| RandomForestClassifier | Bagging    | 100 trees, log2 features, depth 30   |
| ExtraTreesClassifier   | Bagging    | 300 trees, sqrt features             |
| GradientBoostingClassifier | Boosting | 100 trees, lr 0.05, depth 7       |

**Voting:** Soft (probability averaging) — allows weighted confidence scores.

---

## 🌿 Supported Crops

Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes,
Jute, Kidney Beans, Lentil, Maize, Mango, Mothbeans, Mungbean,
Muskmelon, Orange, Papaya, Pigeonpeas, Pomegranate, Rice, Watermelon

---

## 📈 Feature Importance (Random Forest)

The most important features for crop prediction are:
1. **Potassium (K)** — differentiates high-K crops like banana
2. **Rainfall** — key separator between dry/wet crops
3. **Humidity** — critical for tropical vs arid crops
4. **Temperature** — separates seasonal crops
5. **pH** — especially important for coffee and cotton

---

## 🔧 Configuration

Environment variable:
```bash
PORT=8080 python app.py   # run on custom port
```

---

## 📄 License

MIT — free for academic and commercial use.
