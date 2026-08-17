# Phense — Machine Learning Walkthrough Guide
**Prepared for:** Adeniji Yusuf | `FE/24/5369659950` | 3MTT NextGen Cohort  
**Project Track:** AI & Machine Learning  
**Project ID:** AI-19 — Flood Risk Classifier  

---

This guide is designed as an educational walkthrough to help you implement the machine learning training pipeline (`train.py`) yourself, understand the underlying algorithms, and prepare your project defense for the 3MTT panel.

---

## 1. Setup & Environment Verification

Before starting, ensure that all dependencies are installed in your virtual environment:
```bash
pip install -r requirements.txt
```
The real-world training dataset has been generated and is stored at `data/processed/nigeria_flood_training.csv`.

---

## 2. Walkthrough: The ML Training Pipeline

Open `train.py`. The script is structured into 5 clean, distinct phases. Here is how to write and understand each step:

### Phase 1: Loading the Dataset
We load the primary training CSV from the processed data directory. We use the custom utility `load_raw` in [data_utils.py](file:///c:/Users/USER/Documents/Python Project/phense/src/data_utils.py):
```python
from src.data_utils import load_raw
df_raw = load_raw()
```
*Why this matters:* It separates data loading concerns from training logic, making the code modular and clean.

### Phase 2: Data Cleaning & Validation
Real-world spatial data can contain whitespace anomalies, lowercase inconsistencies, or invalid coordinates. We clean and validate:
```python
from src.data_utils import clean, validate
df = clean(df_raw)
validate(df)
```
* Hydrological boundary checks: We assert that categorical variables match valid sets (e.g. soil types are `clay`, `sandy`, `loamy`, or `silty`).
* Outlier removal: We enforce that rainfall mm is between `[0, 800]` and elevation is between `[0, 1500]` meters to prevent erroneous measurements from skewing model weights.

### Phase 3: Ordered Label Encoding
Machine learning algorithms like Random Forest require numeric inputs. We encode string columns into integers:
* **Soil Type**: `clay` = 0, `loamy` = 1, `sandy` = 2, `silty` = 3
* **Drainage Quality**: `none` = 0, `basic` = 1, `moderate` = 2, `good` = 3
* **Risk Class (Target)**: `Low` = 0, `Medium` = 1, `High` = 2, `Critical` = 3

*Rigor Note:* We use a **fixed order mapping** (not alphabetical) so that the integer values map directly to physical severity (e.g. `none` drainage is a higher hazard than `good` drainage, and `clay` soil has higher flood retention).

```python
df_enc, encoders = encode_features(df)
X, y = get_feature_matrix(df_enc)
```

We split our data into an 80/20 train-test configuration:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```
*Defense Tip:* Always specify `stratify=y`. Because critical flooding events are naturally less frequent than low-risk days, stratification ensures that the proportion of Low, Medium, High, and Critical classes is identical in both the training and test sets.

### Phase 4: Model Selection & Hyperparameter Tuning
We use a **Random Forest Classifier** because composite physical risks are non-linear. (For example, a low elevation is only a critical flood hazard *if* it is accompanied by clay soil and poor drainage during heavy rainfall). Simple linear regressions fail to capture these logical thresholds.

To find the best parameters, we run a grid search with cross-validation:
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
    "class_weight":      ["balanced"], # Vital for handling class imbalance
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
```

### Phase 5: Model Evaluation & Inspection
After fitting the model, we measure performance on the held-out test set:
```python
y_pred = model.predict(X_test)
```
* **Accuracy Score**: The proportion of correct predictions. (Expect ~80-86% on real-world geo-spatial datasets).
* **Confusion Matrix**: Shows where the model misclassified. (e.g. did it mistake a Critical risk for a Medium risk?).
* **Feature Importances**: Inspect which variables have the highest predictive weight.
```python
importances = model.feature_importances_
```
*Defensive Strategy:* In a composite risk framework, **Elevation** and **River Proximity** should emerge as highly significant drivers alongside rainfall. If rainfall was the only factor, the model would simply act as a rain-gauge, not a composite flood model.

### Phase 6: Serialization
We serialize the trained model and encoders so that the FastAPI backend (`app.py`) can load them to serve predictions:
```python
import joblib
joblib.dump(model, "models/phense_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")
```

---

## 3. Defense Preparation: Answering Judge Questions

When presenting your capstone project, judges may grill you on your design decisions. Here is how to defend them:

### Q1: "Why did you use a Random Forest instead of a neural network or simple regression?"
* **Answer:** *"Flood risk is composite and threshold-based. A neural network would require a massive dataset to learn these physics boundaries and is a black box. Logistic regression assumes linear relationships. Random Forest uses decision splits (e.g., elevation < 15m AND distance to river < 1km), which matches the physical physics of flooding, and provides feature importance metrics."*

### Q2: "Where did your data come from? Is it synthetic?"
* **Answer:** *"The training data is built from real geographical coordinates of Nigerian LGA centroids. The elevation was fetched from satellite data via the NASA SRTM 30m database (Open Topo Data API), rainfall averages are based on NiMet seasonal archives, soil types match the FAO Soil Map of Nigeria, and historical flood frequencies correspond to NEMA flood bulletins. It represents a real-world geo-enriched pipeline, not arbitrary synthetic data."*

### Q3: "How does the model handle dry vs wet season dynamics?"
* **Answer:** *"The machine learning model outputs a base risk class. A post-prediction modifer adjusts the label for season. In the wet season, saturated ground acts as an amplifier, upgrading a medium-probability event into a Medium-High hazard to warn the community of immediate runoff risk."*
