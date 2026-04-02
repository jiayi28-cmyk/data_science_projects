- This project was completed as part of a group. I was responsible for parts of the exploratory data analysis.
# Diabetes-Prediction
# Objective
- Build and compare machine learning classification models that can accurately identify patients with diabetes using demographic information, clinical measurements, and lifestyle factors.

# Dataset Overview
Target Variable: Diabetes
| Feature | Type | Description |
|---|---|---|
| year | Integer | Year of patient record (2015–2022) |
| gender | Categorical | Female, Male, or Other |
| age | Float | Patient age in years |
| location | Categorical | U.S. state or territory (55 unique) |
| race:AfricanAmerican | Binary | 1 if African American, 0 otherwise |
| race:Asian | Binary | 1 if Asian, 0 otherwise |
| race:Caucasian | Binary | 1 if Caucasian, 0 otherwise |
| race:Hispanic | Binary | 1 if Hispanic, 0 otherwise |
| race:Other | Binary | 1 if Other race, 0 otherwise |
| hypertension | Binary | 1 if patient has hypertension, 0 otherwise |
| heart_disease | Binary | 1 if patient has heart disease, 0 otherwise |
| smoking_history | Categorical | never, current, former, ever, not current, No Info |
| bmi | Float | Body Mass Index (10.01–95.69) |
| hbA1c_level | Float | Hemoglobin A1c level (3.5–9.0), measures avg blood sugar over 2–3 months |
| blood_glucose_level | Integer | Blood glucose level (80–300 mg/dL) |
| **diabetes** | **Binary (Target)** | **1 = has diabetes, 0 = no diabetes** |
# EDA
## Diabetes Class Distribution
<img width="1116" height="490" alt="image" src="https://github.com/user-attachments/assets/c290b47c-7a67-4c55-8630-6c454f2a7b19" />

- Diabetes: 91.5%
  
- Non-Diabetes: 8.5%

**KEY INSIGHT:**

  - Two class is extremely imbalanced, we would apply imbalanced technique to handle this issue in building model stage
## Numeric Features by Diabetes Status
<img width="1380" height="1020" alt="image" src="https://github.com/user-attachments/assets/7bfe3650-ef8a-4b5f-9dc7-1ca1d895057a" />

### Age

- The boxplot shows that individuals with diabetes tend to be **older on average** than those without diabetes.
- The median age for diabetic patients is noticeably higher, suggesting that age may be an important risk factor associated with diabetes.

### BMI

- BMI values are generally **higher among diabetic patients** compared to non-diabetic individuals.
- The median BMI of the diabetic group is clearly larger, indicating that higher body mass index may contribute to increased diabetes risk.

### HbA1c Level

- A strong separation between the two groups can be observed for **HbA1c levels**.
- Diabetic individuals have significantly higher HbA1c values, with most observations exceeding the typical clinical threshold of **6.5**.
- This result aligns with medical knowledge, as HbA1c is commonly used to diagnose diabetes.

### Blood Glucose Level

- Blood glucose levels also show a clear difference between the two groups.
- The diabetic group exhibits substantially higher glucose values, with many observations above **200**, which is consistent with diagnostic criteria for diabetes.

## Diabetes Risk Across Categorical Risk Factors
<img width="1380" height="1020" alt="image" src="https://github.com/user-attachments/assets/8e942836-555b-49b4-8589-b038bdd39d8c" />

### Gender

- The diabetes rate is slightly higher among **male patients** compared to female patients. T
- his suggests that gender may play a modest role in diabetes prevalence, although the difference is relatively small compared to other risk factors.

### Smoking History

- Smoking history shows noticeable differences in diabetes risk.
- Individuals who **formerly smoked or have ever smoked** exhibit higher diabetes rates compared to those who never smoked. 

### Hypertension

- A strong difference is observed between individuals **with and without hypertension**.
- Patients with hypertension have a substantially higher diabetes rate than those without hypertension, indicating that hypertension may be an important comorbidity associated with diabetes.

### Heart Disease

- The most pronounced difference appears in heart disease status.
- Individuals with heart disease show a significantly higher diabetes rate compared to those without heart disease. 

# Feature Engineering

<img width="1990" height="1089" alt="image" src="https://github.com/user-attachments/assets/7a40f682-0c7b-4e4b-823f-452bf49552c5" />



### Dropped 7 Columns (No Predictive Value)
- `year` — near-zero correlation with diabetes (-0.003), data heavily skewed to 2019
- `location` — 55 unique values but diabetes rate barely varies across states (std = 0.0075)
- `race:AfricanAmerican`, `race:Asian`, `race:Caucasian`, `race:Hispanic`, `race:Other` — all ~0.00 correlation with diabetes, identical rates across groups

### Encoded 2 Categorical Features
- `gender` — dropped 18 "Other" rows (0.018% of data), binary encoded (Female=0, Male=1)
- `smoking_history` — ordinal encoded 0–5 based on diabetes risk observed in EDA:

| Code | Category | Diabetes Rate |
|---|---|---|
| 0 | No Info | 4.1% |
| 1 | never | 9.5% |
| 2 | current | 10.2% |
| 3 | not current | 10.7% |
| 4 | ever | 11.8% |
| 5 | former | 17.0% |

### Transformed 1 Feature
- `bmi` — capped at 99th percentile to limit extreme outlier influence (original max was 95.69)

### Created 5 New Features
- Age bins: `age_0-18`, `age_19-35`, `age_36-50`, `age_51-65`, `age_66-80`
- One-hot encoded from continuous age column based on medically meaningful life stages
- Original `age` column dropped to avoid multicollinearity with the bins

### Result
- **Before:** 16 columns, 100,000 rows
- **After:** 14 columns, 99,982 rows, all numeric, model-ready
- Target distribution unchanged: 91.5% No Diabetes / 8.5% Diabetes

# Model Building
## Objective
- Build 5 Model for comparision
    - Bayesian Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost
    - Naive Bayes
 
| Priority | Metric | Why |
|---|---|---|
| Primary | **Recall (minimize FNR)** | A missed diabetic patient (false negative) can lead to severe health complications — kidney failure, blindness, heart disease. This is the high-cost error. |
| Secondary | **Precision (keep FPR reasonable)** | A healthy patient wrongly flagged (false positive) leads to an extra blood test — inconvenient but not dangerous. We want to minimize these without sacrificing recall. |


## Handle Class Imbalance Issue
<img width="1589" height="515" alt="image" src="https://github.com/user-attachments/assets/494b16ab-6135-4c78-941f-798991f0ff3d" />

- We build a Logistic Regression to test which Class Imbalance Technique could improve the overall performance
 
- We decide to choose the simple way to handle Class Imbalance - **Class_Weight = 'Balanced'**

## Model Comparision
### Recall vs. Precision
<img width="1737" height="932" alt="Screenshot 2026-03-11 at 13 12 42" src="https://github.com/user-attachments/assets/326b5f74-8967-476f-8601-c115dca40bd4" />

### Key Findings
- Decision Tree achieves the highest recall (0.928), making it the most effective at identifying diabetic patients.
- Random Forest provides the highest precision (0.985), meaning its predictions are very reliable when it flags a patient as diabetic.
- XGBoost offers a balanced trade-off between recall and precision.
- Bayesian Logistic Regression shows moderate performance.
- Naive Bayes performs the weakest overall.
### Conclusion
- For a screening task where detecting diabetic patients is the priority
- Decision Tree performs best due to its high recall.
- If a more balanced model is required, XGBoost is a strong alternative.



### Error Rate (FPR vs. FNR)
<img width="1735" height="927" alt="Screenshot 2026-03-11 at 13 15 22" src="https://github.com/user-attachments/assets/5eab5269-4ec5-4231-992a-1306510aa705" />



- Every model that pushes FNR down (catches more diabetic patients). Except Random Forest
- Random Forest produces the lowest **False Positive Rate** but **MISSED** 30% **TRUE PATIENT**
- Other model pushes FPR up (Incoorect flags more healthy patients). But **CATH** more **TRUE PATIENT**
- We decide to choose Decision Tree because:
  - Easy Interpretation and identified what key factor driving diabetes
  - Minimizing missed diabetic patients (Lowest FNR)

### Decision Tree Confusion Matrix
<img width="760" height="590" alt="image" src="https://github.com/user-attachments/assets/76bed726-cd40-4d41-be12-1a4813d88f2d" />

- FP: 2,930 healthy people were wrongly classified as diabetes
- FN: 122 diabetes patients we missed


### Choosing Optimal Threshold
**Objective:**
- Industry Standard on False Positve Rate <30%

**Cost of Getting It Wrong**

**False Negative - Missed the Early Intervention Window**: 

- When the model tells a diabetic patient they're healthy, they walk away thinking everything is fine.

- By the time symptoms appear and the patient returns, it's too late for prevention. The hospital now has to treat advanced-stage complications that are far more expensive than early stage.

**False Positive - Follow-Up Diagnosis**

- When the model flags a healthy patient as diabetic, nothing harmful happens. The hospital simply orders a confirmatory blood test

- The patient comes in, gets blood drawn, waits for results, and goes home healthy.

- Total time: 30 minutes to 1 hour. Total cost: ~$75.

<img width="1790" height="617" alt="image" src="https://github.com/user-attachments/assets/df6b8f95-0f15-4203-acd1-a0eeff99b7d9" />

| Threshold | FPR | FNR | FP Count | FN Count |
|---|---|---|---|---|
| 0.10 | 0.258 | 0.017 | 4,720 | 29 |
| 0.20 | 0.214 | 0.031 | 3,913 | 53 |
| 0.30 | 0.214 | 0.031 | 3,913 | 53 |
| 0.40 | 0.214 | 0.031 | 3,913 | 53 |
| 0.50 | 0.160 | 0.072 | 2,930 | 122 |
| 0.60 | 0.046 | 0.211 | 845 | 358 |
| 0.70 | 0.046 | 0.211 | 845 | 358 |
| 0.80 | 0.000 | 0.328 | 0 | 558 |
| 0.90 | 0.000 | 0.328 | 0 | 558 |

- We choose 0.40 as the final threshold because:
  - FNR drops from 0.072 to 0.031 compared to threshold at 0.5. It missed 53 real patients instead of 122
  - FPR of 21.4% — still below the ADA's acceptable range of 30%
  - 69 more diabetic patients caught vs. default threshold
  - Cost of extra false positives is minimal — just a follow-up blood test per patient
 
# How each Feature affect Model Prediction
<img width="769" height="459" alt="image" src="https://github.com/user-attachments/assets/3608e282-1953-4c86-bf03-7e2c2d345a0f" />

## How to Read This Chart

- Each row is a feature, ranked from most important (top) to least important (bottom)
- Color = the patient's actual value for that feature (red = high, blue = low)
- Position on x-axis = how that feature pushed the prediction for that patient (right = toward diabetes, left = away from diabetes)

## Interpretation
- The model relies heavily on two clinical lab tests to make predictions:
  - hbA1c_level and blood_glucose_level show the clearest pattern — high values (red dots) consistently push toward diabetes, low values (blue dots) push away.
  - Age and BMI play a minor supporting role.
  - The remaining features (smoking history, heart disease, hypertension, gender) have almost no influence




