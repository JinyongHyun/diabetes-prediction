# 당뇨병 발병 예측 프로젝트

> 병원 방문 없이 수집 가능한 생활습관 데이터만으로 당뇨 고위험군을 선별할 수 있는가?

CDC BRFSS 설문 데이터를 활용한 당뇨병 발병 여부 이진 분류 프로젝트

## 데이터셋

| 항목 | 내용 |
|---|---|
| 출처 | [Bena345/cdc-diabetes-health-indicators](https://huggingface.co/datasets/Bena345/cdc-diabetes-health-indicators) (HuggingFace) |
| 크기 | 194,825행 × 23컬럼 |
| 출처 기관 | 미국 CDC(질병통제예방센터) BRFSS 설문 데이터 |
| 타겟 | `Diabetes_binary` (Non-Diabetic: 86.1% / Diabetic: 13.9%) |
| 피처 | BMI, 혈압, 콜레스테롤, 흡연, 음주, 신체활동 등 생활습관/건강지표 21개 |

## 환경

- Python 3.11 / Anaconda 가상환경 `project` / CPU Only

## 패키지

```
datasets pandas scikit-learn xgboost lightgbm matplotlib seaborn imbalanced-learn kaggle
```

## 프로젝트 구조

```
diabetes_prediction/
├── data/
│   ├── diabetes.parquet                 # CDC 데이터 로컬 캐시 (최초 실행 시 자동 생성)
│   └── diabetes_prediction_dataset.csv  # Kaggle 임상 피처 데이터
├── 01_eda.ipynb                         # 탐색적 데이터 분석 (한국어 해석 포함)
├── 02_modeling.ipynb                    # 전처리, 모델 학습 및 평가
├── 03_clinical_features.ipynb           # 임상 피처 실험 및 Data Leakage 분석
├── model_comparison.png
├── confusion_matrix.png
├── feature_importance.png
├── roc_curves.png
├── clinical_confusion_matrix.png
├── clinical_feature_importance.png
├── clinical_pr_curve.png
├── clinical_vs_cdc_comparison.png
└── README.md
```

## 진행 단계

- [x] EDA (데이터 탐색, 분포 확인, 클래스 불균형 확인)
- [x] 전처리 (OrdinalEncoder, 60/20/20 분할)
- [x] 베이스라인 모델 (Logistic Regression, Random Forest)
- [x] 성능 개선 (XGBoost, LightGBM, RandomizedSearchCV 튜닝)
- [x] 추가 실험 (Threshold 최적화, SMOTE, Top10 피처 선택)
- [x] 임상 피처 실험 및 Data Leakage 검증

---

## 실험 1 — 생활습관 기반 모델 (CDC BRFSS)

### 모델 성능 비교 (검증 세트)

| 모델 | AUC | F1 | Accuracy |
|---|---|---|---|
| **LightGBM_Tuned** | **0.8260** | 0.4422 | 0.7243 |
| XGBoost | 0.8257 | 0.4427 | 0.7261 |
| LightGBM | 0.8255 | 0.4409 | 0.7232 |
| RandomForest | 0.8181 | 0.4412 | 0.7361 |
| LogisticRegression | 0.8081 | 0.4331 | 0.7218 |

### 최종 모델: LightGBM_Tuned (테스트 세트)

| 지표 | 기본 (threshold=0.5) | 스크리닝 (threshold=0.342) |
|---|---|---|
| AUC | **0.8321** | **0.8321** |
| Recall | 0.784 | **0.900** |
| Precision | 0.308 | 0.253 |
| F1 | 0.442 | 0.395 |
| Accuracy | 0.730 | 0.655 |

> **Precision/F1이 낮은 이유**: 클래스 비율 13.9%라는 구조적 한계 + 스크리닝 목적으로 Recall을 우선하는 의도적 선택.
> 이 모델의 주 평가 지표는 AUC와 Recall이다.

### 핵심 의사결정

**SMOTE vs 클래스 가중치**
대부분 범주형(Yes/No) 피처로 구성된 데이터에서 SMOTE 적용 시 Recall이 **0.78 → 0.21로 급락**.
연속형 피처 보간을 가정하는 SMOTE가 비현실적인 합성 샘플을 생성했기 때문.
→ `is_unbalance=True` (클래스 가중치 방식) 채택.

**Threshold 최적화**

| 목적 | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| 기본 | 0.500 | 0.308 | 0.784 | 0.442 |
| F1 최적 | 0.629 | 0.368 | 0.631 | 0.465 |
| 스크리닝 (Recall ≥ 0.90) | 0.342 | 0.253 | **0.900** | 0.395 |

---

## 실험 2 — 임상 피처 실험 및 Data Leakage 분석

Kaggle [diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (100,000행)에 HbA1c, 혈당 수치를 추가하여 실험.

| 지표 | CDC BRFSS (생활습관) | 임상 피처 (HbA1c+혈당) | 향상 |
|---|---|---|---|
| AUC | 0.8321 | 0.9782 | +0.146 |
| F1 | 0.4478 | 0.6531 | +0.205 |
| Recall | 0.790 | 0.894 | +0.104 |

### ⚠️ 왜 이 결과를 그대로 받아들이지 않았는가

HbA1c ≥ 6.5%, 혈당 ≥ 126 mg/dL이 당뇨 **진단 기준 자체**다.
타겟 레이블이 이 수치를 기반으로 부여됐다면, 모델이 학습한 것이 아니라 진단 기준을 재현한 것이다.

**→ Data Leakage. AUC 0.98은 좋은 모델의 증거가 아니다.**

AUC 0.83의 생활습관 기반 모델이 더 어렵고 더 가치 있는 문제를 푼다.

---

## 주요 발견

- **성능 상한**: 생활습관 설문 기반 피처의 한계로 AUC 0.83이 실질적 천장
- **중요 피처**: BMI > GenHlth > Age > Income > PhysHlth
- **SMOTE**: 범주형 중심 데이터에서 역효과 — 클래스 가중치 방식이 우월
- **Data Leakage**: 임상 진단 기준 수치를 피처로 사용 시 높은 AUC는 무의미

## 실행 방법

```bash
# 커널 등록 (최초 1회)
D:\Anaconda3\envs\project\python.exe -m ipykernel install --user --name project

# Jupyter 실행 후 project 커널 선택
jupyter notebook
```

> `01_eda.ipynb` → `02_modeling.ipynb` 순서로 실행.  
> 첫 실행 시 HuggingFace에서 데이터를 다운로드하여 `data/diabetes.parquet`에 저장.  
> `03_clinical_features.ipynb`는 Kaggle 로그인 후 `data/diabetes_prediction_dataset.csv` 필요.
