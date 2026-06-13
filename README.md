# 당뇨병 발병 예측 프로젝트

CDC 생활습관/건강지표 데이터를 활용한 당뇨병 발병 여부 이진 분류 프로젝트

## 데이터셋

- **출처**: [Bena345/cdc-diabetes-health-indicators](https://huggingface.co/datasets/Bena345/cdc-diabetes-health-indicators) (HuggingFace)
- **크기**: 194,825행 × 23컬럼
- **출처 기관**: 미국 CDC(질병통제예방센터) BRFSS 설문 데이터
- **타겟**: `Diabetes_binary` (Non-Diabetic: 86.1% / Diabetic: 13.9%)
- **피처**: BMI, 혈압, 콜레스테롤, 흡연, 음주, 신체활동 등 생활습관/건강지표 21개

## 환경

- Python 3.11
- Anaconda 가상환경: `project`
- GPU 없음 (로컬 CPU)

## 패키지

```
datasets
pandas
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
imbalanced-learn
```

## 프로젝트 구조

```
diabetes_prediction/
├── data/
│   └── diabetes.parquet       # 로컬 캐시 (최초 실행 시 자동 생성)
├── 01_eda.ipynb               # 탐색적 데이터 분석
├── 02_modeling.ipynb          # 전처리, 모델 학습 및 평가
├── model_comparison.png       # 모델 성능 비교 차트
├── confusion_matrix.png       # 혼동 행렬
├── feature_importance.png     # 피처 중요도
├── roc_curves.png             # ROC 커브 비교
└── README.md
```

## 진행 단계

- [x] 환경 세팅 및 패키지 설치
- [x] EDA (데이터 탐색, 분포 확인, 클래스 불균형 확인)
- [x] 전처리 (OrdinalEncoder, StandardScaler, 60/20/20 분할)
- [x] 베이스라인 모델 (Logistic Regression, Random Forest)
- [x] 성능 개선 (XGBoost, LightGBM, RandomizedSearchCV 튜닝)
- [x] 성능 개선 시도 (Threshold 최적화, SMOTE, Top10 피처 선택)
- [x] 결과 정리 및 인사이트 도출

## 최종 결과

### 모델 성능 비교 (검증 세트)

| 모델 | AUC | F1 | Accuracy |
|---|---|---|---|
| **LightGBM_Tuned** | **0.8260** | 0.4422 | 0.7243 |
| XGBoost | 0.8257 | 0.4427 | 0.7261 |
| LightGBM | 0.8255 | 0.4409 | 0.7232 |
| RandomForest | 0.8181 | 0.4412 | 0.7361 |
| LogisticRegression | 0.8081 | 0.4331 | 0.7218 |

### 최종 모델: LightGBM_Tuned (테스트 세트)

| 지표 | 값 |
|---|---|
| AUC | **0.8321** |
| F1 | 0.4478 |
| Accuracy | 0.7301 |
| Recall (당뇨) | **0.79** |
| Precision (당뇨) | 0.31 |

### Threshold 최적화

| 설정 | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| 기본 | 0.500 | 0.308 | 0.784 | 0.442 |
| F1 최적 | 0.629 | 0.368 | 0.631 | 0.465 |
| Recall ≥ 0.90 | 0.342 | 0.253 | 0.900 | 0.395 |

## 주요 발견

- **클래스 불균형**: 당뇨 비율 13.9% → `is_unbalance=True` / `scale_pos_weight` 적용
- **중요 피처**: BMI, GenHlth(전반적 건강상태), Age, Income, PhysHlth 순으로 영향력 높음
- **SMOTE 비효과**: 합성 샘플 증강 시 오히려 Recall 0.78 → 0.21 급락 (클래스 가중치 방식이 우월)
- **성능 상한**: 생활습관 설문 기반 피처의 한계로 AUC 0.83이 실질적 천장

## 한계 및 향후 과제

- 임상 피처(공복혈당, HbA1c 등) 추가 시 성능 대폭 향상 가능
- 스크리닝 목적이면 threshold=0.342 (Recall 0.90) 권장
- SHAP을 활용한 개별 예측 설명력 분석

## 실행 방법

```bash
# 커널 등록 (최초 1회)
D:\Anaconda3\envs\project\python.exe -m ipykernel install --user --name project

# Jupyter 실행 후 project 커널 선택
jupyter notebook
```

> 첫 실행 시 HuggingFace에서 데이터를 다운로드하여 `data/diabetes.parquet`에 저장합니다.  
> 이후 실행부터는 로컬 캐시를 사용하여 빠르게 로드됩니다.
