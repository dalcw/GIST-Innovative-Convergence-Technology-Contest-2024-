# GIST-Innovative-Convergence-Technology-Contest-2024-
2024 제 8회 GIST 창의 융합 경진대회
- 팀명: AICONIC
- 참가자: 문성수(전남대학교), 나유경(전남대학교)

## Challenges
1. 부하 예측 알고리즘 개발 및 검증
2. 태양광 발전량 예측 알고리즘 개발 및 검증
3. 전기 요금 최소화 알고리즘 개발 및 검증


## Requirements
`pip install -r requirements.txt`
```
numpy, pandas, holidays, matplotlib, torch, tqdm, xgboost, sklearn, cvxpy
```

## PATH
```
├─Challenge01
│  │  2022_train.csv: 2022 학습 데이터 셋 (기상 정보 + 건물별 전력 사용량)
│  │  2023_train.csv: 2023 학습 데이터 셋 (기상 정보 + 건물별 전력 사용량)
│  │  data_preprocessing_2022.py: excel 파일을 csv로 정리하는 코드
│  │  data_preprocessing_2023.py: excel 파일을 csv로 정리하는 코드
│  │  deep_learning_modeling.py: 딥러닝 모델 (CNN-LSTM)
│  │  elec_inference_result.csv: 2023.08.31.에 대한 전력 사용량 추론 결과
│  │  elec_max.pickle: min-max 정규화를 위한 건물별 전력 사용량 최댓값
│  │  elec_min.pickle: min-max 정규화를 위한 건물별 전력 사용량 최솟값
│  │  ensemble.py: XGBoost + CNN-LSTM
│  │  inference.csv: 추론을 위한 입력 데이터
│  │  inference_dataset.py: inference.csv를 생성
│  │  machine_learning_modeling.py: 머신러닝 모델 (XGBoost)
│  │
│  └─model_result
│      ├─cnnlstm: CNN-LSTM 학습 weight 저장
│      └─machinelearning: XGBoost 모델 저장
│
├─Challenge02
│  │  2022_train.csv: 2022 학습 데이터 셋 (기상 정보 + 태양광 발전량)
│  │  2022_weather.csv: 2022 날씨 정보
│  │  2023_train.csv: 2023 학습 데이터 셋 (기상 정보 + 태양광 발전량)
│  │  2023_weather.csv: 2023 날씨 정보
│  │  data_preprocessing_2022.py: excel 파일을 csv로 정리하는 코드
│  │  data_preprocessing_2023.py: excel 파일을 csv로 정리하는 코드
│  │  deep_learning_modeling.py: 딥러닝 모델 (CNN-LSTM)
│  │  ensemble.py: XGBoost + CNN-LSTM
│  │  gen_max.pickle: min-max 정규화를 위한 발전량 최댓값
│  │  gen_min.pickle: min-max 정규화를 위한 발전량 최솟값
│  │  inference.csv: 추론을 위한 입력 데이터 셋
│  │  inference_dataset.py: inference.csv 생성
│  │  machine_learning_modeling.py: 머신러닝 모델 (XGBoost)
│  │  solargrid_inference_result.csv: 2023.08.31.에 대한 전력 생성량 추론 결과
│  │
│  └─model_result
│      ├─cnnlstm: CNN-LSTM 학습 weight 저장
│      └─machinelearning: XGBoost 모델 저장
│
└─Challenge03
        elec_inference_result.csv: 2023.08.31.에 대한 전력 사용량 추론 결과
        optimization.py: 2023.08.31. 일시의 예측 결과를 가지고 전기 요금 최적화
        solargrid_inference_result.csv: 2023.08.31.에 대한 전력 생성량 추론 결과
        temp.py: 임시 파일
```
