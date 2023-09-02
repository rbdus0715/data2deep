## 데이터 탐색
- **features**
    - 피처가 엄청나게 많은 경우 피처의 중복이 있을 수 있으므로 해당 중복되는 피처들 뒤에 _1, _2 ... 을 붙여서 구분
- **target**
    - 분류 문제에서 target 비율 확인 : df['target'].value_counts()
    - target 데이터의 이상치 확인하기
    - 데이터 불균형
        - 데이터 분포가 한쪽으로 치우쳐진 경우 정규 분포 형태로 변환하는 작업 필요 >> StandardScaler
        - 데이터 분포도가 심하게 왜곡되어 있을 경우 log 변환 사용 >> 원래 큰 값을 상대적으로 작은 값으로 변환시킴 >> 넘파이의 log1p()
        - 오버샘플링 >> imblearn.over_sampling.SMOTE
            - SMOTE의 경우 재현율은 높아지나 정밀도는 낮아지는 것이 일반적

- **null**
    - null 값 확인하기 data.info()
    - data.isnull().sum()
    - seaborn.heatmap(data.isnull())

- **시각화**
    - 지리적 위치의 시각화 : geopandas

## 훈련
- 로지스틱 회귀의 경우 일반적으로 숫자 데이터에 스케일링을 적용하는 것이 좋음

## 평가
- 이진분류 문제에서 평가 : [evaluation.get_clf_eval](https://github.com/rbdus0715/Machine-Learning/blob/main/team-note/evaluation.py)
- 이진분류 문제에서 threshold(임계값)에 따른 성능 확인하고 선택하기

## 피드백
- 데이터 탐색에서 본 결과 비율에 따라 정밀도, 재현율 어느 것에 초점을 맞출 것인지 >> 임계값 조절
- df.describe()의 분포를 확인해보고 min, max 처럼 극단적인 수치에 집중한다.</br> 이 값이 현실적으로 가능한 값인지, 모델 학습에 도움이 되는 방향인지 확인하고 삭제하거나 평균값으로 대체한다.
    
