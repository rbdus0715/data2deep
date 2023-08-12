## 데이터 탐색
- 분류 문제에서 결과 비율이 어떻게 된는지 : df[결과].value_counts()
- null 값 확인하기 data.info()

## 훈련
- 로지스틱 회귀의 경우 일반적으로 숫자 데이터에 스케일링을 적용하는 것이 좋음

## 평가
- 이진분류 문제에서 평가 : [evaluation.get_clf_eval](https://github.com/rbdus0715/Machine-Learning/blob/main/team-note/evaluation.py)


## 피드백
- 데이터 탐색에서 본 결과 비율에 따라 정밀도, 재현율 어느 것에 초점을 맞출 것인지 >> 임계값 조절
- df.describe()의 분포를 확인해보고 min, max 처럼 극단적인 수치에 집중한다.</br> 이 값이 현실적으로 가능한 값인지, 모델 학습에 도움이 되는 방향인지 확인하고 삭제하거나 평균값으로 대체한다.
    
