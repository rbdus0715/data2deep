## CO2 배출량 예측하기

### 데이터 분석 단계


**(1) 통계 요약을 통해서 데이터에 대한 전체적인 감을 잡는다. 이때 최대 최소에 대한 정보를 정리해두기!**
```python
# 통계 요약 statistical summaries 할때 더 자세한 정보를 얻을 수 있음
df.describe(include='all')
```


**(2) 타겟 데이터의 분포와 왜도 확인: sns.histplot**
- 왜도(skewness): 데이터 분포의 비대칭성을 나타내는 지표
  - 0에 가까움: 대칭 | 양수: 왼쪽으로 치우침 | 음수: 오른쪽으로 치우침

![imag](https://www.gstatic.com/education/formulas2/553212783/en/skewness.svg) 
- $x_i$: 데이터 포인트
- $\bar{x}$: 평균
- $\sigmaσ$: 데이터의 표준편차
- $n$: 데이터 포인트의 수
​- 해결방법
```python
# 타깃값의 분포 확인하기 
sns.set_style('darkgrid') # 다양한 옵션 : darkgrid, whitegrid, dark, white, ticks
plt.figure(figsize=(13, 7))
sns.histplot(train.emission, kde = True, bins = 15) # bins : 막대의 개수
plt.title('타겟 분포', y=1.02, fontsize=15)
display(plt.show(), train.emission.skew()) # display 함수는 대화형 환경에서의 출력함수
```
- 왜도 현상 해결법
  - 로그변환, box-cox 변환, 루트변환


 **(3) 이상치 확인하기 [공부했던 링크](https://github.com/rbdus0715/Machine-Learning/blob/main/study/sklearn/creditcard_fraud.ipynb)**
![imag](https://www.simplypsychology.org/wp-content/uploads/box-whisker-plot.jpg)
- 그래프 해석
  - 상자: 데이터의 사분위수 범위 Q1~Q3
  - 수염: 이상치의 경계, 보통 1.5배 사분위범위를 사용하여 길이를 정함
  - 이상치: 수염의 범위를 벗어나느 값
  - 상자의 위치에 따라서 데이터의 분포도도 파악 가능
```python
sns.set_style('dark')
plt.figure(figsize=(13,7))
sns.boxplot(train.emission)
plt.title('target data outliers check', y=1.02, fontsize=15)
plt.show()
```

**(4) 데이터에 위도와 경도 수치가 주어진 상황, 지리 정보 시각화 (geopandas)**
- [geopandas with folium 사용설명서](https://geopandas.org/en/stable/gallery/plotting_with_folium.html)

**(5) 피처 데이터 개수 확인**
- 개수 분포를 알고싶은 피처: col1
```python
plt.figure(figsize=(14, 7))
sns.countplot(x='col1', data=df)
```

**(6) 타겟 데이터와 가장 상관관계가 큰 피처 Top 20 시각화**















