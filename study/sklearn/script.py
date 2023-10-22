# 1. 붓꽃 품종 예측
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
iport pandas as pd
iris = load_iris()
iris_data = iris.data
iris_label = iris.target
df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
df['label'] = iris_label
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# 2. 교차검증 간편하게 하기 -> cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)
scores = cross_val_score(dt_clf, features, label, scoring='accuracy', cv=3)
print(np.round(np.mean(scores), 4))


# 3. 그리드서치CV 교차검증 + 하이퍼 파라미터 튜닝
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=121)
dtree = DecisionTreeClassifier()
parameters = {
    'max_depth' : [1, 2, 3],
    'min_samples_split' : [2, 3]
}
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)
grid_dtree.fit(X_train, y_train)
scores_df = pd.DataFrame(grid_dtree.cv_results_) # 결과확인
print(grid_dtree.best_params_) # 최고 파라미터 조합
print(grid_dtree.best_score_) # 최고 점수
estimator = grid_dtree.best_estimator_ # 최종 모델
pred = estimator.predict(X_test) # 최종 모델로 예측
print(accuracy_score(y_test, pred))
