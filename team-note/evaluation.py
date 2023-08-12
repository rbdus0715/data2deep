from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    
    print('오차 행렬\n', confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC: {4:.4f}\n'.format(accuracy, precision, recall, f1, roc_auc))


def get_eval_by_threshold(y_test, pred_proba_positive, thresholds):

    '''
    사용 예시) 
    thresholds = [0.1, 0.5, 0.6] # threshold 값들은 (0, 1) 범위 내
    get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1, 1), thresholds)
    '''
    
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_positive)
        custom_predict = binarizer.transform(pred_proba_positive)
        print('임계값 : ', custom_threshold)
        get_clf_eval(y_test, custom_predict)


def precision_recall_curve_plot(y_test, pred_proba_positive):

    '''prcision_recall_curve_plot
    정밀도와 재현율 트레이드오프 관계를 나타내는 그래프 그리기
    - 이진분류에서 주로 사용
    
    파라미터 : 정답 배열, Positive 칼럼의 예측 확률 배열
    - pred_proba_positive = model.predict_proba(X_test)[:, 1]
    '''
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_positive)

    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='-', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()
