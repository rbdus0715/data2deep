# 참고 코드와 DOCS
[github 링크](https://github.com/object-detection-algorithm/R-CNN/tree/master/py)

[DOCS](https://r-cnn.readthedocs.io/zh-cn/latest/)

[VOC2007 torchvision DOCS](https://blog.zhujian.life/posts/5a56cd45.html)


# 파일 실행 순서

```python
# 데이터셋 구축
python pascal_voc.py
python pascal_voc_car.py
python create_finetune_data.py
python create_classifier_data.py
python create_bbox_regression_data.py

# train
python finetune.py
python linear_svm.py
python bbox_regression.py

# 예측
car_detector.py
```

