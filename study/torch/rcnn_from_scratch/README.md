# 참고 코드와 DOCS
[github 링크](https://github.com/object-detection-algorithm/R-CNN/tree/master/py)
[DOCS](https://r-cnn.readthedocs.io/zh-cn/latest/)
[VOC2007 torchvision DOCS](https://blog.zhujian.life/posts/5a56cd45.html)

# 개발 순서 (코드 읽는 순서)
car_detector
selectivesearch
utils/data/create_finetune_dataset
utils/data/pascal_voc_car.py, pascal_voc
utils/data/custom_batch_sampler
utils/data/create_finetune_data
utils/data/custom_finetune_dataset
utils/util - parse_xml, iou, compute_ious
utils/data/create_classifier_data

# 파일 실행 순서
python pascal_voc.py
pip install xmltodict
python pascal_voc_car.py
