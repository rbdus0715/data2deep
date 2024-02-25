import os
import shutil
import numpy as np
import utils.util as util

# postive bbox 개수：37222

if __name__ == '__main__':
    """
    voc_car/train 에서 bbox 좌표 추출하기
    finetune_car/train에서 훈련 세트의 양성 샘플 좌표 (IoU>=0.5) 추출, IoU>0.6의 경계 상자를 추가로 추출한다.
    bbox_car/ 디렉터리에 저장
    """
    voc_car_train_dir = '../../data/voc_car/train'
    # ground truth
    gt_annotation_dir = os.path.join(voc_car_train_dir, 'Annotations')
    jpeg_dir = os.path.join(voc_car_train_dir, 'JPEGImages')

    classifier_car_train_dir = '../../data/finetune_car/train'
    # positive
    positive_annotation_dir = os.path.join(classifier_car_train_dir, 'Annotations')

    dst_root_dir = '../../data/bbox_regression/'
    dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
    dst_bndbox_dir = os.path.join(dst_root_dir, 'bndboxs')
    dst_positive_dir = os.path.join(dst_root_dir, 'positive')

    util.check_dir(dst_root_dir)
    util.check_dir(dst_jpeg_dir)
    util.check_dir(dst_bndbox_dir)
    util.check_dir(dst_positive_dir)

    samples = util.parse_car_csv(voc_car_train_dir)
    res_samples = list()
    total_positive_num = 0
    for sample_name in samples:
        # positive bbox 좌표 추출 (IoU >= 0.5)
        positive_annotation_path = os.path.join(positive_annotation_dir, sample_name + '_1.csv')
        positive_bndboxes = np.loadtxt(positive_annotation_path, dtype=np.int, delimiter=' ')
        # 정답 경계 상자 추출
        gt_annotation_path = os.path.join(gt_annotation_dir, sample_name + '.xml')
        bndboxs = util.parse_xml(gt_annotation_path)
        # IoU>0.6으로 다시 추출
        positive_list = list()
        if len(positive_bndboxes.shape) == 1 and len(positive_bndboxes) != 0:
            scores = util.iou(positive_bndboxes, bndboxs)
            if np.max(scores) > 0.6:
                positive_list.append(positive_bndboxes)
        elif len(positive_bndboxes.shape) == 2:
            for positive_bndboxe in positive_bndboxes:
                scores = util.iou(positive_bndboxe, bndboxs)
                if np.max(scores) > 0.6:
                    positive_list.append(positive_bndboxe)
        else:
            pass

        # pos (iou>0.6)이 있는 경우 저장
        if len(positive_list) > 0:
            jpeg_path = os.path.join(jpeg_dir, sample_name + ".jpg")
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + ".jpg")
            shutil.copyfile(jpeg_path, dst_jpeg_path)
            # 레이블 경계 상자 저장
            dst_bndbox_path = os.path.join(dst_bndbox_dir, sample_name + ".csv")
            np.savetxt(dst_bndbox_path, bndboxs, fmt='%s', delimiter=' ')
            # 원본 경계 상자 저장
            dst_positive_path = os.path.join(dst_positive_dir, sample_name + ".csv")
            np.savetxt(dst_positive_path, np.array(positive_list), fmt='%s', delimiter=' ')

            total_positive_num += len(positive_list)
            res_samples.append(sample_name)
            print('save {} done'.format(sample_name))
        else:
            print('-------- {} not eligible'.format(sample_name))

    dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
    np.savetxt(dst_csv_path, res_samples, fmt='%s', delimiter=' ')
    print('total positive num: {}'.format(total_positive_num))
    print('done')
