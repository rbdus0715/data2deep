import time
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
import selectivesearch

import utils.util as util


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFilp(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_model(device=None):
    '''
    (1)
    alexnet의 classifier의 마지막 부분을
    원하는 크기(2)로 조정하고 weight를 로드한다.

    (2)
    기울기 추적을 해제하고 model을 eval 상태로 수정한다.
    '''
    model = alexnet()

    num_classes = 2
    # 분류기의 마지막 size를 num_class로 맞춰주기 위해
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)

    model.load_state_dict(
        torch.load('./models/best_linear_svm_alexnet_car.pth'))
    model.eval()

    # 기울기 추적 해제
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


def draw_box_with_text(img, rect_list, score_list):
    '''
    (rect_list 안의 요소 개수만큼 실행)
    파란색으로 박스를 그리고 score를 기입한다.
    '''
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      color=(0, 0, 255), thickness=1)
        cv2.putText(img,
                    "{:.3f}".format(score),
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), 1)


def nms(rect_list, score_list):
    '''nms 수행
    :param rect_list: list, [N, 4]
    :param score_list: list, [N]
    '''
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    # 스코어가 높은 순으로 정렬
    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3
    while len(score_array) > 0:
        nms_rects.append(rect_array[0])
        nms_scores.apend(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        # 예측 박스들을 모두 순회했다면 중지
        length = len(score_array)
        if length <= 0:
            break

        # nms_rects의 마지막 요소와 예측 박스'들'의 iou
        iou_scores = util.iou(
            np.array(nms_rects[len(nms_rects)-1]), rect_array
        )
        # iou 결과가 정해둔 iou thresh를 넘는 것들은 제거 (넘지 않는 것들만 다시 rect_array로 정의)
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores


if __name__ == '__main__':
    '''
    TODO_1 : test_img의 정답 박스 그리기
    TODO_2 : 예측박스 구하기
        (1) selective search 알고리즘을 통해 region proposal
        (2) 각각의 proposal들을 모델에 통과시켜 자동차로 예측이 된 경우만 박스와 score를 저장한다.
    TODO_3 : 제시된 proposal들이 너무 많으므로 nms를 수행하고 결과를 출력한다.
    '''
    device = get_device()
    transform = get_transform()
    model = get_model(device=device)

    gs = selectivesearch.get_selective_search()

    test_img_path = '../imgs/000012.jpg'
    test_xml_path = '../imgs/000012.xml'
    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    # TODO_1 : test_img의 정답 박스 그리기
    bndboxs = util.parse_xml(test_xml_path)
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        # 정답 : 초록색
        cv2.rectangle(dst,
                      (xmin, ymin), (xmax, ymax),
                      color=(0, 255, 0), thickness=1)

    # TODO_2 : 예측박스 구하기
    # (1) selective search 알고리즘을 통해 region proposal
    selectivesearch.config(gs, img, strategy='f')
    rects = selectivesearch.get_rects(gs)
    print('후보 영역 수: %d' % len(rects))

    # (2) 각각의 proposal들을 모델에 통과시켜 자동차로
    # 예측이 된 경우만 박스와 score를 저장한다.
    svm_thresh = 0.60
    score_list = list()
    positive_list = list()

    start = time.time()
    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]
        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]  # 크기 : (Batch, 2)

        # 자동차인 경우만
        if torch.argmax(output).item() == 1:
            probs = torch.softmax(output, dim=0).cpu().numpy()
            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)
                print(rect, output, probs)
    end = time.time()
    print('detect time: %d s' % (end-start))

    # TODO_3 : 제시된 proposal들이 너무 많으므로 nms를 수행하고 결과를 출력한다.
    nms_rects, nms_scores = nms(positive_list, score_list)
    print(nms_rects)
    print(nms_scores)
    draw_box_with_text(dst, nms_rects, nms_scores)

    cv2.imshow('img', dst)
    cv2.waitKey(0)