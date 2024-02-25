'''
# cv2 설치 :
    - $ pip uninstall opencv-contrib-python opencv-python
    - $ pip install opencv-contrib-python
# cv2 selectivesearch docs :
https://docs.opencv.org/3.4/d6/d6d/classcv_1_1ximgproc_1_1segmentation_1_1SelectiveSearchSegmentation.html
'''
import sys
import cv2


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects


if __name__ == '__main__':
    '''
    lena로 selective search 테스트해보기
    '''
    gs = get_selective_search()
    # img = cv2.imread('./src/lena.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='q')

    rects = get_rects(gs)
    print(rects)
