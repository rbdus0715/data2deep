from PIL import Image
from matplotlib.pyplot import imshow

class ImagePreprocessing:
  def __init__(self):
    need_import()
    show_utils()

  import_list = ["PIL.Image", "matplotlib.pyplot.imshow"]
  
  def need_import():
    print("you need to import {}".format(import_list))


  
  # 이미지 불러오기
  def open_image(path="Lenna.png"):
    img = Image(path)
    try: img.show()
    except: imshow(img)

  # 사이즈, 이미지의 컬러, 흑백 여부 확인 확인
  # 모드에 관한 내용 docs : https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
  def color_mode(img):
    print("Size: {}, Mode: {}".format(img.size, img.mode))

  
  def crop(img, coordinates):
    '''사용방법
    img : PIL.Image 형태
    coordinates : (x1, y1, x2, y2)
      좌측 상단 좌표 (x1 y1), 우측 하단 좌표 (x2, y2)
    '''
    img_crop = img.crop(coordinates)
    return img_crop
  
  def  
