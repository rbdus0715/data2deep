![0_WM0eI6d0vvWt0UBr](https://github.com/rbdus0715/Machine-Learning/assets/85426187/adfa297c-ffe5-427a-b384-50ea43137fd9)

# study
- [simpleNN](https://github.com/rbdus0715/Machine-Learning/blob/main/study/torch/1.simpleNN.ipynb)
- [cnn&vgg](https://github.com/rbdus0715/Machine-Learning/blob/main/study/torch/2.cnn_vgg.ipynb)
- [ResNet](https://github.com/rbdus0715/Machine-Learning/blob/main/study/torch/3.resnet.ipynb)
- [RNN](https://github.com/rbdus0715/Machine-Learning/blob/main/study/torch/4.rnn.ipynb)
- [Unet-imageSegmentation](https://github.com/rbdus0715/Machine-Learning/blob/main/study/torch/5.unet_segmentation.ipynb)
- [autoEncoder-1D](https://github.com/rbdus0715/Machine-Learning/blob/main/study/torch/6.auto_encoder_2.ipynb)
- [autoEncoder-2D](https://github.com/rbdus0715/Machine-Learning/blob/main/study/torch/6.auto_encoder.ipynb)
- [auto_coloring](https://github.com/rbdus0715/Machine-Learning/blob/main/study/torch/7.automatic_coloring.ipynb)

# note
*코드*
- Conv2d 이후 이미지 크기 = $\lfloor{(W-K+2P)/S}\rfloor+ 1$ (W=이전 이미지 사이즈, K=커널 사이즈, S=스트라이드, P=패딩)
- Conv2d 이후에도 이미지 크기가 같게하는 조건
  - nn.Conv2d(in_channels, out_channels, **kernel_size=3, padding=1, stride=1**) # stride는 1이 디폴트
  - 증명식 => $(X-3+2)/1+1 = X$
- Conv2d 이후에 이미지 크기가 1/2이 되는 조건
  - nn.Conv2d(in_channels, out_channels, **kernel_size=3, stride=2, padding=1**)
  - 증명식 => $(X-3+2)/2+1=\lfloor{X/2+1/2}\rfloor=X/2$
- upsampling에 사용되는 [Transposed Convolution](https://www.youtube.com/watch?v=U3C8l6w-wn0)
  - nn.ConvTranspose2d()
  - 이후의 이미지 크기 = $S(W-1)+K-2P$
- ConvTranspose2d 이후에도 이미지 크기가 같게하는 조건
  - nn.ConvTranspose2d(in_channels, out_channels, **kernel_size=3, stride=1, padding=1**)
  - 증명식 => $1(X-1)+3-2=X$
- ConvTranspose2d 이후에 이미지의 크기가(한 변의 길이가) 두 배가 되게하는 조건
  - nn.Conv2d(in, out, **kernel_size=2, stride=2**)
  - 증명식 => $2(X-1)+2-0=2X$

*이론*
- LAB: 각각 밝기, 나머지 AB는 컬러 축 **(automatic_coloring에서 사용됨)**
  ```python
  import cv2
  import numpy as np
  from torch.utils.data.dataset import Dataset
  
  def rgb2lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
  
  def lab2rgb(lab):
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
  ```
  - LAB 이미지의 shape은 (height, width, channels)인데, pytorch에서 사용하기 좋은 (C, H, W)로 바꾼다.
  ```python
  lab.transpose((2, 0, 1)).astype(np.float32)
  ```
- 자동채색 모델([let there be color](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf))을 학습할 때 LAB 이미지가 사용되는데, 과정은 다음과 같다.
  1. img(rgb) -> lab 변환
  2. L, AB로 분리
  3. input: L -> output: AB
  4. L과 AB를 concatenation
  5. lab -> rgb 이미지 변환
  6. 출력해서 확인해보기
# error
- *RuntimeError: stack expects each tensor to be equal size, but got [3, 128, 128] at entry 0 and [1, 128, 128] at entry 20*
  - 데이터를 가지고 훈련하기 전, 데이터의 타입, 형식이 무엇인지를 알아야 할 필요가 있다.
  - 잘 정제되지 않은 이미지 데이터의 경우 32bit(4채널), 8bit(1채널) jpg도 있을 수 있어 차원이 안맞는 경우가 발생한다.
  - **32bit 이미지**: 각 픽셀당 32비트(4바이트)의 정보를 사용하여 이미지를 표현한다. 기본 RGB에 알파 투명도 A가 추가되어 RGBA로 불린다.
  - 이미지의 타입이 다름에 의해 생기는 오류를 방지하기 위해 다음 코드를 통해 통일시켜준다.
  ```python
  img = Image.open(images)
  if img.mode == 'RGBA':
    img = img.convert('RGB')
  if img.size(0) == 1:
    img = img.expand(3, -1, -1)
  ```
