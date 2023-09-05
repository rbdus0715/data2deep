# Predict which passengers are transported to an alternate dimension

## 데이터 분석
- [EDA](https://github.com/rbdus0715/Machine-Learning/blob/main/competitions/Spaceship-Titanic/EDA.ipynb)
- 데이터셋 설명 : 타이타닉 우주선의 충돌로 다른 차원으로 갔는지 예측
  - PassengerId
  - HomePlanet : 출발지
  - CryoSleep : 냉동 수면 여부
  - Cabin : 객실 번호, Deck/num/side 형식
  - Destination : 하차하는 곳
  - Age 
  - VIP T/F
  - RoomService, FoodCourt, ShoppingMall, Spa, VRDeck : 타이타닉호의 편의 시설에 대해 청구된 비용
  - Name
  - Transported : 다른 차원으로 갔는지 여부 

## 널값 처리하기
- boolean 널값은 어떻게?
  - 데이터가 충분히 많다면 그 데이터를 무시한다.
  - 주요 값으로 대체한다. 예시) 평균, 최빈값
  - 시계열로 나타나는 데이터라면 앞의 값 혹은 뒤의 값을 참고한다.
  - scikit-learn 라이브러리의 SimpleImputer를 사용하여 대체한다.
- **여기에서 사용한 방법**
  - VIP는 객실 내의 서비스를 많이 이용할 것이므로 RoomService, FoodCourt, ShoppingMall, Spa, VRDeck 를 정렬해서 높은 순서대로 VIP = True로 함
  - 8300개 중의 대략 200개 밖에 없어 더 티도 나지 않고 효과적일 것 같다.

## (꿀팁) 널값 -> 생각을 해보면 되는데...
- 냉동수면과 서비스의 관계
  - 냉동수면을 취한 사람들은 객실 안의 서비스를 사용하지 않을 것이다.
  - 거꾸로 서비스를 0원으로 내지 않은 사람들은 냉동수면을 취한 사람들임을 추측해볼 수 있다. (이것은 확실하지는 않지만 어느정도 일리가 있는 주장이다.)
- 좌석 (Cabin)
  - null값을 찾을 때 꼭 그 칼럼 안에서만 하지 말고, 다른 칼럼에서 흰트를 얻을 수 있나 생각해본다.
  - PassengerId에는 꽤 다양한 정보가 들어있다.
    - PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
  - 이를 통해서 좌석, VIP 등의 null 값을 실제와 거의 비슷하게 대체할 수 있다.
- 나이
  - 다른 피처들과의 상관관계를 이용
