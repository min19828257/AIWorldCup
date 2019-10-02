# Author(s): Luiz Felipe Vecchietti, Chansol Hong, Inbae Jeong
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import math

# convert degree to radian 디그리를 라디안으로
# 각의 단위 '1도, 322도' = 디그리 
# 1라디안 = 57.2958도 = 57.296도
def d2r(deg):
    return deg * math.pi / 180

# convert radian to degree
def r2d(rad):
    return rad * 180 / math.pi

# measure the distance between two coordinates (x1, y1) and (x2, y2)
def dist(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

#두 점 사이의 거리는 피타고라스 정리를 이용하여 구할 수 있다. 
#점은 X 자체만 존채할 수 없고, Y자체로도 존채할 수 없다. 값이 '0'이라도 X와 Y는 존재하는 것이기 때문에,
#그 값들이 모두 0 이든, 하나만 0이든 존재하는 것은 명확하기 때문이다. 즉, 점은 X와 Y가 같이 있을 때 정의된다.

#pow 입력받은 수를 거듭제곱하여 리턴하는 함수
#sqrt(square root = 제곱근) 는 숫자의 제곱근, 즉 루트의 근사값을 구하기 위함. 
#실수 9의 제곱근은 +-3 

# convert radian in (-inf, inf) range to (-PI, PI) range
def trim_radian(rad):
    adj_rad = rad
    while(adj_rad > math.pi):
        adj_rad -= 2*math.pi
    while(adj_rad < -math.pi):
        adj_rad += 2*math.pi
    return adj_rad
