import math
import numpy as np

# 어떤 숫자가 들어와도 0과 1 사이의 값으로 변환해주는 함수
# 여기서는 100, 0, -10의 세 가지를 사용했다. 0을 넣었을 때,
# 가운데 값이므로 정확하게 0.5가 나와야 하는 것도 중요하다.
# z는 값(scalar)일 수도 있고, vector 또는 matrix일 수도 있다.
# math 모듈에 자연상수 e가 들어있는 것을 찾았다. 지수 연산자인 **를 사용해서 아주 쉽게 분모를 구성
# 파이썬3에서는 //는 정수 나눗셈, /는 실수 나눗셈
# sigmoid는 나중에 행렬까지 처리할 수 있어야 하기 때문에 numpy의 배열을 사용해서 검증
# 리스트는 행렬 연산을 지원하지 않는다.

def sigmoid(z):
    return 1 / (1 + math.e ** -z)

print(sigmoid(100))
print(sigmoid(  0))
print(sigmoid(-10))
print(sigmoid(np.array([100, 0, -10])))
