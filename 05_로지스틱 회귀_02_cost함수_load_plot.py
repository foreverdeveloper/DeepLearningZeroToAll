# costFunction 함수야말로 이번 코드의 핵심

import math
import numpy as np
import matplotlib.pyplot as plt

# z는 값(scalar)일 수도 있고, vector 또는 matrix일 수도 있다.
def sigmoid(z):
    return 1 / (1 + math.e ** -z)


# costFunction 함수에서 sigmoid를 호출하고 있다.
# linear regression에서 hypothesis에 해당하는 핵심 공식,  cost를 계산하는 기초
# Logistic Regression이 나온 이유가, 이 공식의 결과가 1보다 크거나 0보다 작을 수 있기 때문

#   H(X) = Wx + b

# 이 공식의 결과를 sigmoid에 전달해야 한다는 것이다.
# W와 X는 모두 행렬이기 때문에 행렬 곱셈을 수행한 결과를 전달해야 한다
# sigmoid 호출 결과는 행렬이 들어왔다면, 그와 똑같은 형태의 행렬을 반환한다.
# numpy에서의 행렬 곱셈은 dot 함수가 수행한다
# W는 1행 3열, X는 3행 100열이므로 결과인 h는 1행 100열이 된다
# 여기서는 이 함수를 1회 호출하므로, h에 들어가는 결과는 주석에 있는 것처럼 전체가 0.5이다.


def costFunction(W, X, y):

    m = y.size                  # 100

    # 최초 실행시 값 : [[ 0.5] [ 0.5] [ 0.5] ... [ 0.5]]
    h = sigmoid(np.dot(W, X))   # 1행 m열

    # -(1/m) ∑ylog(H(x)) + (1-y)log(1-H(x))을 구현
    # h는 앞에서 sigmoid에 적용한 결과로 모든 값은 0과 1 사이에 있다는 것을 보장한다
    # y와 log(h)를 곱하고, (1-y)와 log(1-h)를 곱한다.
    # 행렬 곱셈이 아니라 같은 자리끼리 곱하는 element-wise 곱셈이다.
    # 3행 5열끼리의 곱셈처럼, 곱셈에 참여하는 행렬의 크기가
    # 서로 같아야 한다. cost 변수에 들어가는 결과는 비용이라고 부르는 값 1개이다.
    # 값 1개. 곱셈(*)은 element-wise 곱셈
    cost = -(1/m) * sum(y*np.log(h) + (1-y)*np.log(1-h))

    # h와 y는 모두 1차원 배열이고 크기는 똑같이 m개를 갖고 있다. element-wise 뺄셈을 적용한다
    # X와 (h-y)를 행렬 곱셈한다. X는 3행 100열, (h-y)는 1행 100이다.
    # 일반적인 행렬 연산에서는 에러가 발생해야 한다.
    # 뒤에 오는 (h-y)는 100행 1열이어야 어울리기 때문이다
    # np.dot(X, h-y)의 결과는 크기가 m인 1차원 배열이 된다. 여기에 element-wise 곱셈인 1/m을 한다.
    #
    # (h-y)는 1행 m열
    grad = (1/m) * np.dot(X, h-y)

    return cost, grad

# ex2data1.xcsv 파일에는 아래와 같은 줄이 100개 있다. 쉼표로 구분할 수 있는 csv 파일.
# 34.62365962451697,78.0246928153624,0
# 30.28671076822607,43.89499752400101,0
# 35.84740876993872,72.90219802708364,0
# 60.18259938620976,86.30855209546826,1
# 79.0327360507101,75.3443764369103,1

xy = np.loadtxt('05_로지스틱 회귀_02_ex2data1.xcsv', unpack=True, dtype='float32', delimiter=',')

print(xy.shape)     # (3, 100). 행과 열을 바꿔서 읽어온다.
print(xy[:,:5])     # numpy 문법. 리스트는 안됨

# [[ 34.62366104  30.28671074  35.84740829  60.18259811  79.03273773]
#  [ 78.02469635  43.89499664  72.90219879  86.3085556   75.34437561]
#  [  0.           0.           0.           1.           1.        ]]

x_data = xy[:-1]                    # 2행 100열. 정확하게는 2차원 배열
print ('x_data ==>' , x_data)
y_data = xy[-1]                     # 1행 100열. 정확하게는 1차원 배열
print ('y_data ==>' , y_data)


# y_data가 1 또는 0인 값의 인덱스 배열 생성
pos = np.where(y_data==1)
neg = np.where(y_data==0)


# 옥타브와 비슷한 형태로 그래프 출력
# x_data[0,pos]에서 0은 행, pos는 열을 가리킨다. 쉼표 양쪽에 범위 또는 인덱스 배열 지정 가능.
t1 = plt.plot(x_data[0,pos], x_data[1,pos], color='black', marker='+', markersize=7)
t2 = plt.plot(x_data[0,neg], x_data[1,neg], markerfacecolor='yellow', marker='o', markersize=7)

plt.xlabel('exam 1 score')
plt.ylabel('exam 2 score')
plt.legend([t1[0], t2[0]], ['Admitted', 'Not admitted'])        # 범례

plt.show()

# ---------------------------------------------------------------------- #

n, m = x_data.shape         # [2, 100]. 행과 열의 크기
print('n, m :', n, m)

# numpy에서 새로운 행을 맨 앞에 추가하기 위한 코드다.
# 정확하게는 2개의 배열을 연동해서 새로운 배열을 만든다.
# y 절편에 해당하는 b를 행렬의 맨 앞에 추가해야 Wx + b의 코드가 완성된다.
# ones 함수는 원하는 형태의 배열을 만들고 1로 채워준다. 뒤에는 0으로 채워주는 zero 함수도 나온다.
#
# 1로 구성된 배열을 맨 앞에 추가
x_data = np.vstack((np.ones(m), x_data))
print("x_data ==> " , x_data)
print(x_data.shape)         # (3, 100)
print(x_data[:,:5])

# [[  1.           1.           1.           1.           1.        ]
#  [ 34.62366104  30.28671074  35.84740829  60.18259811  79.03273773]
#  [ 78.02469635  43.89499664  72.90219879  86.3085556   75.34437561]]

# W는 초기값으로 1차원 배열로 만들고 모두 0을 넣었다,  0에 대한 sigmoid 값은 0.5다.
# decision boundary와 같은 직선을 그어서 시각적으로 그룹을 나누려면 log로 되어 있는 cost 함수를 미분
# 해야 하는 엄청난 문제에 봉착한다. 그래서, 앤드류 교수님도 이 부분에 대해서는 코드를 공개해서 그냥 확인할
# 수 있도록 했다.
# 미분을 다룰 수 있는 사람이라면 앞의 코드에 살을 붙여서 decision boundary 직선까지 출력할 수 있도록 하
# 면 좋겠다.

# decision boundary 직선. 앤드류 교수님.
# 안에 들어있는 값은 gradient descent 알고리듬을 구현한 이후에 발생한 값
# x값은 x_data에서의 최소값과 최대값. y값은 W값을 이용해서 계산된 값.
# plt.plot([28.059, 101.828], [96.166, 20.653], 'b')
# 이 코드를 plt.show() 함수 앞에 추가하면 decision boundary 직선을 볼 수 있다.
# decision boundary는 직선일 수도 있고 타원일 수도 있고, 상황에 따라 달라진다.

W = np.zeros(n+1)           # [ 0.  0.  0.]. 1행 3열


cost, grad = costFunction(W, x_data, y_data)
print('------------------------------')
print('cost :',  cost)      # cost : 0.69314718056
print('grad :', *grad)      # grad : -0.1 -12.0092164707 -11.2628421021


