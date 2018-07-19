# https://www.tensorflow.org/install/#pip-installation
#     노드에 연산(계산, operation)을 담고, 에찌(간선)에 데이터를 담고 있다
#     데이터가 있고, 데이터를 연산에 전달하면 수정된 데이터가 나오고.
#     연산에는 입력과 출력이 있는데, 입력 데이터는 결국 에찌가 되고, 출력 데이터 또한 에찌가 될 수 있겠다
import tensorflow as tf

def hello():
    # 변수 a는 문자열 상수를 저장하고 있다. 그렇다면, 앞에서 설명한 바에 따르면 내부적으로 에찌에 저장된다는 말?
    # 머신러닝에서는 현재 데이터가 무엇인지 판단할 수 없는 상황이 매우 많다. 구동시켜 보기 전에는, 즉 run
    # 함수를 호출하기 전에는 값을 알 수 없기 때문에 일관되게 처리하기 위해서는 모든 텐서 객체에 대해 자신이
    # 누구인지만 알려주는 요약본을 출력하는 것이 맞다.
    a = tf.constant('hello, tensorflow!')

    # print(a)에서 데이터가 직접 출력되지 않는 것은 당연
    print(a)  # Tensor("Const:0", shape=(), dtype=string)

    # 구동하기 위해서는 세션이 필요하다
    sess = tf.Session()
    # 텐서플로우 구동은 세션에 포함된 run 함수를 호출하면 된다.
    result = sess.run(a)

    # 2.x 버전에서는 문자열로 출력되지만, 3.x 버전에서는 byte 자료형
    # 문자열로 변환하기 위해 decode 함수로 변환
    print(result)  # b'hello, tensorflow!'
    # run 호출의 결과를 type  함수로 확인했다
    print(type(result))  # <class 'bytes'>
    # 디코딩을 하기 전에는 bytes라고 하는 일종의 바이트 배열이다
    # 어떤 데이터인지 정확하게 알고 있기 때문에 utf-8 인코딩을 적용했고,
    # 파이썬 문자열 타입인 str을 얻을 수 있었다.
    print(result.decode(encoding='utf-8'))  # hello, tensorflow!
    print(type(result.decode(encoding='utf-8')))  # <class 'str'>

    # 세션 닫기, 코드가 종료되기 때문에 닫지 않아도 괜찮긴 하겠지만, 이런 것은 습관의 영역으로 보인다.
    sess.close()


hello()


def constant():
    a = tf.constant(2)
    b = tf.constant(3)

    # with 구문을 사용하면 리소스를 정리하는 close 비슷한 함수를 호출하지 않아도 됨
    # with 구문에서 알아서 호출해 준다. sess.close()를 호출하지 않았다
    # 파일 열기와 닫기 같은, 쌍을 이루는 코드에 주로 사용한다.
    # as 연산자는 별칭을 주는 기능이고, with 구문에서는 변수를 만들 수 없기 때문에 as를 사용해서
    # 이름을 주어야 한다.

    # 정수를 덧셈한 결과를 저장한 자료형은 numpy 모듈에 있는 int32 자료형이다.
    # 파이썬에서 기본적으로 사용하고 있는 int 자료형이 아닌 것이 중요하다.

    # 파이썬은 인터프리터 방식의 엄청나게 느린 언어이기 때문에 굳이 성능을 올리겠다고 생각한다면,
    # 파이썬 코드를 최소한으로 유지하고 모듈에 포함된 기능을 사용하면 된다.
    #
    # numpy는 C 언어에 있는 배열과 같은 형태로 움직이는 다차원 배열을 기반으로 하는 모듈이다.
    # 빅데이터, 머신러닝, 과학산술 등의 수치연산이 필요한 모든 경우에 최적의 성능을 보장해 준다.
    # 그래서, 텐서플로우는 내부적으로 numpy를 사용할 수밖에 없다.
    # 여기서는 numpy 코드가 나오면 함께 설명을 할 생각이다.
    # 참, numpy에 대한 발음은 '넘피'와 '넘파이' 둘 중의 하나를 사용하면 되는데,
    # 외국 동영상 등에 자주 등장하는 발음은 '넘파이'. 처음에는 '넘피'로 발음하다가
    # 지금은 '넘파이'를 쓰고 있다.
    #
    # with 구문을 벗어날 때, 종료 코드가 있다면 대신 호출해 줌
    # 예외가 발생한 경우에도 보장
    with tf.Session() as sess:
        result = sess.run(a+b)
        print(type(result))             # <class 'numpy.int32'>
        print(result)                   # 5

        # 마지막 줄의 코드에서 7을 더하면 자료형이 int64로 바뀐다.
        # 이것은 numpy의 고유한 기능이다.
        # 32비트 숫자 2개를 더하면, 오버플로우(overflow)라는 데이터 넘침 현상이 일어날 수 있기 때문에
        # 수치연산을 많이 하는 numpy에서는 이러한 오버플로우를 막기 위한 당연한 조치.
        # int 자료형과 연산 가능
        print(result + 7)               # 12
        print(type(result + 7))         # <class 'numpy.int64'>

constant()


# placeholder는 자리만 차지하고 있는 물건이나 사람을 뜻하는 영어 단어다.
# 그런데, 텐서플로우에 와서 엄청나게 중요한 역할이 주어졌다.
# 머신러닝에 전달되는 데이터를 변경하기 위한 수단이 되었다.
# 머신에게 공부를 시킨 이유는 내가 궁금한 무엇을 물어보기 위해서다.
# 그렇다면 궁금한 것을 전달해야 하고, 전달할 수 있는 문법이 있어야 하는데, 그것이 placehoder이다.
# placeholder를 만들 때는 우리가 궁금해 하는 데이터의 자료형에 대해 알려줘야 한다.
# 여기서는 매우 작은 정수를 다루기 때문에 tf.int16이라고 지정했고,
# 출력 결과에서는 numpy.int16이라고 표시됐다.
# 사용자에게, 가능하면 numpy라고 하는 생소한 이름을 언급하지 않으려는 배려라고 보여지는 부분이다.
#

def placeHolder():
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)

    # add()와 mul()이라는 덧셈과 곱셈 연산(노드)을 만들었고,
    # 어떤 데이터를 전달할지는 나중에 결정할 수 있도록, placeholder로 처리했다.
    # with 블록 안에서 add에 대해 결과를 요청하면서 2, 3을 파라미터로 전달했다. 지금
    # 은 별거 아닌 것처럼 보이지만, 그래프 기반이라서 add()와 mul() 등을 수십 개 연결할 수 있다고
    # 생각해 보면
    # 엄청나게 복잡한 연산을 매우 쉽게 처리할 수 있는 효과적인 방법이라는 것을 알 수 있다.
    # 딕셔너리(사전) 자료
    # 형을 사용하기 때문에 파라미터의 갯수에는 제한이 없다. 100개를 전달해도 괜찮고 실전에서는 이런 일이 빈번
    # 하게 일어난다고 들었다. 배우는 중이라서 직접 넣어본 적이 없다.
    #
    add = tf.add(a, b)
    mul = tf.multiply(a, b)

    with tf.Session() as sess:
        # {a: 2, b: 3}는 딕셔너리
        # key로 'a'와 'b'를 사용하고, value로 2와 3  사용
        # free_dict를 사용하지 않을 경우 None 기본값 적용
        r1 = sess.run(add, feed_dict={a: 2, b: 3})
        r2 = sess.run(mul, feed_dict={a: 2, b: 3})

        print(type(r1))                 # <class 'numpy.int16'>
        print(r1, r2)                   # 5, 6

placeHolder()


# 정말 세션을 만들고, run 함수를 정상적으로 구동해야 하는지 궁금해서 구글링을 열심히 했다.
# 어쨌건 세션은 반드시 필요하긴 한데,
# 미리 만들어 놓은 세션에 연결하기 위한 InteractiveSession 함수를 찾을 수 있었다.
# 세션을 만들 수 있는 방법이 두 가지 있는데, 대부분 Session 클래스를 사용한다.
# InteractiveSession은 주피터 등에서 코드와 설명을 함께 구성할 때만 사용한다.

def showTensor():
    sess = tf.InteractiveSession()

    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])

    # x에 대해서 연산을 수행해서 결과를 먼저 만든다.
    x.initializer.run()     # Initialize 'x' using the run() method of its initializer op.

    sub = tf.subtract(x, a)      # Add an op to subtract 'a' from 'x'.  Run it and print the result
    print(sub.eval())       # [-2. -1.]

    print('-------------------------------------')

    # 결과를 내장하고 있다면 eval() 사용 가능. initializer 없이 x에 대해서 호출하면 비정상 종료
    print(a.eval())         # [ 3.  3.]
    print(x.eval())         # [ 1.  2.]

    # -1에서 1 사이의 정규분포 난수 3개 생성. b는 1행 3열의 텐서 객체
    b = tf.random_uniform([3], -1.0, 1.0)
    print(type(b))          # <class 'tensorflow.python.framework.ops.Tensor'>
    print(b.eval())         # [-0.16271138 -0.33350062  0.51194   ]

    # tensor라면 initializer 사용
    w = tf.Variable(tf.random_uniform([5, 3], 0, 32, dtype=tf.int32))
    w.initializer.run()
    print(w.eval())         # [[15  1 21] [14 16 27] [13 30 28] [23 21 26] [15 19 16]]

    print('-------------------------------------')

    x = [[1., 1.], [10., 2.]]
    print(tf.reduce_mean(x).eval())         # 3.5, 전체 평균
    print(tf.reduce_mean(x, 0).eval())      # [ 5.5  1.5], 0은 column
    print(tf.reduce_mean(x, 1).eval())      # [ 1.  6.], 1은 row

    sess.close()


showTensor()
