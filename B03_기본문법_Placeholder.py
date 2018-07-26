'''

자료형이라고 하는게 맞는지는 모르겠지만.. 어쨌든.. placeholder 자료형은 조금 특이하다.
선언과 동시에 초기화 하는 것이 아니라. 일단, 선언 후 그 다음 값을 전달한다.
따라서 반드시 실행 시 데이터가 제공되어야 한다.

여기서 값을 전달한다고 되어 있는데, 이는 데이터를 상수값을 전달함과 같이 할당하는 것이 아니라
★ 다른 텐서(Tensor)를 placeholder에 맵핑 시키는 것이라고 보면 된다. ★

placeholder의 전달 파라미터는 다음과 같다.

placeholder(
    dtype,
    shape=None,
    name=None
)
    dtype   : ★ 데이터 타입을 의미하며 반드시 적어주어야 한다. ★
    shape   : 입력 데이터의 형태를 의미한다. 상수 값이 될 수도 있고, 다차원 배열의 정보가 들어올 수도 있다.
            ( 디폴트 파라미터로 None 지정 )
    name    : 해당 placeholder의 이름을 부여하는 것으로 적지 않아도 된다.
            ( 디폴트 파라미터로 None 지정 )



위에서도 말했었지만, placeholder는 다른 텐서를 할당하는 것이라고 했다.
이를 할당하기 위해서는 feed dictionary 라는 것을 활용하게 되는데 세션을 생성할때
feed_dict의 키워드 형태로 텐서를 맵핑 할 수 있다.

위와 같이 선언 후 feed_dict 변수를 할당해도 되고 바로 값을 대입시켜도 무방하다.

'''
import tensorflow as tf

p_holder1 = tf.placeholder(dtype=tf.float32)
p_holder2 = tf.placeholder(dtype=tf.float32)
p_holder3 = tf.placeholder(dtype=tf.float32)

val1 = 5
val2 = 10
val3 = 3

ret_val = p_holder1 * p_holder2 + p_holder3

feed_dict = {p_holder1: val1, p_holder2: val2, p_holder3: val3}
with tf.Session() as sess:
    result = sess.run(ret_val, feed_dict=feed_dict)

    print(result)


'''
placeholder는 여러 형태로 사용할 수 있다. 
머신러닝에서는 다차원 배열 데이터가 많이 들어오게 되는데 
어쨌든 배열의 형태가 값으로 들어가도 무방하다는 것이다.

허접하지만 이와 같이 Image Matrix(영상)의 정보와 각 라벨이 들어가 있다고 가정해보자.
크게 의미가 있는 코드는 아니지만 이와 같이 연산을 수행할 수 있다.
보통 데이터셋(DataSet) 정도의 거대 데이터 정보를 feed로 많이 주는 것으로 활용된다.

'''
mat_img = [1, 2, 3, 4, 5]
label = [10, 20, 30, 40, 50]

ph_img = tf.placeholder(dtype=tf.float32)
ph_lb = tf.placeholder(dtype=tf.float32)

ret_tensor = ph_img + ph_lb

result = sess.run(ret_tensor, feed_dict={ph_img: mat_img, ph_lb: label})
print(result)

