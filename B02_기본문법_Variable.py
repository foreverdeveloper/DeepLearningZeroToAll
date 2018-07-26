'''



'''
import tensorflow as tf

var1 = tf.Variable([5])
var2 = tf.Variable([10])
var3 = tf.Variable([3])

var4 = var1 * var2 + var3

print('출력 1 ==>', var4)
'''
출력
-------------------------------
Tensor("add:0", shape=(1,), dtype=int32)


Variable은 텐서가 아니라 하나의 객체가 되는 것이다. 
즉 Variable 클래스의 인스턴스가 생성되는 것이고, 해당 인스턴스를 그래프에 추가시켜주어야 한다.

실제 global_variables_initializer()를 사용하여야 한다. 이 자체가 연산이 된다.
global_variables_initializer()를 호출하기 전에 그래프의 상태는 각 노드에 값이 아직 없는 상태를 의미한다.
따라서 해당 함수를 사용해주어야 Variable 의 값이 할당 되는 것이고, 텐서의 그래프로써의 효력이 발생하는 것이다.
(이전 버전에서는, .initialize_all_variables() 였다 )
-------------------------------//
'''
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('출력 2 ==>', var4)

    result = sess.run(var4)
    print('출력 3 ==>', result)

