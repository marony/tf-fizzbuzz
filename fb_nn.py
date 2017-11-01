import tensorflow as tf
import numpy as np

"""
TensorFlow$B$N!V(BMNIST For ML Beginners $B!W$NDL$j$N5!3#3X=,(BFizzBuzz
$B@5>o$K(BFizzBuzz$B$G$-$J$$(B
$BD>@~$G%/%i%9$o$1$7$F$k$N$,M}M3$@$H;W$&$1$I4V0c$C$F$k$+$b(B
"""

# $B=i4|CM(B
# $B@0?t(BMAX$BCM(B
M = 100
# $BF~NO%Y%/%H%k$NBg$-$5(B(8bit$B@0?t(B)
N = 8
# $B%/%i%9?t(B(None, Fizz, Buzz, FizzBuzz)
C = 4
# $B71N}2s?t(B
T = 10000

def fizz_buzz(n):
    """
    $B@0?t$K1~$8$F(BFizzBuzz$BJ8;zNs$r=PNO$9$k(B
    FizzBuzz$B$N@52r:n@.$K;HMQ(B
    """
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    return n

def fb_to_one_hot(fb):
    """
    FizzBuzz$BJ8;zNs$+$i$=$N%/%i%9$rI=$9%Y%/%H%k$r:n@.(B(one_hot$B$H$$$&!)!)!)(B)
    """
    if fb == "Fizz":
        return [0, 1, 0, 0]
    elif fb == "Buzz":
        return [0, 0, 1, 0]
    elif fb == "FizzBuzz":
        return [0, 0, 0, 1]
    return [0.3, 0, 0, 0]

def one_hot_to_fb(n, a):
    """
    FizzBuzz$B$N%/%i%9$rI=$9%Y%/%H%k$+$i(BFizzBuzz$BJ8;zNs$KJQ49(B
    """
    i = np.argmax(a)
    if i == 1:
        return "Fizz"
    elif i == 2:
        return "Buzz"
    elif i == 3:
        return "FizzBuzz"
    return n

def n_to_a(n):
    """
    $B@0?t$r(B2$B?J?t$rI=$9(B8$BMWAG%Y%/%H%k$KJQ49(B
    """
    a = np.zeros(N)
    for i in range(N):
        if n & 1 != 0:
            a[N - i - 1] = 1
        n >>= 1
    return a

def create_training_data():
    """
    $B71N}MQ(B($B@52r(B)$B%G!<%?$N:n@.(B
    """
    x_data = []
    y_data = []
    for i in range(M):
        n = i + 1
        a = n_to_a(n)
        x_data.append(a)
        fb = fizz_buzz(n)
        y_data.append(fb_to_one_hot(fb))
    return (x_data, y_data)

# $B%b%G%k:n@.(B
x = tf.placeholder(tf.float32, [None, N])
y_ = tf.placeholder(tf.float32, [None, C])

W = tf.Variable(tf.zeros([N, C]))
b = tf.Variable(tf.zeros([C]))

y = tf.matmul(x, W) + b

# $B3X=,$N$J$K$+(B($B$h$/$o$+$C$F$J$$(B)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# $B%;%C%7%g%s3+;O(B
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # $B71N}(B
    x_data, y_data = create_training_data()
    for _ in range(T):
        train_step.run(feed_dict={x: x_data, y_: y_data})

    # $B71N}7k2L=PNO(B
    for i in range(30):
        n = i + 1
        xx = [n_to_a(n)]
        yy = sess.run(y, feed_dict={x: xx})
#        print(n, one_hot_to_fb(n, yy[0]), yy)
#        print(one_hot_to_fb(n, yy[0]))
        print(one_hot_to_fb(n, yy[0]), fizz_buzz(n))

