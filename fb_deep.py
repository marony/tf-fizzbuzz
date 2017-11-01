import tensorflow as tf
import numpy as np

"""
TensorFlow$B$N>pJs$r%0%0$j$^$/$C$F:n$C$?5!3#3X=,(BFizzBuzz
$B3FAX$NLr3d$H$+?tCM$,E,Ev$+$H$+$h$/$o$+$C$F$J$$(B
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
L = 0.1
# $B1#$lAX(B
H1 = 64
H2 = 64

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
    return [1, 0, 0, 0]

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

def model(x, weights, biases):
    """
    $B%b%G%k$N:n@.(B
    $BF~NO(B(N)$B"*1#$l(B1(N, H1)$B"*1#$l(B2(H1, H2)$B"*=PNO(B(H2, C)
    """
    with tf.name_scope('hidden1'):
        layer_1 = tf.matmul(x, weights['h1']) + biases['b1']
        layer_1 = tf.nn.relu(layer_1)
    with tf.name_scope('hidden2'):
        layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']
        layer_2 = tf.nn.relu(layer_2)
    with tf.name_scope('output'):
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

def loss(logits, labels):
    """
    $B3X=,$N$J$K$+(B($B$h$/$o$+$C$F$J$$(B)
    """
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss

weights = {
    'h1': tf.Variable(tf.truncated_normal([N, H1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([H1, H2], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([H2, C], stddev=0.1))
}
biases = {
    'b1': tf.Variable(tf.zeros([H1])),
    'b2': tf.Variable(tf.zeros([H2])),
    'out': tf.Variable(tf.zeros([C]))
}

# $B%b%G%k:n@.(B
x = tf.placeholder(tf.float32, [None, N])
y = tf.placeholder(tf.float32, [None, C])

# Construct model
pred = model(x, weights, biases)

# $B3X=,$N$J$K$+(B($B$h$/$o$+$C$F$J$$(B
loss = loss(pred, y)
train_step = tf.train.GradientDescentOptimizer(L).minimize(loss)
#optimizer = tf.train.AdamOptimizer(learning_rate=L).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=L).minimize(cost)

init = tf.global_variables_initializer()

# $B%;%C%7%g%s3+;O(B
with tf.Session() as sess:
    sess.run(init)

    # $B71N}(B
    x_data, y_data = create_training_data()
    for _ in range(T):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # $B71N}7k2L=PNO(B
    for i in range(30):
        n = i + 1
        xx = [n_to_a(n)]
        yy = sess.run(pred, feed_dict={x: xx})
#        print(n, one_hot_to_fb(n, yy[0]), np.argmax(yy[0]), yy)
#        print(one_hot_to_fb(n, yy[0]))
        print(one_hot_to_fb(n, yy[0]), fizz_buzz(n))

