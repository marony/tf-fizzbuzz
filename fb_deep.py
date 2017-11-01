import tensorflow as tf
import numpy as np

"""
TensorFlowの情報をググりまくって作った機械学習FizzBuzz
各層の役割とか数値が適当かとかよくわかってない
"""

# 初期値
# 整数MAX値
M = 100
# 入力ベクトルの大きさ(8bit整数)
N = 8
# クラス数(None, Fizz, Buzz, FizzBuzz)
C = 4
# 訓練回数
T = 10000
L = 0.1
# 隠れ層
H1 = 64
H2 = 64

def fizz_buzz(n):
    """
    整数に応じてFizzBuzz文字列を出力する
    FizzBuzzの正解作成に使用
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
    FizzBuzz文字列からそのクラスを表すベクトルを作成(one_hotという？？？)
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
    FizzBuzzのクラスを表すベクトルからFizzBuzz文字列に変換
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
    整数を2進数を表す8要素ベクトルに変換
    """
    a = np.zeros(N)
    for i in range(N):
        if n & 1 != 0:
            a[N - i - 1] = 1
        n >>= 1
    return a

def create_training_data():
    """
    訓練用(正解)データの作成                                                                                         
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
    モデルの作成
    入力(N)→隠れ1(N, H1)→隠れ2(H1, H2)→出力(H2, C)
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
    学習のなにか(よくわかってない)
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

# モデル作成
x = tf.placeholder(tf.float32, [None, N])
y = tf.placeholder(tf.float32, [None, C])

# Construct model
pred = model(x, weights, biases)

# 学習のなにか(よくわかってない
loss = loss(pred, y)
train_step = tf.train.GradientDescentOptimizer(L).minimize(loss)
#optimizer = tf.train.AdamOptimizer(learning_rate=L).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=L).minimize(cost)

init = tf.global_variables_initializer()

# セッション開始
with tf.Session() as sess:
    sess.run(init)

    # 訓練
    x_data, y_data = create_training_data()
    for _ in range(T):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 訓練結果出力
    for i in range(30):
        n = i + 1
        xx = [n_to_a(n)]
        yy = sess.run(pred, feed_dict={x: xx})
#        print(n, one_hot_to_fb(n, yy[0]), np.argmax(yy[0]), yy)
#        print(one_hot_to_fb(n, yy[0]))
        print(one_hot_to_fb(n, yy[0]), fizz_buzz(n))

