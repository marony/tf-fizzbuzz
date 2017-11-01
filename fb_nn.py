import tensorflow as tf
import numpy as np

"""
TensorFlowの「MNIST For ML Beginners 」の通りの機械学習FizzBuzz
正常にFizzBuzzできない
直線でクラスわけしてるのが理由だと思うけど間違ってるかも
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
    return [0.3, 0, 0, 0]

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

# モデル作成
x = tf.placeholder(tf.float32, [None, N])
y_ = tf.placeholder(tf.float32, [None, C])

W = tf.Variable(tf.zeros([N, C]))
b = tf.Variable(tf.zeros([C]))

y = tf.matmul(x, W) + b

# 学習のなにか(よくわかってない)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# セッション開始
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # 訓練
    x_data, y_data = create_training_data()
    for _ in range(T):
        train_step.run(feed_dict={x: x_data, y_: y_data})

    # 訓練結果出力
    for i in range(30):
        n = i + 1
        xx = [n_to_a(n)]
        yy = sess.run(y, feed_dict={x: xx})
#        print(n, one_hot_to_fb(n, yy[0]), yy)
#        print(one_hot_to_fb(n, yy[0]))
        print(one_hot_to_fb(n, yy[0]), fizz_buzz(n))

