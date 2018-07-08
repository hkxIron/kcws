# 参见blog:
# https://blog.csdn.net/pirage/article/details/53424544
# https://mp.weixin.qq.com/s?__biz=MjM5ODIzNDQ3Mw==&mid=2649966433&idx=1&sn=be6c0e5485003d6f33804261df7c3ecf&chksm=beca376789bdbe71ef28c509776132d96e7e662be0adf0460cfd9963ad782b32d2d5787ff499&mpshare=1&scene=2&srcid=1122cZnCbEKZCCzf9LOSAyZ6&from=timeline&key=&ascene=2&uin=&devicetype=android-19&version=26031f30&nettype=WIFI
# paper: # paper: http://www.aclweb.org/anthology/N16-1030
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_data_path', "/Users/tech/code/kcws/train.txt", 'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', "./test.txt", 'Test data dir')
tf.app.flags.DEFINE_string('log_dir', "logs", 'The log  dir')
tf.app.flags.DEFINE_string('embedding_result', "embedding.txt", 'The log  dir')
tf.app.flags.DEFINE_integer("max_sentence_len", 5,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 25, "embedding size")
tf.app.flags.DEFINE_integer("num_hidden", 20, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 100, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 50000, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("num_words", 5902, "embedding size")


class Model:
    def __init__(self, embeddingSize, distinctTagNum, c2vPath, numHidden):
        self.embeddingSize = embeddingSize
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.c2v = self.load_w2v(c2vPath)
        self.words = tf.Variable(self.c2v, name="words")
        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable( # [2*hidden_size, tagNum]
                shape=[numHidden * 2, distinctTagNum],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum], name="bias"))
        self.trains_params = None
        self.inp = tf.placeholder(tf.int32,
                                  shape=[None, FLAGS.max_sentence_len],
                                  name="input_placeholder")
        pass

    def length(self, data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, X, reuse=None, trainMode=True):
        word_vectors = tf.nn.embedding_lookup(self.words, X)  # 按照X顺序返回self.words中的第X行，返回的结果组成tensor。
        length = self.length(word_vectors)
        # length是shape为[batch_size]大小的句子长度的vector
        length_64 = tf.cast(length, tf.int64)
        if trainMode:  # 训练的时候启用dropout，测试的时候关键dropout
            word_vectors = tf.nn.dropout(word_vectors, 0.5)  # 将word_vectors按照50%的概率丢弃某些词，tf增加的一个处理是将其余的词scale 1/0.5
        with tf.variable_scope("rnn_fwbw", reuse=reuse) as scope:
            # word_vectors:[batch_size, max_time,embeddin_size]
            # forward_output: [batch_size, max_time,num_hidden]
            # backward_output: [batch_size, max_time,num_hidden]
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.numHidden),
                word_vectors,
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.numHidden),
                inputs=tf.reverse_sequence(word_vectors,
                                           length_64,
                                           seq_dim=1),
                # seq_dim =1
                # 训练和测试的时候，inputs的格式不同。训练时，tensor shape是[batch_size, max_time,input_size]
                # 测试时，tensor shape是[max_time,batch_size,input_size].
                # tf.reverse_sequence作用就是指定在列上操作（batch_dim表示按行操作）
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backword")
        # tf.nn.dynamic_rnn(cell, inputs, sequence_length,time_major,...)主要参数：
        # cell：搭建好的网络，这里用LSTMCell(num_cell)，num_cell表示一个lstm单元输出的维数（100）
        # inputs：word_vectors，它的shape由time_major决定，默认是false，即[batch_size，max_time，input_size]，如果是测试
        #           过程，time_major设置为True，shape为[max_time，batch_size，input_size]，这里直接做了reverse，省去了time_major设置。
        #         其中，batch_size=100, max_time=80句子最大长度，input_size为字的向量的长度。
        # sequence_length：shape[batch_size]大小的值为句子最大长度的tensor。
        # 输出：
        #   outputs：[batch_size, max_time, hidden_size]
        #   state: shape取决于LSTMCell中state_size的设置，返回Tensor或者tuple。

        backward_output = tf.reverse_sequence(backward_output_,
                                              length_64,
                                              seq_dim=1)
        # 这里的reverse_sequence同上。
        # forward_output：[batch_size, max_time, hidden_size]
        # backward_output：[batch_size, max_time, hidden_size]
        # output: [batch_size, max_time, hidden_size*2]
        output = tf.concat([forward_output, backward_output], axis = 2)
        # 连接两个三维tensor，2表示按照列连接（0表示纵向，1表示行）
        # 连接后，output的shape:[batch_size, max_time, 2*hidden_size],即[100， 80， 2*50]
        output = tf.reshape(output, [-1, self.numHidden * 2])
        # reshape后，output的shape:[batch_size*max_time, hidden_size * 2],即[100, 200]
        # W : [2*hidden_size, tagNum]
        matricized_unary_scores = tf.batch_matmul(output, self.W)
        # 得到未归一化的CRF输出
        # 点乘W的shape[ 100*2, 4],生成[batch_size, 4]大小的matricized_unary_scores
        # unary_score: [batch, max_time, tag_num]
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, FLAGS.max_sentence_len, self.distinctTagNum])
        # reshape后，unary_scores大小为[batch_size，80， 4]
        return unary_scores, length

    def loss(self, X, Y):
        # P : [batch, max_time, tag_num]
        P, sequence_length = self.inference(X)
        # CRF损失计算，训练的时候使用，测试的时候用viterbi解码
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, Y, sequence_length)
        # crf_log_likelihood参数(inputs,tag_indices, sequence_lengths)
        #   inputs:大小为[100, 80, 4]的tensor，CRF层的输入
        #   tag_indices:大小为[100, 80]的矩阵
        #   sequence_length：大小 [100]值为80的向量。
        # 输出：
        #   log_likelihood：[batch_size]大小的vector，log-likelihood值
        #   transition_params：[4,4]大小的矩阵
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def load_w2v(self, path):  # 返回（num+2）*50大小的二维矩阵，其中第一行全是0，最后一行是每个词向量维度的平均值。
        fp = open(path, "r")
        print("load data from:", path)
        line = fp.readline().strip()
        ss = line.split(" ")
        total = int(ss[0])
        dim = int(ss[1])
        assert (dim == (FLAGS.embedding_size))
        ws = []
        mv = [0 for i in range(dim)]
        # The first for 0
        ws.append([0 for i in range(dim)])
        for t in range(total):
            line = fp.readline().strip()
            ss = line.split(" ")
            assert (len(ss) == (dim + 1))
            vals = []
            for i in range(1, dim + 1):
                fv = float(ss[i])
                mv[i - 1] += fv
                vals.append(fv)
            ws.append(vals)
        for i in range(dim):
            mv[i] = mv[i] / total
        ws.append(mv)
        fp.close()
        return np.asarray(ws, dtype=np.float32)

    def test_unary_score(self):
        P, sequence_length = self.inference(self.inp,
                                            reuse=True,
                                            trainMode=False)
        return P, sequence_length


def main(unused_argv):
    curdir = os.path.dirname(os.path.realpath(__file__))
    trainDataPath = tf.app.flags.FLAGS.train_data_path
    if not trainDataPath.startswith("/"):
        trainDataPath = curdir + "/" + trainDataPath
    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.embedding_size, FLAGS.num_tags,
                  FLAGS.word2vec_path, FLAGS.num_hidden)
        print("train data path:", trainDataPath)
        # 读取训练集batch大小的feature和label，各为80大小的数组
        X, Y = inputs(trainDataPath)
        # 读取测试集所有数据的feature和label，各为80大小的数组
        tX, tY = do_load_data(tf.app.flags.FLAGS.test_data_path)
        # 计算训练集的损失
        total_loss = model.loss(X, Y)
        # 使用AdamOptimizer优化方法
        train_op = train(total_loss)
        # 在测试集上做评测
        test_unary_score, test_sequence_length = model.test_unary_score()
        # 创建Supervisor管理模型的分布式训练
        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    _, trainsMatrix = sess.run(
                    [train_op, model.transition_params])
                    # for debugging and learning purposes, see how the loss gets decremented thru training steps
                    if step % 100 == 0:
                        print("[%d] loss: [%r]" % (step, sess.run(total_loss)))
                    if step % 1000 == 0:
                        test_evaluate(sess, test_unary_score,
                                  test_sequence_length, trainsMatrix,
                                  model.inp, tX, tY)
                except KeyboardInterrupt, e:
                    sv.saver.save(sess,
                              FLAGS.log_dir + '/model',
                              global_step=step + 1)
                    raise e
            sv.saver.save(sess, FLAGS.log_dir + '/finnal-model')
            sess.close()
