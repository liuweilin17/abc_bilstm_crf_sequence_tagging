#coding=utf-8
import tensorflow as tf
import numpy

class PropertiesClass(object):
    def __init__(self, FLAGS):
        self.input_x = tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32,[None, FLAGS.max_length], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        input_properties = [self.input_y]
        #embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            emb_W = tf.Variable(tf.random_uniform([FLAGS.vocab_num, FLAGS.embedding_size], 0., 1.), name='emb_W')
            embedding_chars = tf.nn.embedding_lookup(emb_W, self.input_x)
            input_item = embedding_chars

        self.mask_x = tf.sign(self.input_x)
        self.sequence_length = tf.reduce_sum(self.mask_x, axis=1)   #the length of each row in input_x

        rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size)
        rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(rnn_cell_fw, output_keep_prob=self.keep_prob)
        rnn_cell_bw= tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size)
        rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(rnn_cell_bw, output_keep_prob=self.keep_prob)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size * 2)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)

        self.properties_loss = []
        self.properties_accuracy = []
        self.properties_pred =[]
        self.properties_transition = []

        for i, property in enumerate(input_properties):
            with tf.variable_scope('properties_lstm') as scope:
                if i > 0:
                    scope.reuse_variables()
                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell_fw, cell_bw=rnn_cell_bw,
                                                                  dtype=tf.float32, sequence_length=self.sequence_length,
                                                                  inputs=input_item, time_major=False)
                outputs = tf.concat(outputs, 2)
                lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, inputs=outputs, dtype=tf.float32, sequence_length=self.sequence_length, time_major=False)

            with tf.name_scope('properties_softmax'):
                pro_weights = tf.Variable(tf.truncated_normal([FLAGS.hidden_size * 2, FLAGS.classes], stddev=0.5),
                                          name='pro_weights')
                pro_biases = tf.Variable(tf.truncated_normal([FLAGS.classes], stddev=0.5), name='pro_biases')
                output = tf.reshape(lstm_output, [-1, FLAGS.hidden_size * 2])
                self.property_logits = tf.nn.xw_plus_b(output, pro_weights, pro_biases)
#                property_probs = tf.reshape(tf.nn.softmax(self.property_logits), [-1, FLAGS.max_length, FLAGS.classes])
                property_probs = tf.reshape(self.property_logits, [-1, FLAGS.max_length, FLAGS.classes])
#               print property_probs
#                property_pred = tf.to_int32(tf.argmax(property_probs, axis=2))
                self.properties_pred.append(property_probs)

            with tf.variable_scope('properties_calculate{}'.format(i)):
                property_losses, transition_params = tf.contrib.crf.crf_log_likelihood(property_probs, property, self.sequence_length)
                property_loss = tf.reduce_mean(-property_losses)
                self.properties_loss.append(property_loss)
                self.properties_transition.append(transition_params)


        self.loss = tf.reduce_sum(self.properties_loss)

        self.saver = tf.train.Saver()

