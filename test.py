#coding:utf-8
import tensorflow as tf
from utils import *
from model import PropertiesClass
import os, time, datetime, json
from os.path import dirname, join

tf.flags.DEFINE_string("model_name", "runs/1504760284/checkpoints/model-600", "Name(Path) of model")
tf.flags.DEFINE_string("test_file", "test.npy", "test file")
tf.flags.DEFINE_string("vocab", "runs/1504760284/vocab", "vocabulary dict")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


with tf.Session() as sess:
	saver = tf.train.import_meta_graph(FLAGS.model_name + ".meta")
	saver.restore(sess, FLAGS.model_name)
	
	graph = tf.get_default_graph()
	'''
	for op in graph.get_operations():
		print(op.name,op.values())
	'''
	#get x from vocab
	with open(FLAGS.vocab, "r") as f:
		vocab = json.load(f)
	max_length = 101
	x = get_test_data(FLAGS.test_file, vocab, max_length)
	print("x:" + str(x.shape))
	mask_x = tf.sign(x)
	sequence_length = sess.run([tf.reduce_sum(mask_x, axis=1)])

	feed_dict = {
	    graph.get_tensor_by_name('input_x:0'): x, 
        graph.get_tensor_by_name('keep_prob:0'): 1
	}

	'''
	logit_op = graph.get_tensor_by_name('output/logits:0')
    transition_params_op = graph.get_tensor_by_name('transitions:0')
    logits,transition_params = session.run([logit_op, transition_params_op],feed_dict)
	viterbi_decode(logits=logits,transition_params=transition_params,
	seq_length=test_seq_length,x_batch=word_index_sentences_test_pad,word_alphabet=word_alphabet,label_alphabet=label_alphabet, 
	prefix_filename="test",beginZero=PadZeroBegin)
	'''
	properties_probs_ = graph.get_tensor_by_name('properties_softmax/property_probs:0')
	transition_params_ = graph.get_tensor_by_name('properties_calculate0/transitions:0') 
	properties_probs, transition_params = sess.run([properties_probs_, transition_params_], feed_dict)

	prediction = []
	#print("properties_probs:" + str(properties_probs.shape))
	#print("sequence_length:" + str(sequence_length.shape))
	print("properties_probs:" + str(properties_probs))
	print("sequence_length:" + str(sequence_length))

	for tf_unary_scores_, sequence_lengths_ in zip(properties_probs, sequence_length[0]):
		#tf_unary_scores:(max_length, classes), y_:(max_length), sequence_length:int
		tf_unary_scores_ = tf_unary_scores_[:sequence_lengths_]

		#viterbi_sequence is the predicted result
		viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transition_params)
		prediction.append(viterbi_sequence)

	print(prediction[0])
	docs = np.load(FLAGS.test_file)
	print(docs[0])






