#coding:utf-8
import tensorflow as tf
from utils import *
from model import PropertiesClass
import os, time, datetime, json
from os.path import dirname, join

#flags.DEFINE_string("para_name","default_val", "description")
tf.flags.DEFINE_integer('embedding_size', 100, 'embedding size')
tf.flags.DEFINE_integer('hidden_size', 200, 'hidden size')
tf.flags.DEFINE_integer('classes', 10, 'item counts')#5
tf.flags.DEFINE_integer('decay_steps', 500, 'learn decay steps')
tf.flags.DEFINE_string('data_file', '/home/wlliu/entity_recognition/abc_bilstm_crf_sequence_tagging/data.npy', 'location of dataset')
tf.flags.DEFINE_string('label_file', '/home/wlliu/entity_recognition/abc_bilstm_crf_sequence_tagging/label.npy', 'location of labelset')
tf.flags.DEFINE_float('keep_prob', 0.7, 'keep prob of dropout layer')
tf.flags.DEFINE_integer('batch_size', 50, 'batch of train and test')
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer('epoch', 50, 'number of train')
tf.flags.DEFINE_integer('train_rate', 0.7, 'rate of dataset')
tf.flags.DEFINE_boolean("Location_way", False, "use location in sentnce")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("is_train", True, 'this is a flag')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#data process
x, y, vocab, docs = get_data(FLAGS.data_file, FLAGS.label_file)
vocab_index = vocab.values()
vocab_word = vocab.keys()
FLAGS.vocab_num = len(vocab)
num, max_length = x.shape
FLAGS.max_length = max_length

train_x = x[: int(FLAGS.train_rate * num)]
train_y = y[: int(FLAGS.train_rate * num)]
valid_x = x[int(FLAGS.train_rate * num) :]
valid_y = y[int(FLAGS.train_rate * num) :]
valid_docs = docs[int(FLAGS.train_rate * num):]

data_num = np.sum(np.sign(valid_x), dtype=np.float) #sign: >0:1. <0:-1 =0:0
label_num = np.sum(np.sign(valid_y), dtype=np.float)
label_rate = 1. - label_num/data_num

print("label_rate:" + str(label_rate))
print("train_x shape:" + str(train_x.shape))
print("train_y shape:" + str(train_y.shape))
print("valid_x shape:" + str(valid_x.shape))
print("valid_y shape:" + str(valid_y.shape))
print("valid_docs shape:" + str(valid_docs.shape))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = PropertiesClass(FLAGS)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.RMSPropOptimizer(1e-4)   #AdamOptimizer      AdadeltaOptimizer
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        print 'writing to {}\n'.format(out_dir)

        #summary --- train
        loss_summary = tf.summary.scalar('loss', model.loss)
        dev_summary_op = tf.summary.merge_all()
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        #checkpoint --- evaluate
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        jsObj = json.dumps(vocab)
        fileObject = open(os.path.join(out_dir, "vocab"), 'w')
        fileObject.write(jsObj)
        fileObject.close()

        sess.run(tf.global_variables_initializer())

        def train_step(x, y):
            feed_dict = {
                model.input_x: x,
                model.input_y: y,
                model.keep_prob: FLAGS.keep_prob
            }
            loss, step, _, summaries = sess.run([model.loss, global_step, train_op, dev_summary_op], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print '{}: step {}, loss {:g}'.format(time_str, step, loss)
            dev_summary_writer.add_summary(summaries, step)

        def valid_step(x, y):
            model.is_train = False
            feed_dict = {
                model.input_x: x,
                model.input_y: y,
                model.keep_prob: 1.
            }

            y_all = [y]
            properties_pred, transition_params, loss, step, properties_loss, sequence_length = sess.run([model.properties_pred, model.properties_transition, model.loss, global_step, model.properties_loss, model.sequence_length], feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print '{}: step {}, loss {:g}'.format(time_str, step, loss)
            print 'Accuracy of each category:'
            #print '    date_loss      {:g}, 0_rate {:g}'.format(properties_loss[0], label_rate)

            for j, property_probs in enumerate(properties_pred):
                #property_probs(ndarray):[sentence number, FLAGS.max_length, FLAGS.classes]
                
                correct_labels = 0
                total_labels = 0
                correct_sequence = 0
                total_sequence = 0
                prediction = []
                ground_truth = []
                
                #print("property_probs: " + str(property_probs.shape))
                #print("y_all[j]: " + str(y_all[j].shape))
                #print("sequence_length: " + str(sequence_length.shape))
                for tf_unary_scores_, y_, sequence_lengths_ in zip(property_probs, y_all[j], sequence_length):
                        #tf_unary_scores:(max_length, classes), y_:(max_length), sequence_length:int
                        tf_unary_scores_ = tf_unary_scores_[:sequence_lengths_]
                        y_ = y_[:sequence_lengths_]

                        #viterbi_sequence is the predicted result
                        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transition_params[j])
                        

                        correct_labels += np.sum(np.equal(viterbi_sequence, y_)) #the number of correct labels in viterbi_sequence
                        total_labels += sequence_lengths_
                        if(np.sum(np.equal(viterbi_sequence, y_)) == len(y_)):
                                correct_sequence += 1
                        
                        total_sequence += 1
                        
                        prediction.append(viterbi_sequence)
                        ground_truth.append(y_)
                        
                
                accuracy = 100.0 * correct_labels / float(total_labels)
                accurseq = 100.0 * correct_sequence / float(total_sequence)

                print 'Accuarcy: %.2f%%' % accuracy + ' AccuracySequence: %.2f%%' % accurseq
                print '----------------------'
                for line in cal_pre_rec(range(FLAGS.classes), prediction, ground_truth):
                    #print 'label ' + label + '--precision: %.2f%%' % prec + 'recall: %.2f%%' % recall + 'F1: %.2f%%' % F1
                    print 'label %d' % line[0] + '--precision: %.2f%%' % line[1] + ' recall: %.2f%%' % line[2] + ' F1: %.2f%%' % line[3]                    

        #zip(train_x, train_y) 是一个list，list元素是一个大小为2的tuple，tuple的第一个元素为train_x的一行，第二元素是对应train_y的一行
        #batch_iter函数切分数据成若干batch，每个batch是zip(train_x, train_y)的一部分
        batches = batch_iter(zip(train_x, train_y), FLAGS.batch_size, FLAGS.epoch)
        for batch in batches:
            #zip(*) could be treated as unzip
            #x_ is tuple, each row represents a sentence; y_ is a tuple, each row represents the label of sentence
            x_, y_ = zip(*batch) 
            train_step(x_, y_)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                valid_step(valid_x, valid_y)
            if current_step % FLAGS.checkpoint_every == 0:
                path = model.saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        
    print("Optimization Finished!")

