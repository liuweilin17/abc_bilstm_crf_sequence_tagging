#coding=utf-8
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class DataProcessor(object):
    def __init__(self, data_file, label_file):
        self.data_file = data_file
        self.label_file = label_file
        self.docs = self.read_file(self.data_file)
        self.labels = self.read_labels(self.label_file)
        self.vocab = self.gen_vocab(self.docs)
        self.index_docs = self.word2index(self.docs, self.vocab)
        random_order = np.random.permutation(len(self.labels))
        self.labels = self.labels[random_order, :]
        self.index_docs = self.index_docs[random_order, :]
        self.docs = self.docs[random_order]

    @staticmethod
    def read_file(data_file):
        docs = np.load(data_file)
        docs = [str(doc) for doc in docs]
        docs = [doc.decode('utf-8') for doc in docs]
        docs = np.array(docs)
        return docs

    @staticmethod
    def read_labels(label_file, maskid=0):
        labels = np.load(label_file)
        max_len = max([len(label) for label in labels])
        labels = [label + [maskid] * (max_len - len(label)) for label in labels]   #padding
        labels = np.array(labels) #convert the input to an array
        return labels

    @staticmethod
    def gen_vocab(docs, maskid=0):
        word_set = set(word for doc in docs for word in doc.strip()) # ['中' ‘美’ ‘国’]
        vocab = {word: index + 1 for index, word in enumerate(word_set)}
        
        vocab['mask'] = maskid #add 'mask'
        vocab[u'unk'] = len(vocab)    #add 'unk'
        #add variables
        vocab[u'Var_X'] = len(vocab)
        vocab[u'Var_Y'] = len(vocab)
        return vocab

    @staticmethod
    def word2index(docs, vocab):
        index_docs = []
        for doc in docs:
            index_list = []
            for word in doc:
                    if word not in vocab:
                            word = u'unk'
                    index_list.append(vocab[word])
            index_docs.append(index_list)

        max_len = max([len(doc) for doc in index_docs]) + 1
        index_docs = [doc + [vocab['mask']] * (max_len - len(doc)) for doc in index_docs]
        index_docs = np.array(index_docs)
        return index_docs

    def get_inputs(self):
        return self.index_docs

    def get_labels(self):
        return self.labels

    def get_vocab(self):
        return self.vocab

    def get_raw_docs(self):
        return self.docs

def get_data(data_file, label_file):
    dataset = DataProcessor(data_file, label_file)
    data = dataset.get_inputs()
    labels = dataset.get_labels()
    vocab = dataset.get_vocab()
    docs = dataset.get_raw_docs()
    return data, labels, vocab, docs

def get_test_data(test_file, vocab, max_len):
    print("in get_test_data:")
    docs = np.load(test_file)
    docs = [str(doc) for doc in docs]
    docs = [doc.decode('utf-8') for doc in docs]
    docs = np.array(docs)
    print("docs:" + str(docs.shape))
    index_docs = []
    for doc in docs:
        index_list = []
        for word in doc:
                if word not in vocab:
                        word = u'unk'
                index_list.append(vocab[word])
        index_docs.append(index_list)

    #max_len = max([len(doc) for doc in index_docs]) + 1
    index_docs = [doc + [vocab['mask']] * (max_len - len(doc)) for doc in index_docs]
    index_docs = np.array(index_docs)
    return index_docs    

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def cal_pre_rec(label_list, prediction, ground_truth):

    true_record = {}
    predict_record = {}
    ground_record = {}
    ret = []
    for label in label_list:
        true_record[label] = 0
        predict_record[label] = 0
        ground_record[label] = 0

    for predition_, ground_truth_ in zip(prediction, ground_truth):
        for i, j in zip(predition_, ground_truth_):
            if i == j:
                true_record[i] += 1
                predict_record[i] += 1
                ground_record[i] += 1
            else:
                predict_record[i] += 1
                ground_record[j] += 1

    for label in label_list:
        if predict_record[label] != 0:
            prec = true_record[label] / float(predict_record[label])
        else:
            prec = -1
        if ground_record[label] != 0:
            recall = true_record[label] / float(ground_record[label])
        else:
            recall = -1
        if prec == -1 or recall == -1 or prec == 0 or recall == 0:
            F1 = -1
        else:
            F1 = (prec * recall) / float(2 * prec * recall)

        ret.append([label ,prec, recall, F1])
    return ret

if __name__=='__main__':
    x, y1, vocab, docs = GetData('/home/yanyun/leeyanyun/lyy0714/data_170816.npy', '/home/yanyun/leeyanyun/lyy0714/label_170816.npy')

