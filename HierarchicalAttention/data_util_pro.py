# -*- coding: utf-8 -*-
import codecs
import numpy as np
import re
import jieba
import time
# import word2vec
import os
import pickle
PAD_ID = 0
#from tflearn.data_utils import pad_sequences
_GO="_GO"
_END="_END"
_PAD="_PAD"
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def create_voabulary(word2vec_model_path='sgns.wiki.bigram-char',embed_size=300,simple=None,name_scope=''):  
    vocabulary_word2index={}
    vocabulary_index2word={}
    vocabulary_wordind2vec = {}
    vocabulary_word2index['_PAD']=0
    vocabulary_index2word[0]='_PAD'
    vocabulary_wordind2vec[0] = [0] * embed_size
    word2vec_file = codecs.open(word2vec_model_path, 'r', 'utf8')
    ind = 0
    for line in word2vec_file:
        if ind == 0:
            ind += 1
            continue
        arr = line.strip().split(" ")
        word, vec = arr[0].encode('utf-8'), [float(sub) for sub in arr[1:]]        
        #print word
        #print vec
        if word in vocabulary_word2index or len(vec) != embed_size:
            continue
        vocabulary_word2index[word] = ind
        vocabulary_index2word[ind] = word
        vocabulary_wordind2vec[ind] = vec
        ind += 1
    #词典中缺失的词随机初始化embedding
    vocabulary_word2index['_UNK'] = ind
    vocabulary_index2word[ind] = '_UNK'
    bound = np.sqrt(6.0) / np.sqrt(ind + 1)
    vocabulary_wordind2vec[ind] = np.random.uniform(-bound, bound, embed_size)
    
    #test
    # print '似曾相识index: %s' % str(vocabulary_word2index.get('似曾相识'.encode('utf-8'), ind))
    # print '相见恨晚index: %s' % str(vocabulary_word2index.get('相见恨晚'.encode('utf-8'), ind))

    return vocabulary_word2index, vocabulary_index2word, vocabulary_wordind2vec


def load_data(vocabulary_word2index, max_sentence_num, max_sentence_length, \
              valid_portion=0.05, max_training_data=1000000,train_data_path='train_sample.dat'):
    print("load_data.started...")
    data_file = codecs.open(train_data_path, 'r', 'utf8')
    vocab_size = len(vocabulary_word2index)
    X = []
    Y = []

    for line in data_file:
        arr = line.strip().split('\t')
        if len(arr) != 5:
            continue
        nid, mthid, site_name, content, label = arr
        content_seg = ' '.join(jieba.lcut(content))
        sentences = re.split('(。|！|\!|\.|？|\?)'.encode('utf8'), content_seg.encode('utf8'))
        b_sentences = []
        #整合句子、标点
        for i in range(int(len(sentences)/2)):
            sub = sentences[2*i] + sentences[2*i+1]
            if len(sub.strip().split(' ')) < 4:
                continue
            b_sentences.append(sub)

        one_sample = np.zeros((max_sentence_num, max_sentence_length))

        for i, sentence in enumerate(b_sentences):
            if i < max_sentence_num:
                words = sentence.strip().split(' ')
                for j, word in enumerate(words):
                    index = vocabulary_word2index.get(word, vocab_size)
                    if j < max_sentence_length:
                        one_sample[i, j] = index
        X.append(one_sample)
        #label
        label = int(label.strip().strip('__label__'))
        Y.append(label)  

    #划分训练集、测试集
    number_examples = len(X)
    print("number_examples:",number_examples) #
    train = (X[0:int((1 - valid_portion) * number_examples)], Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])

    return train, test

def load_final_test_data(file_path):
    final_test_file_predict_object = codecs.open(file_path, 'r', 'utf8')
    lines=final_test_file_predict_object.readlines()
    pred_lists_result=[]
    for i,line in enumerate(lines):
        arr = line.strip().split('\t')
        if len(arr) != 8:
            continue
        nid, mthid, site_name, pred_string = arr[:4]
        pred_string = pred_string.strip().replace("\n","")
        pred_lists_result.append((nid, pred_string))
    print("length of total question lists:",len(pred_lists_result))
    return pred_lists_result

def load_data_predict(vocabulary_word2index, max_sentence_num, max_sentence_length, vocab_size, pred_lists_result):
    print len(pred_lists_result)
    final_list = []
    cnt = 0
    for i, item in enumerate(pred_lists_result):
        nid, pred_string = item
        pred_string_seg = ' '.join(jieba.lcut(pred_string))
        sentences = re.split('(。|！|\!|\.|？|\?)'.encode('utf8'), pred_string_seg.encode('utf8'))
        b_sentences = []
        #整合句子、标点
        for i in range(int(len(sentences)/2)):
            sub = sentences[2*i] + sentences[2*i+1]
            if len(sub.strip().split(' ')) < 4:
                continue
            b_sentences.append(sub)

        #生成一个样本
        one_sample = np.zeros((max_sentence_num, max_sentence_length))

        for i, sentence in enumerate(b_sentences):
            if i < max_sentence_num:
                words = sentence.strip().split(' ')
                for j, word in enumerate(words):
                    index = vocabulary_word2index.get(word, vocab_size)
                    if j < max_sentence_length:
                        one_sample[i, j] = index
        final_list.append((nid, one_sample))
        cnt += 1
        if cnt % 100000 == 0:
            print "processed nid num: ", cnt
    number_examples = len(final_list)
    print("number_examples:",number_examples) #
    return  final_list
        
if __name__ == "__main__":
    vocabulary_word2index, vocabulary_index2word, vocabulary_wordind2vec \
        = create_voabulary(word2vec_model_path='./sgns.wiki.bigram-char')
    vocab_size = len(vocabulary_word2index) 
    pred_lists_result = load_final_test_data('./predict/tmp_data') 
    time1 = time.time()
    final_list = load_data_predict(vocabulary_word2index, 150, 50, vocab_size, pred_lists_result)
    time2 = time.time() 
    cost_time = time2 - time1
    print "cost_time: ", cost_time
    #print pred_lists_result[0][1]

    print final_list[0]
    #train, test = load_data(vocabulary_word2index, 20, 80, train_data_path='./demo_sample.data')
    #trainX, trainY = train
    #testX, testY = test
     


    
