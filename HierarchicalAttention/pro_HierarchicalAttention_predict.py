# -*- coding: utf-8 -*-
#prediction using model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from data_util_pro import load_data, create_voabulary, load_final_test_data, load_data_predict
# from tflearn.data_utils import pad_sequences #to_categorical
import os
import codecs
from pro_HierarchicalAttention_model import HierarchicalAttention
os.environ['CUDA_VISIBLE_DEVICES']='6'

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",1999,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 80, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint_hier_atten_title/text_hier_atten_title_desc_checkpoint_0609/","checkpoint location for the model")
# tf.app.flags.DEFINE_integer("sequence_length",100,"max sentence length")
tf.app.flags.DEFINE_integer("max_sentence_num",100,"max sentence num")
tf.app.flags.DEFINE_integer("max_sentence_length",100,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
#train-zhihu4-only-title-all.txt
tf.app.flags.DEFINE_string("traning_data_path","train-zhihu4-only-title-all.txt","path of traning data.") #O.K.train-zhihu4-only-title-all.txt-->training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
tf.app.flags.DEFINE_string("word2vec_model_path","zhihu-word2vec-title-desc.bin-100","word2vec's vocabulary and vectors") #zhihu-word2vec.bin-100-->zhihu-word2vec-multilabel-minicount15.bin-100
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.")
# tf.app.flags.DEFINE_integer("num_sentences", 4, "number of sentences in the document") #每10轮做一次验证
tf.app.flags.DEFINE_integer("hidden_size",100,"hidden size")
tf.app.flags.DEFINE_string("predict_out_file","checkpoint_hier_atten_title/text_hier_atten_title_desc_checkpoint_0609/zhihu_result_hier_atten_multilabel_b512_DROPOUT4.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'test-zhihu-forpredict-title-desc-v6.txt',"target file path for final prediction") #test-zhihu-forpredict-v4only-title.txt

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
# 1.load data with vocabulary of words and labels


def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word, vocabulary_wordind2vec \
        = create_voabulary(simple='simple',word2vec_model_path=FLAGS.word2vec_model_path,name_scope="hierAtten")
    vocab_size = len(vocabulary_word2index)
    # vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="hierAtten")
    nid_pred_string_list = load_final_test_data(FLAGS.predict_source_file)
    # questionid_question_lists=load_final_test_data(FLAGS.predict_source_file)
    test = load_data_predict(vocabulary_word2index,
                            FLAGS.max_sentence_num,
                            FLAGS.max_sentence_length,
                            vocab_size,
                            nid_pred_string_list)
    testX=[]
    pred_info_list=[]
    for item in test:
        nid, mthid, pubt_time, one_sample = item
        pred_info_list.append((nid, mthid, pubt_time))
        testX.append(one_sample)
   # 2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 3.Instantiate Model
        model = HierarchicalAttention(FLAGS.num_classes, 
                                      FLAGS.learning_rate, 
                                      FLAGS.batch_size, 
                                      FLAGS.decay_steps,
                                      FLAGS.decay_rate, 
                                      FLAGS.max_sentence_num,
                                      FLAGS.max_sentence_length, 
                                      vocab_size, 
                                      FLAGS.embed_size, 
                                      FLAGS.hidden_size,
                                      FLAGS.is_training, multi_label_flag=FLAGS.multi_label_flag)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"/checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 4.feed data, to get logits
        number_of_training_data=len(testX);print("number_of_training_data:",number_of_training_data)
        index=0
        predict_out_file = codecs.open(FLAGS.predict_out_file, 'a', 'utf8')
        for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
            logits=sess.run(model.logits,feed_dict={model.input_x:testX[start:end],model.dropout_keep_prob:1}) #'shape of logits:', ( 1, 1999)
            # 5. get lable using logtis
            #predicted_labels=get_label_using_logits(logits[0],vocabulary_index2word_label)
            #write_question_id_with_labels(question_id_list[index],predicted_labels,predict_target_file_f)
            pred_info_sublist=pred_info_list[start:end]
            get_label_1prob_using_logits_batch(pred_nid_sublist, logits, predict_out_file)

            index=index+1
        predict_out_file.close()

def get_label_1prob_using_logits_batch(pred_info_list, logits_batch, f_out):
    for i, logits in enumerate(logits_batch):
        pred_index = np.argsort(logits)[-1]
        pred_logits1 = logits[1]
        pred_logits0 = logits[0]
        pred_prob1 = 1.0 / (1 + np.exp(-pred_logits1))
        softmax_prob1 = np.exp(pred_logits1) / (np.exp(pred_logits1) + np.exp(pred_logits0))
        pred_label = pred_index
        nid, mthid, pubt_time = pred_info_list[i]
        f_out.write('\t'.join([str(sub) for sub in [mthid, nid, pred_label, pred_prob1, softmax_prob1, pubt_time]]) + '\n')

# get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=5):
    index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list=index_list[::-1]
    label_list=[]
    for index in index_list:
        label=vocabulary_index2word_label[index]
        label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return label_list

# get label using logits
def get_label_using_logits_with_value(logits,vocabulary_index2word_label,top_number=5):
    index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list=index_list[::-1]
    value_list=[]
    label_list=[]
    for index in index_list:
        label=vocabulary_index2word_label[index]
        label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        value_list.append(logits[index])
    return label_list,value_list

# write question id and labels to file system.
def write_question_id_with_labels(question_id,labels_list,f):
    labels_string=",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

# get label using logits
def get_label_using_logits_batch(question_id_sublist,logits_batch,vocabulary_index2word_label,f,top_number=5):
    #print("get_label_using_logits.shape:", logits_batch.shape) # (10, 1999))=[batch_size,num_labels]===>需要(10,5)
    for i,logits in enumerate(logits_batch):
        index_list=np.argsort(logits)[-top_number:] #print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
        index_list=index_list[::-1]
        label_list=[]
        for index in index_list:
            label=vocabulary_index2word_label[index]
            label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        #print("get_label_using_logits.label_list",label_list)
        write_question_id_with_labels(question_id_sublist[i], label_list, f)
    f.flush()
    #return label_list
# write question id and labels to file system.
def write_question_id_with_labels(question_id,labels_list,f):
    labels_string=",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

if __name__ == "__main__":
    tf.app.run()
