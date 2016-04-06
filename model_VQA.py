#-*- coding: utf-8 -*-
#########################################################################################
#      Author : Ching-Yao Chuang						        #
#      Dept. of Electrical Engineering, NTHU       					#
#      Tensorflow edition of Deeper LSTM+ normalized CNN for Visual Question Answering  #
#########################################################################################
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from tensorflow.models.rnn import rnn_cell
from sklearn.metrics import average_precision_score

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, max_words_q, vocabulary_size, drop_out_rate):

	self.rnn_size = rnn_size
	self.rnn_layer = rnn_layer
	self.batch_size = batch_size
	self.input_embedding_size = input_embedding_size
	self.dim_image = dim_image
	self.dim_hidden = dim_hidden
	self.max_words_q = max_words_q
	self.vocabulary_size = vocabulary_size	
	self.drop_out_rate = drop_out_rate

	# Network definitions
	# question-embedding
	self.embed_ques_W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_ques_W')

	# encoder: RNN body
	self.lstm = rnn_cell.BasicLSTMCell(rnn_size)	# change basic LSTM to LSTM
	self.lstm_dropout = rnn_cell.DropoutWrapper(self.lstm, output_keep_prob = 1 - self.drop_out_rate)
	self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout] * self.rnn_layer)

	# MULTIMODAL 
	# state-embedding
        self.embed_state_W = tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, self.dim_hidden], -0.08,0.08),name='embed_state_W')
	# image-embedding
	self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image_W')
	# score-embedding
	self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')


    def build_model(self):

	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
	question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
	question_length = tf.placeholder(tf.int32, [self.batch_size])
	label = tf.placeholder(tf.int32, [self.batch_size,]) # (batch_size, )
	
	# question --> answer(ground truth)
	labels = tf.expand_dims(label, 1) # (batch_size, 1)
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size, 1)
        concated = tf.concat(1, [indices, labels]) # (batch_size, 2)
        answer = tf.sparse_to_dense(concated, tf.pack([self.batch_size, num_answer]), 1.0, 0.0) # (batch_size, num_answer)
	
	state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])

	loss = 0.0

	# question mask
        labels_q = tf.expand_dims(question_length, 1) # b x 1
        indices_q = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
        concated_q = tf.concat(1, [indices_q, labels_q]) # b x 2
        question_mask = tf.sparse_to_dense(concated_q, tf.pack([self.batch_size, max_words_q]), 1.0, 0.0) # (batch_size, max_words_q)
	

	state_temp = []
	for i in range(max_words_q):
	    # emcedded question --> stcked_lstm
	    if i==0:
		ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
	    else:
	    	tf.get_variable_scope().reuse_variables()
		# one-hot encoding
		labels = tf.expand_dims(question[:,i-1], 1)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat(1, [indices, labels])
                question_onehot = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.vocabulary_size]), 1.0, 0.0)
		ques_emb_linear = tf.matmul(question_onehot, self.embed_ques_W)
	    ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
	    ques_emb = tf.tanh(ques_emb_drop)

	    output, state = self.stacked_lstm(ques_emb, state)
	    state_temp.append(state)

	# state
        state_temp=tf.pack(state_temp)
        mask_local = tf.to_float(question_mask)
        mask_local = tf.expand_dims(mask_local,2)
        mask_local = tf.tile(mask_local,tf.constant([1,1,self.stacked_lstm.state_size]))
        mask_local = tf.transpose(mask_local, perm=[1, 0, 2])
        # do element wise mul & sum
        state = tf.reduce_sum(tf.mul(state_temp, mask_local),0)
	
	# multimodal (fusing question & image)
	state_emb_linear = tf.matmul(state, self.embed_state_W)
        state_emb_drop = tf.nn.dropout(state_emb_linear, 1-self.drop_out_rate)
        state_emb = tf.tanh(state_emb_drop)

	image_emb_linear = tf.matmul(image, self.embed_image_W)
	image_emb_drop = tf.nn.dropout(image_emb_linear, 1-self.drop_out_rate)
	image_emb = tf.tanh(image_emb_drop)

	#scores = tf.mul(ques_enco_emb, image_emb) # (500,1024)
	scores = tf.mul(state_emb, image_emb)
	scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
	scores_emb = tf.matmul(scores_drop, self.embed_scor_W)

	# Calculate cross entropy
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scores_emb, answer)

	# Calculate loss
	loss = tf.reduce_mean(cross_entropy)

	# checking one-hot encoder
	labels = tf.expand_dims(question[:,5], 1)
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
        concated = tf.concat(1, [indices, labels])
        question_onehot = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.vocabulary_size]), 1.0, 0.0)
	
	return loss, image, question, question_length, label, question_onehot

    
    def build_generator(self):
	
	image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        question_length = tf.placeholder(tf.int32, [self.batch_size])

        state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])

        loss = 0.0

        # question mask
        labels_q = tf.expand_dims(question_length, 1) # b x 1
        indices_q = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
        concated_q = tf.concat(1, [indices_q, labels_q]) # b x 2
        question_mask = tf.sparse_to_dense(concated_q, tf.pack([self.batch_size, max_words_q]), 1.0, 0.0) # (batch_size, max_words_q)


        state_temp = []
	for i in range(max_words_q):
            # emcedded question --> stcked_lstm
            if i==0:
                ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
                labels = tf.expand_dims(question[:,i-1], 1)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat(1, [indices, labels])
                question_onehot = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.vocabulary_size]), 1.0, 0.0)
                ques_emb_linear = tf.matmul(question_onehot, self.embed_ques_W)
	    #ques_emb_linear = tf.add(ques_emb_linear, self.embed_ques_b)
	    ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
	    ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)
            state_temp.append(state)

        # state
        state_temp=tf.pack(state_temp)
        mask_local = tf.to_float(question_mask)
        mask_local = tf.expand_dims(mask_local,2)
        mask_local = tf.tile(mask_local,tf.constant([1,1,self.stacked_lstm.state_size]))
        mask_local = tf.transpose(mask_local, perm=[1, 0, 2])
        # do element wise mul & sum
        state = tf.reduce_sum(tf.mul(state_temp, mask_local),0)

        # multimodal (fusing question & image)
	state_emb_linear = tf.matmul(state, self.embed_state_W)
	state_emb_drop = tf.nn.dropout(state_emb_linear, 1-self.drop_out_rate)
	state_emb = tf.tanh(state_emb_drop)

	image_emb_linear = tf.matmul(image, self.embed_image_W)
        image_emb_drop = tf.nn.dropout(image_emb_linear, 1-self.drop_out_rate)
        image_emb = tf.tanh(image_emb_drop)

        scores = tf.mul(state_emb, image_emb)
        scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
	scores_emb = tf.matmul(scores_drop, self.embed_scor_W)

        # FINAL ANSWER
	generated_ANS = tf.matmul(scores, self.embed_scor_W)

	return generated_ANS, image, question, question_length, question_mask
    
#####################################################
#                 Global Parameters		    #  
#####################################################
print 'Loading parameters ...'
# Data input setting
input_img_h5 = 'data_img.h5'
input_ques_h5 = 'data_prepro.h5'
input_json = 'data_prepro.json'

# Train Parameters setting
learning_rate = 3e-4			# learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1		# at what iteration to start decaying learning rate? (-1 = dont)
learning_rate_decay_every = 50000	# every how many iterations thereafter to drop LR by half?
batch_size = 500			# batch_size for each iterations
max_iters = 150000			# max number of iterations to run for
input_embedding_size = 200		# he encoding size of each token in the vocabulary
rnn_size = 512				# size of the rnn in number of hidden nodes in each layer
rnn_layer = 2				# number of the rnn layer
dim_image = 4096
dim_hidden = 1024 #1024			# size of the common embedding vector
num_output = 1000			# number of output answers
img_norm = 1				# normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083

# Check point
checkpoint_path = 'model/'

# misc
n_epochs = 300
backend = 'cudnn'
seed = '123'
max_words_q = 26
num_answer = 1000
#####################################################

def get_data():

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        #-----1~82460-----
        tem = hf.get('img_pos_train')
	# convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem,(1,4096)))

    return dataset, img_feature, train_data

def get_data_test():
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_test')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        # -----1~82460-----
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
	# MC_answer_test
	tem = hf.get('MC_ans_test')
	test_data['MC_ans_test'] = np.array(tem)

    print('Normalizing image feature')
    if img_norm:
        tem =  np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem,(1,4096)))

    return dataset, img_feature, test_data

def train():
    print 'loading dataset...'
    dataset, img_feature, train_data = get_data()
    num_train = train_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())+1
    print 'vocabulary_size : ' + str(vocabulary_size)

    print 'constructing  model...'
    model = Answer_Generator(
	    rnn_size = rnn_size,
            rnn_layer = rnn_layer,
	    batch_size = batch_size,
	    input_embedding_size = input_embedding_size,
	    dim_image = dim_image,
	    dim_hidden = dim_hidden,
	    max_words_q = max_words_q,	
	    vocabulary_size = vocabulary_size,
	    drop_out_rate = 0.5)

    tf_loss, tf_image, tf_question, tf_question_length, tf_label, tf_hot = model.build_model()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=100)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()

    print 'start training...'

    tStart_total = time.time()  
    for epoch in range(n_epochs):  
	# shuffle the training data
        index = np.arange(num_train)
        np.random.shuffle(index)
	train_data['question'] = train_data['question'][index,:]
        train_data['length_q'] = train_data['length_q'][index]
        train_data['answers'] = train_data['answers'][index]
        train_data['img_list'] = train_data['img_list'][index]

	tStart_epoch = time.time()
        loss_epoch = np.zeros(num_train)
	num_batch = num_train/batch_size + 1
	split_batch = np.array_split(np.arange(num_train),num_batch)
	for current_batch_file_idx in split_batch:
	    tStart = time.time()
	    # set current data
	    current_train_data = {}
            current_question = train_data['question'][current_batch_file_idx,:]
            current_length_q = train_data['length_q'][current_batch_file_idx]
            current_answers = train_data['answers'][current_batch_file_idx]
            current_img_list = train_data['img_list'][current_batch_file_idx]
	    # init parameters
	    current_img = np.zeros((batch_size, dim_image))
	    current_img = img_feature[current_img_list,:] # (batch_size, dim_image)
	    if(len(current_img)<500):
		pad_img = np.zeros((500-len(current_img),dim_image))
		pad_q = np.zeros((500-len(current_img),max_words_q))
		pad_q_len = np.zeros(1,)
		pad_ans = np.zeros(1,)
	 	current_img = np.concatenate((current_img, pad_img))
	   	current_question = np.concatenate((current_question, pad_q))
	        current_length_q = np.concatenate((current_length_q, pad_q_len))
		current_answers = np.concatenate((current_answers, pad_ans))
	    # do the training process!!!
	    _, loss, onehot = sess.run(
                    [train_op, tf_loss, tf_hot],
                    feed_dict={
                        tf_image: current_img,
                        tf_question: current_question,
                        tf_question_length: current_length_q,
                        tf_label: current_answers
                        })

	    print onehot
	    print np.sum(onehot[1,:])
	    print np.sum(onehot[10,:])
            loss_epoch[current_batch_file_idx] = loss
            tStop = time.time()
            print ("Epoch:", epoch, " Batch:", current_batch_file_idx[0], " Loss:", loss)
            print ("Time Cost:", round(tStop - tStart,2), "s")
	if np.mod(epoch, 25) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
	    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=epoch)
	print "Epoch:", epoch, " done. Loss:", np.mean(loss_epoch)
        tStop_epoch = time.time()
        print "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s"

    print "Finally, saving the model ..."
    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"

def test(model_path='model2/model-300'):
    print 'loading dataset...'
    dataset, img_feature, test_data = get_data_test()
    num_test = test_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())+1
    print 'vocabulary_size : ' + str(vocabulary_size)

    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = batch_size,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,
            vocabulary_size = vocabulary_size,
            drop_out_rate = 0.5)

    tf_answer, tf_image, tf_question, tf_question_length, tf_question_mask = model.build_generator()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    tStart_total = time.time()
    result = []
    for current_batch_start_idx in xrange(0,num_test-1,batch_size):
    #for current_batch_start_idx in xrange(0,3,batch_size):
	tStart = time.time()
        # set data into current*
        if current_batch_start_idx + batch_size < num_test:
            current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
        else:
            current_batch_file_idx = range(current_batch_start_idx,num_test)

        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]
        current_img_list = test_data['img_list'][current_batch_file_idx]
    	current_ques_id  = test_data['ques_id'][current_batch_file_idx]
        current_img = img_feature[current_img_list,:] 

	# deal with the last batch
        if(len(current_img)<500):
                pad_img = np.zeros((500-len(current_img),dim_image),dtype=np.int)
                pad_q = np.zeros((500-len(current_img),max_words_q),dtype=np.int)
                pad_q_len = np.zeros(500-len(current_length_q),dtype=np.int)
		pad_q_id = np.zeros(500-len(current_length_q),dtype=np.int)
		pad_ques_id = np.zeros(500-len(current_length_q),dtype=np.int)
		pad_img_list = np.zeros(500-len(current_length_q),dtype=np.int)
                current_img = np.concatenate((current_img, pad_img))
                current_question = np.concatenate((current_question, pad_q))
                current_length_q = np.concatenate((current_length_q, pad_q_len))
		current_ques_id = np.concatenate((current_ques_id, pad_q_id))
		current_img_list = np.concatenate((current_img_list, pad_img_list))


	generated_ans = sess.run(
                tf_answer,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question,
                    tf_question_length: current_length_q
                    })

	top_ans = np.argmax(generated_ans, axis=1)


	# initialize json list
	for i in xrange(0,500):
	    ans = dataset['ix_to_ans'][str(top_ans[i]+1)]
	    if(current_ques_id[i] == 0):
		continue
	    result.append({u'answer': ans, u'question_id': str(current_ques_id[i])})
        
	tStop = time.time()
        print ("Testing batch: ", current_batch_file_idx[0])
        print ("Time Cost:", round(tStop - tStart,2), "s")	
    print ("Testing done.")
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
    # Save to JSON
    print 'Saving result...'
    my_list = list(result)
    dd = json.dump(my_list,open('data.json','w'))

if __name__ == '__main__':
    # commend test() when you train()
    with tf.device('/gpu:'+str(1)):
        train()
    test()
    
