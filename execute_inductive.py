import time
import numpy as np
import tensorflow as tf
import sys

from models import GAT
from utils import process
from utils import process_inductive

dataset = 'ppi'

checkpt_file = 'pre_trained/{}/mod_{}.ckpt'.format(dataset,dataset)

# training params
batch_size = 2
nb_epochs = 2000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0  # weight decay
dropout = 0
hid_units = [256, 256] # numbers of hidden units per each attention head in each layer
n_heads = [4, 4, 6] # additional entry for the output layer
residual = True
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: PPI')
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))


train_adj, val_adj, test_adj, train_feat, val_feat, test_feat, train_labels, val_labels, test_labels, train_nodes, val_nodes, test_nodes, train_mask, val_mask, test_mask = process_inductive.load_ppi(dataset)

for i in range(train_feat.shape[0]):
    train_feat[i] = process.preprocess_features2(train_feat[i])
for i in range(val_feat.shape[0]):
    val_feat[i] = process.preprocess_features2(val_feat[i])
for i in range(test_feat.shape[0]):
    test_feat[i] = process.preprocess_features2(test_feat[i])

nb_nodes = train_feat.shape[1]
ft_size = train_feat.shape[2]
nb_classes = train_labels.shape[2]

train_biases = process.adj_to_bias(train_adj, [nb_nodes]*train_adj.shape[0], nhood=1)
val_biases = process.adj_to_bias(val_adj, [nb_nodes]*val_adj.shape[0], nhood=1)
test_biases = process.adj_to_bias(test_adj, [nb_nodes]*test_adj.shape[0], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_sigmoid_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.micro_f1(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = train_feat.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: train_feat[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: train_biases[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: train_labels[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: dropout, ffd_drop: dropout})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = val_feat.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: val_feat[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: val_biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: val_labels[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0, ffd_drop: 0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1
            
            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = test_feat.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: test_feat[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: test_biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: test_labels[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
