import argparse
from datetime import datetime
import errno
from glob import glob
import numpy as np
import os
import pickle
from time import time

import tensorflow as tf

from model import Model


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_data(train_file_path, test_file_path):

    train, val, test = [], [], []
    for file_path_i in glob(os.path.join(train_file_path, '*.pkl')):
        with open(os.path.join(file_path_i), 'rb') as f:
            if 'train' in file_path_i:
                train += [pickle.load(f) + (file_path_i,)]
            elif 'val' in file_path_i:
                val += [pickle.load(f) + (file_path_i,)]
            else:
                assert 'test' in file_path_i

    for file_path_i in glob(os.path.join(test_file_path, '*.pkl')):
        with open(os.path.join(file_path_i), 'rb') as f:
            if 'train' not in file_path_i and 'val' not in file_path_i:
                assert 'test' in file_path_i
                test += [pickle.load(f) + (file_path_i,)]

    return train, val, test


def batch_iter(c, q, l, a, c_real_len, q_real_len,
               batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    c = np.array(c)
    q = np.array(q)
    l = np.array(l)
    a = np.array(a)
    c_real_len = np.array(c_real_len)
    q_real_len = np.array(q_real_len)
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        print("num_epochs")
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            c_shuffled = c[shuffle_indices]
            q_shuffled = q[shuffle_indices]
            l_shuffled = l[shuffle_indices]
            a_shuffled = a[shuffle_indices]
            c_real_len_shuffled = c_real_len[shuffle_indices]
            q_real_len_shuffled = q_real_len[shuffle_indices]
        else:
            c_shuffled = c
            q_shuffled = q
            l_shuffled = l
            a_shuffled = a
            c_real_len_shuffled = c_real_len
            q_real_len_shuffled = q_real_len

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index < data_size:
                c_batch = c_shuffled[start_index:end_index]
                q_batch = q_shuffled[start_index:end_index]
                l_batch = l_shuffled[start_index:end_index]
                a_batch = a_shuffled[start_index:end_index]
                c_real_len_batch = c_real_len_shuffled[start_index:end_index]
                q_real_len_batch = q_real_len_shuffled[start_index:end_index]
                yield list(zip(c_batch, q_batch, l_batch, a_batch,
                               c_real_len_batch, q_real_len_batch))


def concatenate_datasets(datasets):
    q, a, c, l, c_real_len, q_real_len = [], [], [], [], [], []
    for dataset in datasets:
        q += dataset[0]
        a += dataset[1]
        c += dataset[2]
        l += dataset[3]
        c_real_len += dataset[4]
        q_real_len += dataset[5]
    return q, a, c, l, c_real_len, q_real_len


def main(args):

    (c_max_len,
     s_max_len,
     q_max_len,
     c_vocab_size,
     q_vocab_size,
     a_vocab_size,
     ) = [int(x) for x in args.train_data_path.split("_")[-6:]]

    (train_datasets,
     val_datasets, test_datasets) = read_data(args.train_data_path,
                                              args.test_data_path)

    # Concatenate training datasets
    train_dataset = concatenate_datasets(train_datasets)

    # Concatenate validation datasets
    val_dataset = concatenate_datasets(val_datasets)

    assert q_max_len == len(train_dataset[0][0])
    a_vocab_size = len(train_dataset[1][0])
    assert c_max_len == len(train_dataset[2][0])

    date = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
    model_id = "RN-s_hidden-{}-q_hidden-{}-context_size-{}-lr-{}-batch_size-{}-{}".format(
        args.s_hidden_size,
        args.q_hidden_size,
        c_max_len,
        args.learning_rate,
        args.batch_size,
        date,
    )

    child = args.train_data_path.split('/')[-1]
    save_dir = os.path.join(args.save_path, child, model_id)
    save_summary_path = save_dir
    save_variable_path = os.path.join(save_dir, 'model')

    if not os.path.exists(save_dir):
        mkdir_p(os.path.join(args.save_path))
        mkdir_p(os.path.join(args.save_path, child))
        mkdir_p(os.path.join(args.save_path, child, model_id))

    config = {
        'batch_size': args.batch_size,
        's_hidden': args.s_hidden_size,
        'q_hidden': args.q_hidden_size,
        'context_vocab_size': c_vocab_size,
        'question_vocab_size': q_vocab_size,
        'answer_vocab_size': a_vocab_size,
        'c_max_len': c_max_len,
        'q_max_len': q_max_len,
        's_max_len': s_max_len,
        'iter_time': int(args.iter_time),
        'iter_time': int(args.display_step),
    }

    with tf.Graph().as_default():

        sess = tf.Session()
        start_time = time()
        with sess.as_default():

            rn = Model(config)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)

            if args.max_train_iters > 0:
                global_step = tf.Variable(0, name='global_step', trainable=False)
                opt = tf.train.AdamOptimizer(args.learning_rate)
                optimizer = opt.minimize(rn.loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())

            if args.max_train_iters > 0:
                # Define training procedure

                loss_train = tf.summary.scalar("loss_train", rn.loss)
                accuracy_train = tf.summary.scalar("accuracy_train", rn.accuracy)
                train_summary_ops = tf.summary.merge([loss_train, accuracy_train])

                loss_val = tf.summary.scalar("loss_val", rn.loss)
                accuracy_val = tf.summary.scalar("accuracy_val", rn.accuracy)
                val_summary_ops = tf.summary.merge([loss_val, accuracy_val])

                summary_writer = tf.summary.FileWriter(save_summary_path, sess.graph)
                batch_train = batch_iter(c=train_dataset[2],
                                         q=train_dataset[0],
                                         l=train_dataset[3],
                                         a=train_dataset[1],
                                         c_real_len=train_dataset[4],
                                         q_real_len=train_dataset[5],
                                         num_epochs=config['iter_time'],
                                         batch_size=config['batch_size'])
                for i, train in enumerate(batch_train):
                    if i > args.max_train_iters:
                        print("Maximum training iterations reached.")
                        break
                    c_batch, q_batch, l_batch, a_batch, \
                        c_real_len_batch, q_real_len_batch = zip(*train)
                    feed_dict = {rn.context: c_batch,
                                 rn.question: q_batch,
                                 rn.label: l_batch,
                                 rn.answer: a_batch,
                                 rn.context_real_len: c_real_len_batch,
                                 rn.question_real_len: q_real_len_batch,
                                 rn.is_training: True}
                    current_step = sess.run(global_step, feed_dict=feed_dict)
                    optimizer.run(feed_dict=feed_dict)
                    train_summary = sess.run(train_summary_ops, feed_dict=feed_dict)
                    summary_writer.add_summary(train_summary, current_step)
                    if current_step % (args.display_step) == 0:
                        print("step: {}".format(current_step))
                        print("====validation start====")
                        batch_val = batch_iter(c=val_dataset[2],
                                               q=val_dataset[0],
                                               l=val_dataset[3],
                                               a=val_dataset[1],
                                               c_real_len=val_dataset[4],
                                               q_real_len=val_dataset[5],
                                               num_epochs=1,
                                               batch_size=args.batch_size)
                        accs = []
                        for val in batch_val:
                            c_val, q_val, l_val, a_val, \
                                c_real_len_val, q_real_len_val = zip(*val)
                            feed_dict = {rn.context: c_val,
                                         rn.question: q_val,
                                         rn.label: l_val,
                                         rn.answer: a_val,
                                         rn.context_real_len: c_real_len_val,
                                         rn.question_real_len: q_real_len_val,
                                         rn.is_training: False}
                            acc = rn.accuracy.eval(feed_dict=feed_dict)
                            accs.append(acc)
                            val_summary = sess.run(val_summary_ops, feed_dict=feed_dict)
                            summary_writer.add_summary(val_summary, current_step)
                        print("Mean accuracy=" + str(sum(accs) / len(accs)))
                        saver.save(sess, save_path=save_variable_path, global_step=current_step)
                        print("====training====")
                end_time = time()
                print("Training finished in {}sec".format(end_time-start_time))

            config['batch_size'] = 1
            tf.get_variable_scope().reuse_variables()
            rn = Model(config)

            if args.model_path is not None:
                saver.restore(sess, tf.train.latest_checkpoint(args.model_path))

            mean_accs = []
            for test_dataset in test_datasets:
                test_dataset_name = test_dataset[6]
                batch_test = batch_iter(c=test_dataset[2],
                                        q=test_dataset[0],
                                        l=test_dataset[3],
                                        a=test_dataset[1],
                                        c_real_len=test_dataset[4],
                                        q_real_len=test_dataset[5],
                                        num_epochs=1,
                                        batch_size=1,  # for testing
                                        )

                accs = []
                for test in batch_test:
                    c_test, q_test, l_test, a_test, \
                        c_real_len_test, q_real_len_test = zip(*test)
                    feed_dict = {rn.context: c_test,
                                 rn.question: q_test,
                                 rn.label: l_test,
                                 rn.answer: a_test,
                                 rn.context_real_len: c_real_len_test,
                                 rn.question_real_len: q_real_len_test,
                                 rn.is_training: False}
                    acc = rn.accuracy.eval(feed_dict=feed_dict)
                    accs.append(acc)
                mean_acc = np.mean(accs)
                print("Test dataset: {}".format(test_dataset_name))
                print("Mean accuracy= {}".format(mean_acc))
                mean_accs += [mean_acc]
            print("Accuracy across test datasets: {}".format(np.mean(mean_accs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path')
    parser.add_argument('--test_data_path')
    parser.add_argument('--save_path',       default='results')
    parser.add_argument('--model_path',      default=None)
    parser.add_argument('--batch_size',      type=int,   nargs='?', const=64)
    parser.add_argument('--q_hidden_size',   type=int,   nargs='?', const=32)
    parser.add_argument('--s_hidden_size',   type=int,   nargs='?', const=32)
    parser.add_argument('--learning_rate',   type=float, nargs='?', const=2e-4)
    parser.add_argument('--iter_time',       type=int,   nargs='?', const=10)
    parser.add_argument('--max_train_iters', type=int,   nargs='?', const=2500)
    parser.add_argument('--display_step',    type=int,   nargs='?', const=100)
    main(parser.parse_args())
