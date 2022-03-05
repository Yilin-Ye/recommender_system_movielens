import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
from server import PS
from sklearn.metrics import roc_auc_score

#batch = 32
#embedding_dim = 8
#learning_rate = 0.1

def mf_fn(inputs, is_test):
    # Extract the feature and it's target value， feature： user_id和 movie_id
    embed_layer =  inputs['feature_embedding']  #[batch , 2 , embedding_dim]
    embed_layer = tf.reshape(embed_layer, shape=[-1, 2, embedding_dim])
    label = inputs['label'] #[batch , 1]

    # Split the data， Get user_id's embedding and movie_id's embedding
    embed_layer = tf.split(embed_layer, num_or_size_splits=2, axis=1)   #[batch , embedding * 2]
    user_id_embedding = tf.reshape(embed_layer[0], shape=[-1, embedding_dim]) # [batch, embedding_dim]
    movie_id_embedding = tf.reshape(embed_layer[1], shape=[-1, embedding_dim]) #[batch , embedding_dim]


    # Matrix factorization user*item
    out_ = tf.reduce_mean(
        user_id_embedding * movie_id_embedding, axis=1
    )

    # Estimation
    out_tmp = tf.sigmoid(out_)

    if is_test:
        tf.compat.v1.add_to_collections('input_tensor', embed_layer)
        tf.compat.v1.add_to_collections('output_tensor', out_tmp)
        
    # loss function
    label_ = tf.reshape(label, [-1])  #[batch]
    loss_ = tf.reduce_sum(tf.square(label_ - out_))

    out_dic = {
        'loss' : loss_,
        'ground_truth' : label_,
        'prediction' : out_
    }

    return out_dic


# Define the whole structure

def setup_graph(inputs, is_test = False):
    result = {}
    with tf.compat.v1.variable_scope('net_graph', reuse=is_test):
        # Initialize
        net_out_dic = mf_fn(inputs, is_test)
        loss = net_out_dic['loss']
        result['out'] = net_out_dic
        if is_test:
            return result

        #sgd
        emb_grad = tf.gradients(
            loss, [inputs['feature_embedding']], name='feature_embedding')[0]

        result['feature_new_embedding'] = inputs['feature_embedding'] - learning_rate * emb_grad

        result['feature_embedding'] = inputs['feature_embedding']
        result['feature'] = inputs['feature']

        return result


class AUCUtil(object):
    def __init__(self):
        self.reset()

    def add(self, loss, g=np.array([]), p=np.array([])):

        self.loss.append(loss)
        self.ground_truth += g.flatten().tolist()
        self.prediction += p.flatten().tolist()



    def calc(self):
        return {
            'loss_num': len(self.loss),
            'loss': np.array(self.loss).mean(),
            'auc_sum': len(self.ground_truth),
            'auc': roc_auc_score(self.ground_truth, self.prediction) if len(self.ground_truth) > 0 else 0,
            'pcoc': sum(self.prediction) / sum(self.ground_truth)


        }

    def calc_str(self):
        res = self.calc()
        return 'loss: %f(%d), auc: %f(%d), pcoc: %f' % (
            res['loss'], res['loss_num'],
            res['auc'], res['auc_sum'],
            res['pcoc']
        )

    def reset(self):
        self.loss = []
        self.ground_truth = []
        self.prediction = []


from server import InputFn

batch = 32
embedding_dim = 8
local_ps = PS(embedding_dim)
n_parse_threads = 4
shuffle_buffer_size = 1024
prefetch_buffer_size = 16
max_steps = 100000
test_show_step = 1000
learning_rate = 0.15
# Input the value
inputs = InputFn(local_ps)

last_test_auc = 0.

# start to train
train_metric = AUCUtil()
test_metric = AUCUtil()

train_file = 'data/train'
test_file = 'data/test'

# model file
saved_embedding = 'data/saved_embedding'


train_iter, train_inputs = inputs.input_fn(train_file, is_test=False)
train_dic = setup_graph(train_inputs, is_test=False)

test_iter, test_inputs = inputs.input_fn(test_file, is_test=True)
test_dic = setup_graph(test_inputs, is_test=True)

train_log_iter = 1000
last_test_auc = 0.5


def train():
    _step = 0
    print('#' * 80)
    with tf.compat.v1.Session() as sess:
        # Initialize parameter
        sess.run([tf.compat.v1.global_variables_initializer(),
                 tf.compat.v1.local_variables_initializer()])

        # start train
        sess.run(train_iter.initializer)
        while _step < max_steps:
            feature_old_embedding, feature_new_embedding, keys, out= sess.run(
                [train_dic['feature_embedding'],
                 train_dic['feature_new_embedding'],
                 train_dic['feature'],
                 train_dic['out']]
            )

            train_metric.add(
                out['loss'],
                out['ground_truth'],
                out['prediction']
            )

            local_ps.push(keys, feature_new_embedding)
            _step += 1
            #
            if _step % train_log_iter == 0:
                print('Train at step %d: %s', _step, train_metric.calc_str())
                train_metric.reset()

            if _step % test_show_step == 0:
                valid_step(sess, test_iter, test_dic)



def valid_step(sess, test_iter, test_dic):
    test_metric.reset()
    sess.run(test_iter.initializer)
    global last_test_auc
    while True:
        try:
            out = sess.run(test_dic['out'])
            test_metric.add(
                out['loss'],
                out['ground_truth'],
                out['prediction']
            )
        except tf.errors.OutOfRangeError:
            print('Test at step: %s,', test_metric.calc_str())
            if test_metric.calc()['auc'] > last_test_auc:
                last_test_auc = test_metric.calc()['auc']
                local_ps.save(saved_embedding)

            break


if __name__ == '__main__':
    train()