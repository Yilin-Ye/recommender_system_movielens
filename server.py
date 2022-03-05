import numpy as np
import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()


# define a singleton class
class Singleton(type):
    _instance = {}
    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instance:
            Singleton._instance[cls] = type.__call__(cls, *args, **kwargs)
        return Singleton._instance[cls]

# Define a parameter server with k-v(key, value) structure map [hashcode, embedding]
class PS(metaclass=Singleton):
    def __init__(self, embedding_dim):
        np.random.seed(2020)
        self.params_server = dict()
        self.dim = embedding_dim
        print('ps init')


    def pull(self, keys):
        values = []
        # data format is  [batch, feature_len]
        # keys = [[123, 234], [567, 891]]
        for k in keys:
            tmp = []
            for arr in k:
                value = self.params_server.get(arr, None)
                if value is None:
                    value = np.random.rand(self.dim)
                    self.params_server[arr] = value
                tmp.append(value)
            values.append(tmp)
        return np.asarray(values, dtype='float32')


    def push(self, keys, values):
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.params_server[keys[i][j]] = values[i][j]

    def delete(self, keys):
        for k in keys:
            self.params_server.pop(k)

    def save(self, path):
        print('Total keys included: ', len(self.params_server))
        writer = open(path, 'w')
        for k, v in self.params_server.items():
            writer.write(str(k) + '\t' + ','.join(['%.8f' % _ for _ in v]) + '\n')
        writer.close()



class InputFn:
    def __init__(self, local_ps):
        self.feature_len = 2
        self.label_len = 1
        self.n_parse_threads = 4
        self.shuffle_buffer_size = 1024
        self.prefetch_buffer_size = 1
        self.batch = 32
        self.local_ps = local_ps

    def input_fn(self, data_dir, is_test=False):
        def _parse_example(example):
            features = {
                'feature': tf.io.FixedLenFeature(self.feature_len, tf.int64),
                'label': tf.io.FixedLenFeature(self.label_len, tf.float32)
            }
            return tf.io.parse_single_example(example, features)

        def _get_embedding(parsed):
            keys = parsed['feature']
            # Extract the feature vector from the corresponding key in parameter server
            keys_array = tf.compat.v1.py_func(self.local_ps.pull, [keys], tf.float32)
            result = {
                'feature': parsed['feature'],   # [batch, 2]
                'label': parsed['label'],  #[batch， 1]
                'feature_embedding': keys_array,    #[batch, 2, embedding_dim]
            }
            return result

        file_list = os.listdir(data_dir)
        files = []
        for i in range(len(file_list)):
            files.append(os.path.join(data_dir, file_list[i]))

        # Read the file and convert it into dataset
        dataset = tf.compat.v1.data.Dataset.list_files(files)

        # Determine how many replication
        if is_test:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.repeat()

        # Read the tfrecord data
        dataset = dataset.interleave(
            lambda _: tf.compat.v1.data.TFRecordDataset(_),
            cycle_length = 1
        )


        # analyze tfrecord data
        dataset = dataset.map(
            _parse_example,
            num_parallel_calls = self.n_parse_threads
        )

        # batch data
        dataset = dataset.batch(
            self.batch, drop_remainder=True
        )

        dataset = dataset.map(
            _get_embedding,
            num_parallel_calls=self.n_parse_threads
        )

        # Shuffle the data
        if not is_test:
            dataset.shuffle(self.shuffle_buffer_size)

        # Prefetch the data
        dataset = dataset.prefetch(
            buffer_size = self.prefetch_buffer_size
        )

        # Iterator
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

        return iterator, iterator.get_next()








