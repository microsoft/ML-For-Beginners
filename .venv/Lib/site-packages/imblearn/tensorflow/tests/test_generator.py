from distutils.version import LooseVersion

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from imblearn.under_sampling import NearMiss

tf = pytest.importorskip("tensorflow")


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    X, y = make_imbalance(X, y, sampling_strategy={0: 30, 1: 50, 2: 40})
    X = X.astype(np.float32)
    return X, y


def check_balanced_batch_generator_tf_1_X_X(dataset, sampler):
    X, y = dataset
    batch_size = 10
    training_generator, steps_per_epoch = balanced_batch_generator(
        X,
        y,
        sample_weight=None,
        sampler=sampler,
        batch_size=batch_size,
        random_state=42,
    )

    learning_rate = 0.01
    epochs = 10
    input_size = X.shape[1]
    output_size = 3

    # helper functions
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def accuracy(y_true, y_pred):
        return np.mean(np.argmax(y_pred, axis=1) == y_true)

    # input and output
    data = tf.placeholder("float32", shape=[None, input_size])
    targets = tf.placeholder("int32", shape=[None])

    # build the model and weights
    W = init_weights([input_size, output_size])
    b = init_weights([output_size])
    out_act = tf.nn.sigmoid(tf.matmul(data, W) + b)

    # build the loss, predict, and train operator
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=out_act, labels=targets
    )
    loss = tf.reduce_sum(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    predict = tf.nn.softmax(out_act)

    # Initialization of all variables in the graph
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for e in range(epochs):
            for i in range(steps_per_epoch):
                X_batch, y_batch = next(training_generator)
                sess.run(
                    [train_op, loss],
                    feed_dict={data: X_batch, targets: y_batch},
                )

            # For each epoch, run accuracy on train and test
            predicts_train = sess.run(predict, feed_dict={data: X})
            print(f"epoch: {e} train accuracy: {accuracy(y, predicts_train):.3f}")


def check_balanced_batch_generator_tf_2_X_X_compat_1_X_X(dataset, sampler):
    tf.compat.v1.disable_eager_execution()

    X, y = dataset
    batch_size = 10
    training_generator, steps_per_epoch = balanced_batch_generator(
        X,
        y,
        sample_weight=None,
        sampler=sampler,
        batch_size=batch_size,
        random_state=42,
    )

    learning_rate = 0.01
    epochs = 10
    input_size = X.shape[1]
    output_size = 3

    # helper functions
    def init_weights(shape):
        return tf.Variable(tf.random.normal(shape, stddev=0.01))

    def accuracy(y_true, y_pred):
        return np.mean(np.argmax(y_pred, axis=1) == y_true)

    # input and output
    data = tf.compat.v1.placeholder("float32", shape=[None, input_size])
    targets = tf.compat.v1.placeholder("int32", shape=[None])

    # build the model and weights
    W = init_weights([input_size, output_size])
    b = init_weights([output_size])
    out_act = tf.nn.sigmoid(tf.matmul(data, W) + b)

    # build the loss, predict, and train operator
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=out_act, labels=targets
    )
    loss = tf.reduce_sum(input_tensor=cross_entropy)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    predict = tf.nn.softmax(out_act)

    # Initialization of all variables in the graph
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        for e in range(epochs):
            for i in range(steps_per_epoch):
                X_batch, y_batch = next(training_generator)
                sess.run(
                    [train_op, loss],
                    feed_dict={data: X_batch, targets: y_batch},
                )

            # For each epoch, run accuracy on train and test
            predicts_train = sess.run(predict, feed_dict={data: X})
            print(f"epoch: {e} train accuracy: {accuracy(y, predicts_train):.3f}")


@pytest.mark.parametrize("sampler", [None, NearMiss(), RandomOverSampler()])
def test_balanced_batch_generator(data, sampler):
    if LooseVersion(tf.__version__) < "2":
        check_balanced_batch_generator_tf_1_X_X(data, sampler)
    else:
        check_balanced_batch_generator_tf_2_X_X_compat_1_X_X(data, sampler)


@pytest.mark.parametrize("keep_sparse", [True, False])
def test_balanced_batch_generator_function_sparse(data, keep_sparse):
    X, y = data

    training_generator, steps_per_epoch = balanced_batch_generator(
        sparse.csr_matrix(X),
        y,
        keep_sparse=keep_sparse,
        batch_size=10,
        random_state=42,
    )
    for idx in range(steps_per_epoch):
        X_batch, y_batch = next(training_generator)
        if keep_sparse:
            assert sparse.issparse(X_batch)
        else:
            assert not sparse.issparse(X_batch)
