import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import slim
from tensorflow.contrib.estimator import TowerOptimizer


def _reshape_to_rnn_dims(inputs):
    batch_size, height, width, num_channels = inputs.get_shape().as_list()
    if batch_size is None:
        batch_size = -1
    transposed_inputs = tf.transpose(inputs, (0, 2, 1, 3))
    return tf.reshape(transposed_inputs, [batch_size, width, height * num_channels])


def bidirectional_rnn(features, num_hidden, concat_output=True, scope=None):
    with tf.variable_scope(scope, "bidirectional_rnn", [features]):
        cell_fw = rnn.BasicLSTMCell(num_hidden)
        cell_bw = rnn.BasicLSTMCell(num_hidden)
        outputs = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                  cell_bw,
                                                  features,
                                                  dtype=features.dtype)
        if concat_output:
            return tf.concat(outputs, 2)
        return outputs


def _network_fn(features):
    dynamic_input_shape = _get_dynamic_input_shape(features)
    features = tf.reshape(features, dynamic_input_shape, name="reshape_input_layer")
    features = slim.conv2d(features, 16, [3, 3], stride=1, padding='VALID')
    features = slim.max_pool2d(features, 2, stride=2, padding='VALID')
    features = _reshape_to_rnn_dims(features)
    features = bidirectional_rnn(features, 32)
    return features


def _get_dynamic_input_shape(features):
    input_shape = features.get_shape().as_list()
    input_shape[0] = -1
    return input_shape


def _convert_to_ctc_dims(features, num_classes, num_steps, num_outputs):
    outputs = tf.reshape(features, [-1, num_outputs])
    logits = slim.fully_connected(outputs, num_classes,
                                  weights_initializer=slim.xavier_initializer())
    logits = tf.reshape(logits, [num_steps, -1, num_classes])
    return logits


def _get_sequence_length(features):
    dims = tf.shape(features)[1]
    sequence_length = tf.fill([dims], features.shape[0])
    return sequence_length


def _get_decoded_outputs(features, num_classes):
    features = _convert_to_ctc_dims(features,
                                    num_classes,
                                    num_steps=features.shape[1],
                                    num_outputs=features.shape[-1])
    decoded, _ = tf.nn.ctc_beam_search_decoder(features, _get_sequence_length(features), merge_repeated=True)
    return tf.sparse_to_dense(tf.to_int32(decoded[0].indices),
                              tf.to_int32(decoded[0].values),
                              tf.to_int32(decoded[0].dense_shape),
                              name="output")


def _dense_to_sparse(tensor, token_to_ignore=0):
    indices = tf.where(tf.not_equal(tensor, tf.constant(token_to_ignore, dtype=tensor.dtype)))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def _ctc_loss(labels, features, num_classes):
    features = _convert_to_ctc_dims(features,
                                    num_classes,
                                    num_steps=features.shape[1],
                                    num_outputs=features.shape[-1])
    labels = _dense_to_sparse(labels, token_to_ignore=-1)
    return tf.reduce_mean(tf.nn.ctc_loss(labels, features, _get_sequence_length(features),
                                         preprocess_collapse_repeated=False,
                                         ctc_merge_repeated=True,
                                         time_major=True))


def _create_train_op(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(loss, learning_rate)
    optimizer = TowerOptimizer(optimizer)
    return slim.learning.create_train_op(loss, optimizer, global_step=tf.train.get_or_create_global_step())


def _create_model_fn(mode, predictions, loss=None, train_op=None, export_outputs=None):
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      export_outputs=export_outputs)


def _train_model_fn(features, labels, mode, params):
    features, outputs = _get_fed_features_and_resulting_predictions(features, params['num_classes'])
    predictions = {"outputs": outputs}

    loss = _ctc_loss(labels, features, params['num_classes'])
    train_op = _create_train_op(loss, learning_rate=params['learning_rate'])

    return _create_model_fn(mode,
                            predictions,
                            loss,
                            train_op)


def _get_fed_features_and_resulting_predictions(features, num_classes):
    features = _network_fn(features)
    outputs = _get_decoded_outputs(features, num_classes)
    return features, outputs


def _predict_model_fn(features, mode, params):
    features, outputs = _get_fed_features_and_resulting_predictions(features, params['num_classes'])
    predictions = {"outputs": outputs}

    return _create_model_fn(mode, predictions,
                            export_outputs={
                                "outputs": tf.estimator.export.PredictOutput(outputs)
                            })


def _input_fn(features, labels=None, batch_size=1, num_epochs=None, shuffle=True):
    if labels:
        labels = np.array(labels, dtype=np.int32)
    return tf.estimator.inputs.numpy_input_fn(
        x=np.array(features),
        y=labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


def train(features, labels, num_classes, params, checkpoint_dir,
          batch_size=1, num_epochs=1, save_checkpoint_every_n_epochs=1):
    num_steps_per_epoch = len(features) // batch_size
    save_checkpoint_steps = save_checkpoint_every_n_epochs * num_steps_per_epoch
    params['num_classes'] = num_classes
    params['log_step_count_steps'] = num_steps_per_epoch
    estimator = tf.estimator.Estimator(model_fn=_train_model_fn,
                                       params=params,
                                       model_dir=checkpoint_dir,
                                       config=tf.estimator.RunConfig(
                                           save_checkpoints_steps=save_checkpoint_steps,
                                           log_step_count_steps=num_steps_per_epoch,
                                           save_summary_steps=num_steps_per_epoch
                                       ))
    estimator.train(input_fn=_input_fn(features, labels, batch_size),
                    steps=num_epochs * num_steps_per_epoch)


def predict(features, params, checkpoint_dir):
    estimator = tf.estimator.Estimator(model_fn=_predict_model_fn,
                                       params=params,
                                       model_dir=checkpoint_dir)
    predictions = estimator.predict(input_fn=_input_fn(features))
    for i, p in enumerate(predictions):
        print(i, p)
