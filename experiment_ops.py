import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import slim
from tensorflow.contrib.estimator import TowerOptimizer

tf.logging.set_verbosity(tf.logging.INFO)


def dense_to_sparse(tensor, token_to_ignore=0):
    indices = tf.where(tf.not_equal(tensor, tf.constant(token_to_ignore, dtype=tensor.dtype)))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def _set_dynamic_batch_size(inputs):
    new_shape = inputs.get_shape().as_list()
    new_shape[0] = -1
    inputs = tf.reshape(inputs, new_shape, name="input_layer")
    return inputs


def _network_fn(features):
    features = _set_dynamic_batch_size(features)
    features = slim.conv2d(features, 16, [3, 3], stride=1, padding='valid')
    features = slim.max_pool2d(features, 2, stride=2, padding='valid')
    features = _reshape_to_rnn_dims(features)
    features = bidirectional_rnn(features, 32)
    return features


def _reshape_to_rnn_dims(inputs):
    batch_size, height, width, num_channels = inputs.get_shape().as_list()
    if batch_size is None:
        batch_size = -1
    time_major_inputs = tf.transpose(inputs, (2, 0, 1, 3))
    reshaped_time_major_inputs = tf.reshape(time_major_inputs,
                                            [width, batch_size, height * num_channels]
                                            )
    batch_major_inputs = tf.transpose(reshaped_time_major_inputs, (1, 0, 2))
    return batch_major_inputs


def bidirectional_rnn(inputs, num_hidden, concat_output=True,
                      scope=None):
    with tf.variable_scope(scope, "bidirectional_rnn", [inputs]):
        cell_fw = rnn.LSTMCell(num_hidden, initializer=slim.xavier_initializer(), activation=tf.nn.tanh)
        cell_bw = rnn.LSTMCell(num_hidden, initializer=slim.xavier_initializer(), activation=tf.nn.tanh)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                     cell_bw,
                                                     inputs,
                                                     dtype=tf.float32)
        if concat_output:
            return tf.concat(outputs, 2)
        return outputs


def _sparse_to_dense(sparse_tensor, name="sparse_to_dense"):
    return tf.sparse_to_dense(tf.to_int32(sparse_tensor.indices),
                              tf.to_int32(sparse_tensor.dense_shape),
                              tf.to_int32(sparse_tensor.values),
                              name=name)


def train(params, features, labels, num_classes, checkpoint_dir,
          batch_size=1, num_epochs=1,
          save_checkpoint_every_n_epochs=1):
    num_steps_per_epoch = len(features) // batch_size
    save_checkpoint_steps = save_checkpoint_every_n_epochs * num_steps_per_epoch
    params['num_classes'] = num_classes
    params['log_step_count_steps'] = num_steps_per_epoch
    training_hooks = []
    estimator = tf.estimator.Estimator(model_fn=_train_model_fn,
                                       params=params,
                                       model_dir=checkpoint_dir,
                                       config=tf.estimator.RunConfig(
                                           save_checkpoints_steps=save_checkpoint_steps,
                                           log_step_count_steps=num_steps_per_epoch,
                                           save_summary_steps=num_steps_per_epoch
                                       ))
    estimator.train(input_fn=_input_fn(features, labels, batch_size),
                    steps=num_epochs * num_steps_per_epoch,
                    hooks=training_hooks)


def predict(params, features, checkpoint_dir):
    estimator = tf.estimator.Estimator(model_fn=_predict_model_fn,
                                       params=params,
                                       model_dir=checkpoint_dir)
    predictions = estimator.predict(input_fn=_input_fn(features, num_epochs=1))
    for i, p in enumerate(predictions):
        print(i, p)


def _input_fn(features, labels=None, batch_size=1, num_epochs=None, shuffle=True):
    if labels:
        labels = np.array(labels, dtype=np.int32)
    return tf.estimator.inputs.numpy_input_fn(
        x={'features': np.array(features)},
        y=labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


def _add_to_summary(name, value):
    tf.summary.scalar(name, value)


def _create_train_op(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = TowerOptimizer(optimizer)
    return slim.learning.create_train_op(loss, optimizer, global_step=tf.train.get_or_create_global_step())


def _create_model_fn(mode, predictions, loss=None, train_op=None,
                     eval_metric_ops=None, training_hooks=None,
                     evaluation_hooks=None, export_outputs=None):
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      training_hooks=training_hooks,
                                      evaluation_hooks=evaluation_hooks,
                                      export_outputs=export_outputs)


def convert_to_ctc_dims(inputs, num_classes, num_steps, num_outputs):
    outputs = tf.reshape(inputs, [-1, num_outputs])
    logits = slim.fully_connected(outputs, num_classes,
                                  weights_initializer=slim.xavier_initializer())
    logits = slim.fully_connected(logits, num_classes,
                                  weights_initializer=slim.xavier_initializer())
    logits = tf.reshape(logits, [num_steps, -1, num_classes])
    return logits


def _get_output(inputs, num_classes):
    inputs = convert_to_ctc_dims(inputs,
                                         num_classes=num_classes,
                                         num_steps=inputs.shape[1],
                                         num_outputs=inputs.shape[-1])
    decoded, _ = tf.nn.ctc_beam_search_decoder(inputs, _get_sequence_lengths(inputs))
    return _sparse_to_dense(decoded[0], name="output")


def _get_sequence_lengths(inputs):
    dims = tf.shape(inputs)[1]
    sequence_length = tf.fill([dims], inputs.shape[0])
    return sequence_length


def _get_metrics(y_pred, y_true, num_classes):
    metrics_dict = {}
    y_pred = convert_to_ctc_dims(y_pred,
                                 num_classes=num_classes,
                                 num_steps=y_pred.shape[1],
                                 num_outputs=y_pred.shape[-1])
    y_pred, _ = tf.nn.ctc_beam_search_decoder(y_pred, _get_sequence_lengths(y_pred))
    y_true = dense_to_sparse(y_true, token_to_ignore=-1)
    value = label_error_rate(y_pred[0], y_true)
    metrics_dict["label_error_rate"] = value
    return metrics_dict


def label_error_rate(y_pred, y_true, name="label_error_rate"):
    return tf.reduce_mean(tf.edit_distance(tf.cast(y_pred, tf.int32), y_true), name=name)


def _predict_model_fn(features, mode, params):
    features = _network_fn(features)
    outputs = _get_output(features, params["num_classes"])
    predictions = {
        "outputs": outputs
    }

    return _create_model_fn(mode, predictions=predictions,
                            export_outputs={
                                "outputs": tf.estimator.export.PredictOutput(predictions)
                            })


def _train_model_fn(features, labels, mode, params):
    loss, metrics, predictions = _get_evaluation_parameters(features, labels, params)

    train_op = _create_train_op(loss,
                                learning_rate=params["learning_rate"])

    training_hooks = []
    for metric_key in metrics:
        _add_to_summary(metric_key, metrics[metric_key])
        training_hooks.append(tf.train.LoggingTensorHook(
            {metric_key: metric_key},
            every_n_iter=params["log_step_count_steps"])
        )
    return _create_model_fn(mode,
                            predictions=predictions,
                            loss=loss,
                            train_op=train_op,
                            training_hooks=training_hooks)


def _get_evaluation_parameters(features, labels, params):
    features, predictions = _get_fed_features_and_resulting_predictions(features, params)
    loss = _get_loss(labels=labels, inputs=features, num_classes=params["num_classes"])
    metrics = _get_metrics(y_pred=features,
                           y_true=labels,
                           num_classes=params["num_classes"])
    return loss, metrics, predictions


def _get_loss(labels, inputs, num_classes):
    inputs = convert_to_ctc_dims(inputs,
                                         num_classes=num_classes,
                                         num_steps=inputs.shape[1],
                                         num_outputs=inputs.shape[-1])
    labels = dense_to_sparse(labels, token_to_ignore=-1)
    return tf.reduce_mean(tf.nn.ctc_loss(labels, inputs, _get_sequence_lengths(inputs),
                                         preprocess_collapse_repeated=False,
                                         ctc_merge_repeated=True,
                                         time_major=True))


def _get_fed_features_and_resulting_predictions(features, params):
    features = features['features']
    features = _network_fn(features)
    outputs = _get_output(features, params["num_classes"])
    predictions = {
        "outputs": outputs
    }
    return features, predictions
