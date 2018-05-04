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
    features = slim.conv2d(features, 16, [3, 3])
    features = slim.max_pool2d(features, 2)
    features = slim.conv2d(features, 32, [3, 3])
    features = slim.max_pool2d(features, 2)
    features = slim.conv2d(features, 64, [3, 3])
    features = slim.max_pool2d(features, 2)
    features = slim.conv2d(features, 128, [3, 3])
    features = slim.max_pool2d(features, 2)
    features = slim.conv2d(features, 256, [3, 3])
    features = slim.max_pool2d(features, 2)
    features = _reshape_to_rnn_dims(features)
    features = bidirectional_rnn(features, 128)
    return features


def _reshape_to_rnn_dims(inputs):
    batch_size, height, width, num_channels = inputs.get_shape().as_list()
    if batch_size is None:
        batch_size = -1
    nwhc_cnn_outputs = tf.transpose(inputs, (0, 2, 1, 3))
    batch_major_rnn_inputs = tf.reshape(nwhc_cnn_outputs,
                                        [batch_size, width, height * num_channels]
                                        )
    return batch_major_rnn_inputs


def bidirectional_rnn(inputs, num_hidden, concat_output=True,
                      scope=None):
    with tf.variable_scope(scope, "bidirectional_rnn", [inputs]):
        cell_fw = rnn.LSTMCell(num_hidden)
        cell_bw = rnn.LSTMCell(num_hidden)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                     cell_bw,
                                                     inputs,
                                                     dtype=tf.float32)
        if concat_output:
            return tf.concat(outputs, 2)
        return outputs


def mdrnn(inputs, num_hidden, kernel_size=None, scope=None):
    if kernel_size is not None:
        inputs = _get_blocks(inputs, kernel_size)
    with tf.variable_scope(scope, "multidimensional_rnn", [inputs]):
        hidden_sequence_horizontal = _bidirectional_rnn_scan(inputs,
                                                             num_hidden // 2)
        with tf.variable_scope("vertical"):
            transposed = tf.transpose(hidden_sequence_horizontal, [0, 2, 1, 3])
            output_transposed = _bidirectional_rnn_scan(transposed, num_hidden // 2)
        output = tf.transpose(output_transposed, [0, 2, 1, 3])
        return output


def _get_blocks(inputs, kernel_size):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    with tf.variable_scope("image_blocks"):
        batch_size, height, width, channels = _get_shape_as_list(inputs)
        if batch_size is None:
            batch_size = -1

        if height % kernel_size[0] != 0:
            offset = tf.fill([tf.shape(inputs)[0],
                              kernel_size[0] - (height % kernel_size[0]),
                              width,
                              channels], 0.0)
            inputs = tf.concat([inputs, offset], 1)
            _, height, width, channels = _get_shape_as_list(inputs)
        if width % kernel_size[1] != 0:
            offset = tf.fill([tf.shape(inputs)[0],
                              height,
                              kernel_size[1] - (width % kernel_size[1]),
                              channels], 0.0)
            inputs = tf.concat([inputs, offset], 2)
            _, height, width, channels = _get_shape_as_list(inputs)

        h, w = int(height / kernel_size[0]), int(width / kernel_size[1])
        features = kernel_size[1] * kernel_size[0] * channels

        lines = tf.split(inputs, h, axis=1)
        line_blocks = []
        for line in lines:
            line = tf.transpose(line, [0, 2, 3, 1])
            line = tf.reshape(line, [batch_size, w, features])
            line_blocks.append(line)

        return tf.stack(line_blocks, axis=1)


def images_to_sequence(inputs):
    _, _, width, num_channels = _get_shape_as_list(inputs)
    s = tf.shape(inputs)
    batch_size, height = s[0], s[1]
    return tf.reshape(inputs, [batch_size * height, width, num_channels])


def _get_shape_as_list(tensor):
    return tensor.get_shape().as_list()


def sequence_to_images(tensor, height):
    num_batches, width, depth = tensor.get_shape().as_list()
    if num_batches is None:
        num_batches = -1
    else:
        num_batches = num_batches // height
    reshaped = tf.reshape(tensor,
                          [num_batches, width, height, depth])
    return tf.transpose(reshaped, [0, 2, 1, 3])


def _bidirectional_rnn_scan(inputs, num_hidden):
    with tf.variable_scope("BidirectionalRNN", [inputs]):
        height = inputs.get_shape().as_list()[1]
        inputs = images_to_sequence(inputs)
        output_sequence = bidirectional_rnn(inputs, num_hidden)
        output = sequence_to_images(output_sequence, height)
        return output


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
    print(list(predictions))


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
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9)
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


def _convert_to_ctc_dims(inputs, num_classes, num_steps, num_outputs):
    outputs = tf.reshape(inputs, [-1, num_outputs])
    logits = slim.fully_connected(outputs, num_classes)
    logits = tf.reshape(logits, [-1, num_steps, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))
    return logits


def _get_output(inputs, num_classes):
    inputs = _convert_to_ctc_dims(inputs,
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
    y_pred = _convert_to_ctc_dims(y_pred,
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
    _, predictions = _get_fed_features_and_resulting_predictions(features, params)

    return _create_model_fn(mode, predictions=predictions,
                            export_outputs={
                                "outputs": tf.estimator.export.PredictOutput(predictions)
                            })


def _train_model_fn(features, labels, mode, params):
    loss, metrics, predictions = _get_evaluation_parameters(features, labels, params)

    train_op = _create_train_op(loss,
                                learning_rate=params["learning_rate"])

    training_hooks = [tf.train.LoggingTensorHook(
        predictions,
        every_n_iter=params["log_step_count_steps"]),
        tf.train.LoggingTensorHook(
            {"labels": _sparse_to_dense(dense_to_sparse(labels, token_to_ignore=-1))},
            every_n_iter=params["log_step_count_steps"]
        )
    ]
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
    inputs = _convert_to_ctc_dims(inputs,
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
