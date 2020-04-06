
# coding: utf-8

# In[ ]:

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.layers import  normalization as normalization_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import variables as tf_variables
DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


def batch_norm(x, phase_train, scope='batch_norm'):
    # return tf.contrib.layers.batch_norm(x,center=True,scale=True,
    #                                     is_training=phase_train,
    #                                     scope=tf.get_variable_scope())

    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affine:      whether to affine-transform outputs
    Return:
        normed:      batch-normalized maps
    From: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177?noredirect=1#comment55758348_33950177
    """


    x_shape = tf.Tensor.get_shape(x)
    n_out = x_shape.as_list()[3]
    with tf.device(None):
        with tf.variable_scope(scope) as cursc:
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                name='gamma', trainable=True)

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                ema = tf.train.ExponentialMovingAverage(decay=0.9)
                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                mean, var = tf.cond(phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)),
                                    name='ema')
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
# Below is buggy - sigh.. MK Aug 9 2016            
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
#             def mean_var_with_update():
#                 with tf.control_dependencies([ema_apply_op]):
#                     return tf.identity(batch_mean), tf.identity(batch_var)
#             mean, var = control_flow_ops.cond(tf.constant(phase_train),
#                 mean_var_with_update,
#                 lambda: (ema_mean, ema_var))

#     #        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
#     #            beta, gamma, 1e-3, affine)
#     #       above is is deprecated
#             normed = tf.nn.batch_normalization(x, mean, var, 
#              beta, gamma, 1e-3)
#     return normed

def batch_norm_2D(x, phase_train, scope='batch_norm'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 2D BF input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affine:      whether to affine-transform outputs
    Return:
        normed:      batch-normalized maps
    From: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177?noredirect=1#comment55758348_33950177
    """
    x_shape = tf.Tensor.get_shape(x)
    n_out = x_shape.as_list()[1]
    with tf.device(None):
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                name='gamma', trainable=True)

            batch_mean, batch_var = tf.nn.moments(x, [0,], name='moments')

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                ema = tf.train.ExponentialMovingAverage(decay=0.9)
                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                mean, var = tf.cond(phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed            

# why did this ever exist?    
# def batch_norm_2D(x, phase_train, scope='batch_norm'):
#     return x

def _add_variable_to_collections(variable, collections_set, collections_name):
  """Adds variable (or all its parts) to all collections with that name."""
  collections = utils.get_variable_collections(
      collections_set, collections_name) or []
  variables_list = [variable]
  if isinstance(variable, tf_variables.PartitionedVariable):
    variables_list = [v for v in variable]
  for collection in collections:
    for var in variables_list:
      if var not in ops.get_collection(collection):
        ops.add_to_collection(collection, var)

def _build_variable_getter(rename=None):
  """Build a model variable getter that respects scope getter and renames."""
  # VariableScope will nest the getters
  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)
  return layer_variable_getter

def _model_variable_getter(getter, name, shape=None, dtype=None,
                           initializer=None, regularizer=None, trainable=True,
                           collections=None, caching_device=None,
                           partitioner=None, rename=None, use_resource=None,
                           **_):
  """Getter that uses model_variable for compatibility with core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return variables.model_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, collections=collections, trainable=trainable,
      caching_device=caching_device, partitioner=partitioner,
      custom_getter=getter, use_resource=use_resource)


def batch_norm_mine(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=False,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99):
  """
    The earlier version of my modification to batch norm used
current_mean and current_variance if is_training is True and
moving_mean and moving_variance otherwise. This led to a large divergence between
the results depending upon whether the is_training was set to True or not.

This version always uses moving_mean and moving_variance.

  Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
copy of tensorflow.contrib.layers
  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance. Try zero_debias_moving_mean=True for improved stability.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    param_regularizers: Optional regularizer for beta and gamma.
    updates_collections: Collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    batch_weights: An optional tensor of shape `[batch_size]`,
      containing a frequency weight for each batch item. If present,
      then the batch normalization uses weighted mean and
      variance. (This can be used to correct for bias in training
      example selection.)
    fused:  Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
      pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
    scope: Optional scope for `variable_scope`.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_decay: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `decay` is still applied
      to get the means and variances for inference.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If `batch_weights` is not None and `fused` is True.
    ValueError: If `param_regularizers` is not None and `fused` is True.
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  if fused:
    if batch_weights is not None:
      raise ValueError('Weighted mean and variance is not currently '
                       'supported for fused batch norm.')
    if param_regularizers is not None:
      raise ValueError('Regularizers are not currently '
                       'supported for fused batch norm.')
    if renorm:
      raise ValueError('Renorm is not supported for fused batch norm.')
    return _fused_batch_norm(
        inputs,
        decay=decay,
        center=center,
        scale=scale,
        epsilon=epsilon,
        activation_fn=activation_fn,
        param_initializers=param_initializers,
        updates_collections=updates_collections,
        is_training=is_training,
        reuse=reuse,
        variables_collections=variables_collections,
        outputs_collections=outputs_collections,
        trainable=trainable,
        data_format=data_format,
        zero_debias_moving_mean=zero_debias_moving_mean,
        scope=scope)

  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  layer_variable_getter = _build_variable_getter()
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    # Determine whether we can use the core layer class.
    if (batch_weights is None and
        updates_collections is ops.GraphKeys.UPDATE_OPS and
        not zero_debias_moving_mean and False):
        # !!!!!!!!!!!!!!!!!!!!!!!!
        # This is different than earlier batch_norm
        # !!!!!!!!!!!!!!!!!!!!!!!!
        # Making sure that this layer is never used.

        # Use the core layer class.
      axis = 1 if data_format == DATA_FORMAT_NCHW else -1
      if not param_initializers:
        param_initializers = {}
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      if not param_regularizers:
        param_regularizers = {}
      beta_regularizer = param_regularizers.get('beta')
      gamma_regularizer = param_regularizers.get('gamma')
      layer = normalization_layers.BatchNormalization(
          axis=axis,
          momentum=decay,
          epsilon=epsilon,
          center=center,
          scale=scale,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer,
          moving_mean_initializer=moving_mean_initializer,
          moving_variance_initializer=moving_variance_initializer,
          beta_regularizer=beta_regularizer,
          gamma_regularizer=gamma_regularizer,
          trainable=trainable,
          renorm=renorm,
          renorm_clipping=renorm_clipping,
          renorm_momentum=renorm_decay,
          name=sc.name,
          _scope=sc,
          _reuse=reuse)
      outputs = layer.apply(inputs, training=is_training)

      # Add variables to collections.
      _add_variable_to_collections(
          layer.moving_mean, variables_collections, 'moving_mean')
      _add_variable_to_collections(
          layer.moving_variance, variables_collections, 'moving_variance')
      if layer.beta:
        _add_variable_to_collections(layer.beta, variables_collections, 'beta')
      if layer.gamma:
        _add_variable_to_collections(
            layer.gamma, variables_collections, 'gamma')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return utils.collect_named_outputs(outputs_collections,
                                         sc.original_name_scope, outputs)

    # Not supported by layer class: batch_weights argument,
    # and custom updates_collections. In that case, use the legacy BN
    # implementation.
    # Custom updates collections are not supported because the update logic
    # is different in this case, in particular w.r.t. "forced updates" and
    # update op reuse.
    if renorm:
      raise ValueError('renorm is not supported with batch_weights, '
                       'updates_collections or zero_debias_moving_mean')
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if batch_weights is not None:
      batch_weights = ops.convert_to_tensor(batch_weights)
      inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
      # Reshape batch weight values so they broadcast across inputs.
      nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
      batch_weights = array_ops.reshape(batch_weights, nshape)

    if data_format == DATA_FORMAT_NCHW:
      moments_axes = [0] + list(range(2, inputs_rank))
      params_shape = inputs_shape[1:2]
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
    else:
      moments_axes = list(range(inputs_rank - 1))
      params_shape = inputs_shape[-1:]
      params_shape_broadcast = None
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if not param_initializers:
      param_initializers = {}
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=beta_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=gamma_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)

    # Create moving_mean and moving_variance variables and add them to the
    # appropriate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables.
    partitioner = variable_scope.get_variable_scope().partitioner
    try:
      variable_scope.get_variable_scope().set_partitioner(None)
      moving_mean_collections = utils.get_variable_collections(
          variables_collections, 'moving_mean')
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_mean = variables.model_variable(
          'moving_mean',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_mean_initializer,
          trainable=False,
          collections=moving_mean_collections)
      moving_variance_collections = utils.get_variable_collections(
          variables_collections, 'moving_variance')
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      moving_variance = variables.model_variable(
          'moving_variance',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_variance_initializer,
          trainable=False,
          collections=moving_variance_collections)
    finally:
      variable_scope.get_variable_scope().set_partitioner(partitioner)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
      # Calculate the moments based on the individual batch.
      if batch_weights is None:
        if data_format == DATA_FORMAT_NCHW:
          mean, _ = nn.moments(inputs, moments_axes, keep_dims=True)
          variance,_ = nn.moments( (inputs-moving_mean)**2, moments_axes, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, _ = nn.moments(inputs, moments_axes)
          variance, _ = nn.moments( (inputs-moving_mean)**2, moments_axes)
      else:
        if data_format == DATA_FORMAT_NCHW:
          mean, _ = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights, keep_dims=True)
          variance, _ = nn.weighted_moments( (inputs-moving_mean)**2, moments_axes,
                                               batch_weights, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, _ = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights)
          variance, _ = nn.weighted_moments( (inputs-moving_mean)**2, moments_axes,
                                               batch_weights)

      moving_vars_fn = lambda: (moving_mean, moving_variance)
      if updates_collections is None:
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          with ops.control_dependencies([update_moving_mean,
                                         update_moving_variance]):
            # !!!!!!!!!!!!!!!!!!!!!!!!
            # This is different than earlier batch_norm
            # !!!!!!!!!!!!!!!!!!!!!!!!
            # return array_ops.identity(mean), array_ops.identity(variance)
            return array_ops.identity(moving_mean), array_ops.identity(moving_variance)
        mean, variance = utils.smart_cond(is_training,
                                          _force_updates,
                                          moving_vars_fn)
      else:
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          return update_moving_mean, update_moving_variance

        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)
        # Use computed moments during training and moving_vars otherwise.
        # !!!!!!!!!!!!!!!!!!!!!!!!
        # This is different than earlier batch_norm
        # !!!!!!!!!!!!!!!!!!!!!!!!
        # vars_fn = lambda: (mean, variance)
        vars_fn = lambda: (moving_mean, moving_variance)
        mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
    else:
      mean, variance = moving_mean, moving_variance
    if data_format == DATA_FORMAT_NCHW:
      mean = array_ops.reshape(mean, params_shape_broadcast)
      variance = array_ops.reshape(variance, params_shape_broadcast)
      beta = array_ops.reshape(beta, params_shape_broadcast)
      if gamma is not None:
        gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Compute batch_normalization.
    outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                     epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)



def batch_norm_mine_old(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=False,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99):
  """
  This earlier version of my modification to batch norm uses
current_mean and current_variance if is_training is True and
moving_mean and moving_variance otherwise. This was leading a large divergence between
the results depending upon whether the is_training set to True or not.

I think ideally it should always use moving_mean and moving_variance. batch_norm_mine
does this.

  Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
copy of tensorflow.contrib.layers
  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance. Try zero_debias_moving_mean=True for improved stability.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    param_regularizers: Optional regularizer for beta and gamma.
    updates_collections: Collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    batch_weights: An optional tensor of shape `[batch_size]`,
      containing a frequency weight for each batch item. If present,
      then the batch normalization uses weighted mean and
      variance. (This can be used to correct for bias in training
      example selection.)
    fused:  Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
      pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
    scope: Optional scope for `variable_scope`.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_decay: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `decay` is still applied
      to get the means and variances for inference.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If `batch_weights` is not None and `fused` is True.
    ValueError: If `param_regularizers` is not None and `fused` is True.
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  if fused:
    if batch_weights is not None:
      raise ValueError('Weighted mean and variance is not currently '
                       'supported for fused batch norm.')
    if param_regularizers is not None:
      raise ValueError('Regularizers are not currently '
                       'supported for fused batch norm.')
    if renorm:
      raise ValueError('Renorm is not supported for fused batch norm.')
    return _fused_batch_norm(
        inputs,
        decay=decay,
        center=center,
        scale=scale,
        epsilon=epsilon,
        activation_fn=activation_fn,
        param_initializers=param_initializers,
        updates_collections=updates_collections,
        is_training=is_training,
        reuse=reuse,
        variables_collections=variables_collections,
        outputs_collections=outputs_collections,
        trainable=trainable,
        data_format=data_format,
        zero_debias_moving_mean=zero_debias_moving_mean,
        scope=scope)

  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  layer_variable_getter = _build_variable_getter()
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    # Determine whether we can use the core layer class.
    if (batch_weights is None and
        updates_collections is ops.GraphKeys.UPDATE_OPS and
        not zero_debias_moving_mean):
      # Use the core layer class.
      axis = 1 if data_format == DATA_FORMAT_NCHW else -1
      if not param_initializers:
        param_initializers = {}
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      if not param_regularizers:
        param_regularizers = {}
      beta_regularizer = param_regularizers.get('beta')
      gamma_regularizer = param_regularizers.get('gamma')
      layer = normalization_layers.BatchNormalization(
          axis=axis,
          momentum=decay,
          epsilon=epsilon,
          center=center,
          scale=scale,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer,
          moving_mean_initializer=moving_mean_initializer,
          moving_variance_initializer=moving_variance_initializer,
          beta_regularizer=beta_regularizer,
          gamma_regularizer=gamma_regularizer,
          trainable=trainable,
          renorm=renorm,
          renorm_clipping=renorm_clipping,
          renorm_momentum=renorm_decay,
          name=sc.name,
          _scope=sc,
          _reuse=reuse)
      outputs = layer.apply(inputs, training=is_training)

      # Add variables to collections.
      _add_variable_to_collections(
          layer.moving_mean, variables_collections, 'moving_mean')
      _add_variable_to_collections(
          layer.moving_variance, variables_collections, 'moving_variance')
      if layer.beta:
        _add_variable_to_collections(layer.beta, variables_collections, 'beta')
      if layer.gamma:
        _add_variable_to_collections(
            layer.gamma, variables_collections, 'gamma')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return utils.collect_named_outputs(outputs_collections,
                                         sc.original_name_scope, outputs)

    # Not supported by layer class: batch_weights argument,
    # and custom updates_collections. In that case, use the legacy BN
    # implementation.
    # Custom updates collections are not supported because the update logic
    # is different in this case, in particular w.r.t. "forced updates" and
    # update op reuse.
    if renorm:
      raise ValueError('renorm is not supported with batch_weights, '
                       'updates_collections or zero_debias_moving_mean')
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if batch_weights is not None:
      batch_weights = ops.convert_to_tensor(batch_weights)
      inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
      # Reshape batch weight values so they broadcast across inputs.
      nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
      batch_weights = array_ops.reshape(batch_weights, nshape)

    if data_format == DATA_FORMAT_NCHW:
      moments_axes = [0] + list(range(2, inputs_rank))
      params_shape = inputs_shape[1:2]
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
    else:
      moments_axes = list(range(inputs_rank - 1))
      params_shape = inputs_shape[-1:]
      params_shape_broadcast = None
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if not param_initializers:
      param_initializers = {}
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=beta_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=gamma_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)

    # Create moving_mean and moving_variance variables and add them to the
    # appropriate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables.
    partitioner = variable_scope.get_variable_scope().partitioner
    try:
      variable_scope.get_variable_scope().set_partitioner(None)
      moving_mean_collections = utils.get_variable_collections(
          variables_collections, 'moving_mean')
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_mean = variables.model_variable(
          'moving_mean',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_mean_initializer,
          trainable=False,
          collections=moving_mean_collections)
      moving_variance_collections = utils.get_variable_collections(
          variables_collections, 'moving_variance')
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      moving_variance = variables.model_variable(
          'moving_variance',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_variance_initializer,
          trainable=False,
          collections=moving_variance_collections)
    finally:
      variable_scope.get_variable_scope().set_partitioner(partitioner)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
      # Calculate the moments based on the individual batch.
      if batch_weights is None:
        if data_format == DATA_FORMAT_NCHW:
          mean, _ = nn.moments(inputs, moments_axes, keep_dims=True)
          variance,_ = nn.moments( (inputs-moving_mean)**2, moments_axes, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, _ = nn.moments(inputs, moments_axes)
          variance, _ = nn.moments( (inputs-moving_mean)**2, moments_axes)
      else:
        if data_format == DATA_FORMAT_NCHW:
          mean, _ = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights, keep_dims=True)
          variance, _ = nn.weighted_moments( (inputs-moving_mean)**2, moments_axes,
                                               batch_weights, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, _ = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights)
          variance, _ = nn.weighted_moments( (inputs-moving_mean)**2, moments_axes,
                                               batch_weights)

      moving_vars_fn = lambda: (moving_mean, moving_variance)
      if updates_collections is None:
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          with ops.control_dependencies([update_moving_mean,
                                         update_moving_variance]):
            return array_ops.identity(mean), array_ops.identity(variance)
        mean, variance = utils.smart_cond(is_training,
                                          _force_updates,
                                          moving_vars_fn)
      else:
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          return update_moving_mean, update_moving_variance

        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)
        # Use computed moments during training and moving_vars otherwise.
        vars_fn = lambda: (mean, variance)
        mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
    else:
      mean, variance = moving_mean, moving_variance
    if data_format == DATA_FORMAT_NCHW:
      mean = array_ops.reshape(mean, params_shape_broadcast)
      variance = array_ops.reshape(variance, params_shape_broadcast)
      beta = array_ops.reshape(beta, params_shape_broadcast)
      if gamma is not None:
        gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Compute batch_normalization.
    outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                     epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)

