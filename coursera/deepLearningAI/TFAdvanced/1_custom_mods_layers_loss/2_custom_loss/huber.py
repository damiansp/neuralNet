def my_huber(y, preds):
    thresh = 1
    err = y - preds
    is_small_err = tf.abs(err) <= thresh
    small_err_loss = tf.square(err) / 2
    big_err_loss = thresh * (tf.abs(err) - (thresh / 2.))
    return tf.where(is_small_err, small_err_loss, big_err_loss)

# in use:
#mod = ...
mod.compile(optimizer='sgd', loss=my_huber)
