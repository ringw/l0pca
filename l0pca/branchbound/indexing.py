import tensorflow as tf

def index_symmetric(M, d):
    return tf.transpose(
        tf.gather(
            tf.transpose(
                tf.gather(M, d),
            ),
            d,
        )
    )

def index_columns(M, d):
    return tf.transpose(
        tf.gather(
            tf.transpose(M), d,
        )
    )