import tensorflow as tf
from tensorflow.keras import layers

def process_images(images):
    images = tf.cast(images, tf.float32)  # Convert to float32
    images = images / 255.0
    images = images - 0.5
    images = images * 2.0
    return images

def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 create_logits=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 scope="InceptionV3"):

    # Only consider the inception model to be in training mode if it's trainable.
    is_inception_model_training = trainable and is_training

    if use_batch_norm:
        # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "training": is_inception_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "momentum": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "moving_mean_initializer": "zeros",
                "moving_variance_initializer": "ones"
            }
    else:
        batch_norm_params = None

    if trainable:
        weights_regularizer = tf.keras.regularizers.l2(weight_decay)
    else:
        weights_regularizer = None

    with tf.name_scope(scope):
        net = images

        net = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                     kernel_regularizer=weights_regularizer, trainable=trainable, name='conv1')(net)
        net = tf.keras.layers.BatchNormalization(**batch_norm_params)(net)
        net = tf.nn.relu(net)
        net = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(net)

        # Add more layers following the Inception V3 architecture

        if create_logits:
            net = tf.keras.layers.GlobalAveragePooling2D()(net)
            net = tf.keras.layers.Dropout(rate=1.0 - dropout_keep_prob)(net)
            net = tf.keras.layers.Dense(units=1000, activation='softmax', name='logits')(net)

    return net
