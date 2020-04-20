import tensorflow as tf
def get_LTC_v1_0(inputShape, dropout, nClass, weigh_decay):
    entrada = tf.keras.Input(shape=inputShape, name="input_video")
    #Conv1
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_1')(entrada)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pooling3d_1')(x)

    #Conv2
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_2')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2), name='max_pooling3d_2')(x)

    #Conv3
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_3')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_3')(x)

    #Conv4
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_4')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_4')(x)

    #Conv5
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_5')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_5')(x)

    #fc6s
    x = tf.keras.layers.Flatten(name='flatten_6')(x)
    x = tf.keras.layers.Dense(2048, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                         name='dense_6')(x)
    x = tf.keras.layers.Dropout(rate=dropout,name='dropout_6')(x)

    #fc7
    x = tf.keras.layers.Dense(2048, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                         name='dense_7')(x)
    x = tf.keras.layers.Dropout(rate=dropout,name='dropout_7')(x)

    #fc8
    salidas = tf.keras.layers.Dense(nClass, activation="softmax", 
                              kernel_regularizer=tf.keras.regularizers.l2(weigh_decay), 
                              name='dense_8')(x)

    return tf.keras.Model(entrada, salidas, name="LTC_v1_0")

def get_LTC_v1_1(inputShape, dropout, nClass, weigh_decay):
    entrada = tf.keras.Input(shape=inputShape, name="input_video")
    #Conv1
    x = tf.keras.layers.BatchNormalization(name="batch_norm_1")(entrada)
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_1')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pooling3d_1')(x)

    #Conv2
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_2')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2), name='max_pooling3d_2')(x)

    #Conv3
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_3')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_3')(x)

    #Conv4
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_4')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_4')(x)

    #Conv5
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="valid", activation="relu", 
                          kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                          name='conv3d_5')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),name='max_pooling3d_5')(x)

    #fc6s
    x = tf.keras.layers.Flatten(name='flatten_6')(x)
    x = tf.keras.layers.Dense(2048, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                         name='dense_6')(x)
    x = tf.keras.layers.Dropout(rate=dropout,name='dropout_6')(x)

    #fc7
    x = tf.keras.layers.Dense(2048, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weigh_decay),
                         name='dense_7')(x)
    x = tf.keras.layers.Dropout(rate=dropout,name='dropout_7')(x)

    #fc8
    salidas = tf.keras.layers.Dense(nClass, activation="softmax", 
                              kernel_regularizer=tf.keras.regularizers.l2(weigh_decay), 
                              name='dense_8')(x)

    return tf.keras.Model(entrada, salidas, name="LTC_v1_1")