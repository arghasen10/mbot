import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_cnn2d_dop():
    model2d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2, 5), (1, 2), padding="same", activation='relu', input_shape=(182, 256, 2)),
        tf.keras.layers.Conv2D(64, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3)
    ], name='dopcnn2d')
    return model2d

def get_cnn2d_ang():
    model2d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2, 5), (1, 2), padding="same", activation='relu', input_shape=(64, 256, 2)),
        tf.keras.layers.Conv2D(64, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3)
    ], name='angcnn2d')
    return model2d


def get_cnn1d():
    model1d = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 8, strides=2, padding="valid", activation='relu', input_shape=(256, 2)),
    tf.keras.layers.Conv1D(64, 8, strides=2, padding="valid", activation='relu'),
    tf.keras.layers.Conv1D(96, 4, strides=2, padding="valid", activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(rate=0.3)
    ], name='cnn1d')
    return model1d

def get_feature_shape():
    X_1 = tf.keras.layers.Input(shape=(182, 256, 2))
    X_2 = tf.keras.layers.Input(shape=(64, 256, 2))
    X_3 = tf.keras.layers.Input(shape=(256, 2))
    return X_1, X_2, X_3

def featureExtractor(X_1, X_2, X_3, dopcnn2d, angcnn2d, cnn1d):
    emb1 = dopcnn2d(X_1)
    emb2 = angcnn2d(X_2)
    emb3 = cnn1d(X_3)
    return tf.keras.layers.Concatenate(axis=1)([emb1, emb2, emb3])

def get_vel_regressor():
    ann = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer='l2', input_shape=(352,)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=2)
    ], name='vel_regressor')
    return ann


def connect_feature_embeddings():
    dopcnn2d = get_cnn2d_dop()
    angcnn2d = get_cnn2d_ang()
    cnn1d = get_cnn1d()
    x1, x2, x3 = get_feature_shape()
    vel = get_vel_regressor()
    # Connect
    emb = featureExtractor(x1, x2, x3, dopcnn2d, angcnn2d, cnn1d)
    out_vel = vel(emb)
    return x1, x2, x3, out_vel

def get_fused_cnn_model():
    X_1, X_2, X_3, out_vel = connect_feature_embeddings()
    model = tf.keras.Model(inputs=[X_1, X_2, X_3], outputs=[out_vel], name='Fused_Vel_model')
    print(model.summary())
    return model

def train_cnn(model, X1, X2, X3, Vx, Vy, epochs=3000):
    print(X1.shape, X2.shape, X3.shape)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'accuracy'])
    history = \
        model.fit(
            [X1, X2, X3],
            np.array([Vx, Vy]).T,
            epochs=epochs,
            validation_split=0.2,
            batch_size=10,
        )
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.legend()
    plt.savefig("acc.png")
    plt.cla()
    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['val_loss'], label='val_mae')
    plt.legend()
    plt.savefig("loss.png")
    return model


def test():
    model = get_fused_cnn_model()
    X1 = np.random.rand(31, 182, 256, 2)
    X2 = np.random.rand(31, 64, 256, 2)
    X3 = np.random.rand(31, 256, 2)
    Vx = np.random.rand(31)
    Vy = np.random.rand(31)
    model = train_cnn(model, X1, X2, X3, Vx, Vy, epochs=10)

if __name__=="__main__":
    test()
