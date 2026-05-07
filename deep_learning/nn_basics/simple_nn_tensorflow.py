import tensorflow as tf
from tensorflow.keras import layers, models, _initializers

def build_model(n_in: int = 2, n_h: int = 16, n_out: int = 1) -> tf.keras.Model:
    return models.Sequential([
        layers.Input(shape=(n_in,)),
        layers.Dense(
            n_h, activation="relu",
            kernel_initializer=initializers.HeNormal(),
        ),
        layers.Dense(n_out, activation=None),
    ])

def train_tf(X_train, y_train, X_test, y_test,
            hidden: int = 16, epochs: int = 1000, lr: int = 0.1,
            batch_size: int = 64, verbose: int = 0):
    tf.random.set_seed(0)
    model = build_model(n_in=X_train.shape[1], n_h=hidden, n_out=1)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)],
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size, verbose=verbose,
    )
    return model, history.history["loss"]