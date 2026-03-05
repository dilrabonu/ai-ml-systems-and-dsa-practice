import numpy as np
import tensorflow as tf 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    

# Input data
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target.astype(np.int64)

# Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

inputs = tf.keras.Input(shape=(4,))

x = tf.keras.layers.Dense(32, kernel_initializer="he_normal", bias_initializer="zeros")(inputs)

# Normalization inside network
x = tf.keras.layers.BatchNormalization()(x)

# Activation function (non-linearity)
x = tf.keras.layers.ReLU()(x)

# Regularization
x = tf.keras.layers.Dropout(0.2)(x)

# Output layer: logits for 3 classes
logits = tf.keras.layers.Dense(3, kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

model = tf.keras.Model(inputs=inputs, outputs=logits)

# Loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Optimizer 
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2)

# Learning Rate Scheduler
def step_lr_schedule(epoch, lr):
    if epoch > 0 and epoch % 30 == 0:
        return lr * 0.5
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(step_lr_schedule)

optimizer.clipnorm = 1.0

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=120,
    batch_size=16,
    callbacks=[lr_callback],
    verbose=1
)
# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", test_acc)