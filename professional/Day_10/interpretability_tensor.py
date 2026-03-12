import tensorflow as tf
import numpy as np 

X = np.array([
    [25., 50000., 3., 1.],
    [45., 80000., 10., 0.],
    [30., 60000., 5., 1.],
    [50., 90000., 12., 0.]
], dtype=np.float32)

y = np.array([[0.], [1.], [0.], [1.]], dtype=np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics==['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

baseline_loss, baseline_acc = model.evaluate(X, y, verbose=0)
print("Baseline accuracy:", baseline_acc)

feature_names = ['age', 'income', 'experience', 'owns_house']

for j, name in enumerate(feature_names):
    X_perm = X.copy()
    np.random.shuffle(X_perm[:, j])
    _, perm_acc = model.evaluate(X_perm, y, verbose=0)
    importance = baseline_acc - perm_acc
    print(f"{name}: {importance:.4f}")