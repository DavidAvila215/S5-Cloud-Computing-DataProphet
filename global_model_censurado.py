import os
import tensorflow as tf
import numpy as np
from model import build_model
from sklearn.metrics import classification_report

def load_weights_from_npz(path):
    data = np.load(path)
    weights = [data[key] for key in data]
    return weights


fedavg_weights = load_weights_from_npz("fedavg_weights.npz")
fedmedian_weights = load_weights_from_npz("fedmedian_weights.npz")
fedweighted_weights = load_weights_from_npz("fedweighted_weights.npz")


(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test / 255.0, -1)

for name, weights in [("FedAvg", fedavg_weights), ("FedMedian", fedmedian_weights), ("FedWeightedAvg", fedweighted_weights)]:
    model = build_model()
    model.set_weights(weights)
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\n=== Reporte de clasificación para {name} ===")
    print(classification_report(y_test, y_pred_classes))
    model.save(f"global_{name}.keras")