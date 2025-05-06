import os
import tensorflow as tf
import numpy as np
from model import build_model
from sklearn.metrics import classification_report

# Cargar todos los pesos
weights_list = []
for i in range(5):
    model = build_model()  
    model.load_weights(f"model_weights_{i}.h5")  
    weights_list.append(model.get_weights())


fedavg_weights = [np.mean(w, axis=0) for w in zip(*weights_list)]
fedmedian_weights = [np.median(w, axis=0) for w in zip(*weights_list)]

data_sizes = [np.load(f"Division/Dataprophet_PT{i}.npz")['x'].shape[0] for i in range(5)]
total_size = sum(data_sizes)
fedweighted_weights = []
for weights in zip(*weights_list):
    weighted = sum(w * s for w, s in zip(weights, data_sizes))
    fedweighted_weights.append(weighted / total_size)


np.savez("fedavg_weights.npz", *fedavg_weights)
np.savez("fedmedian_weights.npz", *fedmedian_weights)
np.savez("fedweighted_weights.npz", *fedweighted_weights)

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