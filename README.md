# S5-Cloud-Computing-DataProphet
Nombre del equipo: DataProphet.

Miembros del equipo:

1.- Ángel David Ávila Pérez. Departamento: Bases de datos. Puesto: Limpieza de datos

2.- Samantha Ruelas Valtierra. Departamento: Cloud Computing. Puesto: Software Development.

3.- Erika Martínez Meneses. Departamento: Modelos e implementación. Puesto: Científico de datos

4.- David Emmanuel Ramirez Stanford. Departamento: Ciberseguridad. Puesto: Control de Accesos.

5.- Andrea Axel Hernández Galgani. Departamento: Modelos e implementación. Puesto: Científico de datos.



Este proyecto implementa un flujo de Aprendizaje Federado usando TensorFlow y el conjunto de datos MNIST.

La estructura de este repositorio se divide en tres partes principales:

- model.py define la arquitectura de la red neuronal que utilizan todos los miembros del equipo.
- local_train.ipynb es el script que usa cada miembro para entrenar el modelo localmente con su subconjunto de datos.
- global_model.py se encarga de agregar los modelos entrenados por cada miembro en un modelo global, utilizando estrategias como FedAvg, FedMedian y FedWeightedAvg.Es solo representativo de como lo hicimos nosotros localmente.

Como parte del aprendizaje federado, cada miembro entrena con sus propio parte del dataset, siendo datos privados. Por esta razón, los archivos con los datos divididos (como split_data.py que se usó para la división del dataset y los archivos .npz) no están incluidos en este repositorio, ya que se consideran confidenciales.
- global_model_censurado.py es la version del codigo global_model que muestra los resultados del entrenamiento pero sin necesidad de mostrar los datos privados en el repositorio.Se puede correr para ver los resultados.


