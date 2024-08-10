# Predicción mercado valores por Numerai

Para instalar tensorflow se han seguido las especificaciones de la página oficial para windows:
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

python -m pip install "tensorflow<2.11"

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

# 1. [Descarga de datos](https://github.com/PacoBermudez/tfm_modelo_mervado_valores/blob/main/download_data.py)

Para realizar la descarga de datos directamente desde numerai hay que ejecutar el script:

download_data.py

Cuando se ejecute este script se creara una nueva carpeta en nuestro directorio, la cuál contendrá los ficheros necesarios tanto para el entrenamiento como para la validación de los resultados.

# 2. Preparación de los datos 
