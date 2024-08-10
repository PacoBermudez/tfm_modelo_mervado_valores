# tfm_modelo_mervado_valores

Para instalar tensorflow se han seguido las especificaciones de la página oficial para windows:
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

python -m pip install "tensorflow<2.11"

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
