# STT
Speech To Text (STT) model

La carpeta "stt" contiene el código del modelo. Si se quiere realizar el entrenamiento se recomienda que se realice un entorno con CUDA y CUDNN instalados para acelerar el proceso de entrenamiento. Además se deben instalar las librerías de mltu, TensorFlow, tkinter para la interfaz, y finalmente torch y onnx para leer el archivo .onnx.

En la carpeta "Models"se encuentra el archivo .onnx del modelo pre-entrenado bajo 'Models/05_sound_to_text/202411180003/model.onnx'. Para solamente correr el modelo se debe instalar este archivo y el archivo inferenceModel.py en un entorno con las librerías anteriormente mencionadas.
