# ========================================
#   STREAMLIT APP - CLASIFICACIÓN INCENDIOS FORESTALES
# ========================================

# Instalar dependencias (ejecutar en terminal antes de correr streamlit)
# pip install streamlit pillow opencv-python matplotlib scikit-image tensorflow==2.15 keras==2.15

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import streamlit as st
import numpy as np
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage import io

# ==============================
# PATCH para DepthwiseConv2D (por compatibilidad con keras==2.15.0)
# ==============================
original_from_config = DepthwiseConv2D.from_config

@classmethod
def patched_from_config(cls, config):
    config.pop('groups', None)  # elimina argumento 'groups' si existe
    return original_from_config(config)

DepthwiseConv2D.from_config = patched_from_config

# ==============================
# CARGAR MODELO
# ==============================
model = tf.keras.models.load_model("keras_modelset.h5", compile=False)

# Labels (wildfire or nowildfire)
class_labels = ["wildfire", "nowildfire"]

# Crear directorio temporal
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

# ==============================
# FUNCIÓN DE CLASIFICACIÓN
# ==============================
def clasificar_imagen(imagen_path):
    img_array = io.imread(imagen_path) / 255.0
    img_resized = ImageOps.fit(
        Image.fromarray((img_array * 255).astype(np.uint8)),
        (224, 224),
        Image.Resampling.LANCZOS
    )
    img_array_resized = np.asarray(img_resized)
    normalized_image_array = (img_array_resized.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    pred = model.predict(data)[0]
    return pred

# ==============================
# INTERFAZ STREAMLIT
# ==============================
st.title("🔥🌲 Clasificador de Incendios Forestales")
st.write("Sube hasta 10 imágenes y el modelo las clasificará como con o sin incendio.")

# Modificar st.file_uploader para aceptar múltiples archivos
uploaded_files = st.file_uploader("Selecciona imágenes", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Limitar a un máximo de 10 archivos
    if len(uploaded_files) > 10:
        st.warning("Solo se procesarán las primeras 10 imágenes.")
        uploaded_files = uploaded_files[:10]
    
    # Crear columnas para la matriz
    cols = st.columns(3) # Tres columnas por fila para la matriz

    for i, uploaded_file in enumerate(uploaded_files):
        # Guardar archivo temporal
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Clasificar
        pred = clasificar_imagen(temp_path)
        predicted_class = np.argmax(pred)
        predicted_probability = pred[predicted_class]
        
        # Seleccionar la columna para la imagen actual
        with cols[i % 3]: # Usa el módulo para ciclar entre las 3 columnas
            # Mostrar la imagen
            st.image(uploaded_file, caption=uploaded_file.name)

            # Color dinámico según clase
            color = "red" if predicted_class == 0 else "green"

            # Mostrar resultado con formato
            message = f'<p style="color: {color}; font-size: 20px;"><b>{class_labels[predicted_class]}</b></p>'
            probability_message = f'<p style="font-size: 16px;">Probabilidad: {predicted_probability:.3f}</p>'
            st.markdown(message, unsafe_allow_html=True)
            st.markdown(probability_message, unsafe_allow_html=True)
            
        # Eliminar archivo temporal después de procesar
        os.remove(temp_path)