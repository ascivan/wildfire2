# ========================================
#   STREAMLIT APP - CLASIFICACIÓN INCENDIOS FORESTALES (MÚLTIPLES IMÁGENES)
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
# Asegúrate de que tu modelo 'keras_model.h5' esté en el mismo directorio que tu script.
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Labels (wildfire or nowildfire)
class_labels = ["wildfire", "nowildfire"]

# ==============================
# FUNCIÓN DE CLASIFICACIÓN
# ==============================
def clasificar_imagen(imagen_bytes):
    """Clasifica una imagen en bytes y retorna las predicciones."""
    img_array = io.imread(imagen_bytes) / 255.0
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
st.title("🔥🌳 Clasificador de Incendios Forestales")
st.write("Sube hasta 5 imágenes y el modelo las clasificará como con o sin incendio.")

uploaded_files = st.file_uploader(
    "Selecciona una o más imágenes (hasta 5)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Se ha superado el límite de 5 imágenes. Solo se procesarán las primeras 5.")
        uploaded_files = uploaded_files[:5]
    
    st.markdown("---")
    
    # Crea una matriz de columnas para cada imagen
    cols = st.columns(len(uploaded_files))
    
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i]:
            st.markdown(f"**Imagen {i+1}**")
            
            # Muestra la imagen subida
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
            
            # Clasifica la imagen
            # Usa BytesIO para leer el archivo sin guardarlo en disco
            pred = clasificar_imagen(uploaded_file)
            predicted_class = np.argmax(pred)
            predicted_probability = pred[predicted_class]

            # Muestra la clasificación y la probabilidad
            color = "red" if predicted_class == 0 else "green"
            message = f'<p style="color: {color}; font-size: 18px;"><b>{class_labels[predicted_class]}</b><br>Probabilidad: {predicted_probability:.3f}</p>'
            st.markdown(message, unsafe_allow_html=True)
            
            # Añade una línea horizontal para separar los resultados
            st.markdown("---")
