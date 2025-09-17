# ========================================
#   STREAMLIT APP - CLASIFICACIÓN INCENDIOS FORESTALES
# ========================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import streamlit as st
import numpy as np
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
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
# Asegúrate de que tu modelo 'keras_model.h5' esté en el mismo directorio.
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
st.title("🔥🌲 Clasificador de Incendios Forestales")
st.write("Sube hasta 10 imágenes y verás la clasificación y probabilidad de cada una.")

# Inicializa st.session_state si aún no existe
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ==============================
# GESTIÓN DE ARCHIVOS Y BOTONES
# ==============================
def reset_uploader():
    """Función para limpiar los archivos subidos y forzar el reinicio del uploader."""
    st.session_state.uploaded_files = []
    st.session_state.uploader_key += 1
    st.experimental_rerun()

# Contenedor para los botones y el uploader
col_uploader, col_reset = st.columns([4, 1])

with col_uploader:
    new_uploaded_files = st.file_uploader(
        "Selecciona una o más imágenes (máximo 10)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key
    )
    # Si se cargaron nuevos archivos, actualiza el estado de la sesión
    if new_uploaded_files:
        st.session_state.uploaded_files = new_uploaded_files
        
with col_reset:
    st.write("") # Espacio para alinear el botón
    if st.button("Reiniciar"):
        reset_uploader()

# ==============================
# MOSTRAR RESULTADOS
# ==============================
if st.session_state.uploaded_files:
    if len(st.session_state.uploaded_files) > 10:
        st.warning("Se ha superado el límite de 10 imágenes. Solo se procesarán las primeras 10.")
        st.session_state.uploaded_files = st.session_state.uploaded_files[:10]
    
    st.markdown("---")
    
    # Crea las columnas de la tabla (solo los encabezados)
    # Ajustar el ratio de las columnas si es necesario para acomodar imágenes más grandes
    # Por ejemplo, la primera columna (imagen) podría necesitar más espacio
    col1, col2, col3, col4 = st.columns([1.5, 2, 1.5, 1]) # Ajustado col1 de 1 a 1.5
    with col1:
        st.subheader("Imagen")
    with col2:
        st.subheader("Nombre de archivo")
    with col3:
        st.subheader("Clasificación")
    with col4:
        st.subheader("Probabilidad")
    
    st.markdown("---")

    for i, uploaded_file in enumerate(st.session_state.uploaded_files):
        # Divide la fila en 4 columnas
        col1, col2, col3, col4 = st.columns([1.5, 2, 1.5, 1]) # Ajustado col1 de 1 a 1.5
        
        # Columna 1: Imagen
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, width=150) # ¡Aquí está el cambio!
            
        # Clasificar la imagen
        pred = clasificar_imagen(uploaded_file)
        predicted_class = np.argmax(pred)
        predicted_probability = pred[predicted_class]

        # Columna 2: Nombre de archivo
        with col2:
            st.markdown(f"**{uploaded_file.name}**")
            
        # Columna 3: Clasificación
        with col3:
            color = "red" if predicted_class == 0 else "green"
            message = f'<p style="color: {color}; font-size: 18px;"><b>{class_labels[predicted_class]}</b></p>'
            st.markdown(message, unsafe_allow_html=True)
            
        # Columna 4: Probabilidad
        with col4:
            st.markdown(f"**{predicted_probability:.3f}**")
        
        st.markdown("---")
