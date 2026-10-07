import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Control de Calidad Industrial",
    page_icon="🔍",
    layout="wide",
)

# Estilos limpios y corporativos (menos IA, más industrial)
st.markdown("""
<style>
    .main-header {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 16px;
        margin-bottom: 24px;
    }
    .badge-ok {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        font-size: 13px;
    }
    .badge-def {
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        font-size: 13px;
    }
    .card-ok {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 24px;
        margin-top: 10px;
    }
    .card-def {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 24px;
        margin-top: 10px;
    }
    .title-ok {
        color: #15803d;
        font-size: 28px;
        font-weight: 800;
        margin: 8px 0;
    }
    .title-def {
        color: #b91c1c;
        font-size: 28px;
        font-weight: 800;
        margin: 8px 0;
    }
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        border-top: 1px solid #e2e8f0;
        padding-top: 16px;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(BASE_DIR, "modelo_inspeccion_industrial.h5")
IMG_SIZE = 128

@st.cache_resource
def load_trained_model():
    if not os.path.exists(H5_PATH):
        st.error(f"No se encontró el archivo del modelo en: {H5_PATH}")
        return None
    return tf.keras.models.load_model(H5_PATH)

model = load_trained_model()

# Encabezado
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; color: #0f172a; font-size: 30px;">Control de Calidad de Piezas</h1>
    <p style="margin: 6px 0 0 0; color: #64748b; font-size: 15px;">
        Inspección visual automatizada mediante Deep Learning (MobileNetV2 Fine-Tuning)
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Cargar o Seleccionar Pieza")
    
    examples_dir = os.path.join(BASE_DIR, "assets", "examples")
    sample_files = []
    if os.path.exists(examples_dir):
        sample_files = sorted([f for f in os.listdir(examples_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    if not sample_files:
        examples_dir = BASE_DIR
        sample_files = sorted([f for f in os.listdir(BASE_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png')) and ('def' in f or 'ok' in f or 'sample' in f)])

    input_mode = st.radio(
        "Modo de entrada:",
        ["Muestras de referencia (1 clic)", "Subir imagen personalizada"],
        horizontal=True
    )

    selected_image = None

    if input_mode == "Muestras de referencia (1 clic)":
        if sample_files:
            sample_choice = st.selectbox(
                "Selecciona una muestra de fundición:",
                sample_files,
                format_func=lambda x: f"⚠️ Defecto: {x}" if "def" in x else f"✅ Conforme: {x}"
            )
            selected_path = os.path.join(examples_dir, sample_choice)
            selected_image = Image.open(selected_path)
    else:
        uploaded_file = st.file_uploader(
            "Arrastra una imagen de la pieza (.jpeg, .png, .jpg):",
            type=["jpeg", "jpg", "png"]
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)

    if selected_image is not None:
        st.image(selected_image, caption="Fotografía de la pieza en inspección", use_container_width=True)

with col2:
    st.subheader("Dictamen de Inspección")

    if selected_image is None:
        st.info("👈 Selecciona una muestra o sube una fotografía para ejecutar el control de calidad en tiempo real.")
    else:
        if model is None:
            st.error("El modelo no está disponible.")
        else:
            with st.spinner("Analizando microestructura y defectos superficiales..."):
                img_rgb = selected_image.convert("RGB")
                img_resized = img_rgb.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                pred = float(model.predict(img_array, verbose=0)[0][0])

            # En Keras alfabético: def_front=0, ok_front=1
            if pred > 0.5:
                confidence = pred
                conf_pct = f"{confidence * 100:.2f}%"
                st.markdown(f"""
                <div class="card-ok">
                    <span class="badge-ok">PIEZA CONFORME</span>
                    <div class="title-ok">APROBADA</div>
                    <p style="color: #334155; margin: 4px 0 12px 0; font-size: 14px;">
                        <strong>Nivel de confianza:</strong> {conf_pct}
                    </p>
                    <p style="color: #1e293b; font-size: 14px; margin: 0; line-height: 1.5;">
                        <strong>Acción operativa:</strong> Apta para continuar en la línea de ensamble de bomba sumergible. No se detectan anomalías de fundición ni porosidades.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(confidence)
            else:
                confidence = 1.0 - pred
                conf_pct = f"{confidence * 100:.2f}%"
                st.markdown(f"""
                <div class="card-def">
                    <span class="badge-def">DEFECTO DETECTADO</span>
                    <div class="title-def">RECHAZADA</div>
                    <p style="color: #334155; margin: 4px 0 12px 0; font-size: 14px;">
                        <strong>Nivel de confianza:</strong> {conf_pct}
                    </p>
                    <p style="color: #1e293b; font-size: 14px; margin: 0; line-height: 1.5;">
                        <strong>Acción operativa:</strong> Retirar inmediatamente de la línea de ensamble. Pieza enviada a contenedor de descarte o reproceso por fisura / poro metálico.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(confidence)

            st.write("")
            with st.expander("Detalles Técnicos de Inferencia"):
                st.write(f"- **Salida de la capa sigmoide:** `{pred:.6f}`")
                st.write(f"- **Umbral de clasificación:** `0.5`")
                st.write(f"- **Arquitectura:** `MobileNetV2 + Fine-Tuning (ImageNet)`")
                st.write(f"- **Resolución procesada:** `{IMG_SIZE}x{IMG_SIZE} px (3 canales RGB)`")
                st.write(f"- **Precisión global validada:** `99.72%`")

st.markdown("""
<div class="footer-text">
    Proyecto Final: Inspección Industrial y Control de Calidad mediante Deep Learning<br>
    Profesor: Guillermo R. Aragón Pacheco · Alumno: Alfredo Ramos Olivo
</div>
""", unsafe_allow_html=True)
