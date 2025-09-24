import streamlit as st
from dotenv import load_dotenv
import os
import yaml
from src.chatbot import TechnicalDocumentChatbot
from src.utils import validate_environment, setup_logging
import logging

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Asistente de Documentos Técnicos",
    page_icon="🤖",
    layout="wide"
)

# --- Carga de Entorno y Logging ---
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# --- Funciones Auxiliares de la UI ---
def load_config(config_path='config/config.yaml'):
    """Carga la configuración de la aplicación."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"El archivo de configuración '{config_path}' no fue encontrado. Asegúrate de que existe.")
        st.stop()

def initialize_chatbot():
    """Inicializa el chatbot y lo guarda en el estado de la sesión."""
    if "chatbot" not in st.session_state:
        logger.info("Inicializando chatbot por primera vez.")
        try:
            validate_environment()
            st.session_state.chatbot = TechnicalDocumentChatbot()
            if "messages" not in st.session_state:
                st.session_state.messages = []
            logger.info("Chatbot inicializado y guardado en session_state.")
        except EnvironmentError as e:
            st.error(f"Error de Configuración: {e}. Revisa tu archivo .env.")
            logger.critical(f"Error de variable de entorno: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error Crítico al Inicializar el Chatbot: {e}")
            logger.critical(f"No se pudo inicializar el chatbot: {e}", exc_info=True)
            st.stop()

def display_chat_history():
    """Muestra el historial del chat en la interfaz."""
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt: str):
    """Procesa la entrada del usuario, obtiene respuesta y actualiza el chat."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Analizando documentos y generando respuesta..."):
            response = st.session_state.chatbot.ask(prompt)
            answer = response.get('answer', 'Lo siento, no pude generar una respuesta.')
            message_placeholder.markdown(answer)

            if response.get("source_documents"):
                with st.expander("Ver fuentes consultadas"):
                    for doc in response["source_documents"]:
                        source_name = doc.metadata.get('file_name', 'N/A')
                        content_preview = doc.page_content[:250].replace('\n', ' ') + "..."
                        st.info(f"**Fuente:** {source_name}\n\n**Fragmento:**\n>{content_preview}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Interfaz Principal ---
config = load_config()
st.title(config['ui']['title'])
st.caption("Sube tus documentos técnicos (PDF, DOCX, TXT) y haz preguntas específicas sobre su contenido.")

initialize_chatbot()

# --- Barra Lateral ---
with st.sidebar:
    st.header("Gestión de Documentos")

    uploaded_files = st.file_uploader(
        "Sube tus archivos aquí",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Puedes subir múltiples documentos a la vez."
    )

    if st.button("Procesar Documentos", use_container_width=True):
        if uploaded_files:
            with st.spinner("Procesando documentos... Esto puede tardar unos minutos."):
                result = st.session_state.chatbot.process_documents(uploaded_files)
                if result.get("status") == "success":
                    st.success(f"¡Éxito! Se añadieron {result.get('added_documents', 0)} nuevos fragmentos a la base de conocimiento.")
                else:
                    st.warning(result.get("message", "No se procesaron nuevos documentos."))
        else:
            st.warning("Por favor, sube al menos un archivo para procesar.")

    st.divider()

    st.header("Control de la Sesión")
    if "chatbot" in st.session_state:
        doc_count = st.session_state.chatbot.vector_store.get_document_count()
        st.metric("Fragmentos en Memoria", doc_count)

    if st.button("Limpiar Conversación", use_container_width=True):
        st.session_state.chatbot.clear_conversation()
        st.session_state.messages = []
        st.success("El historial de la conversación ha sido limpiado.")
        st.rerun()

    if st.button("Resetear Base de Conocimiento", type="primary", use_container_width=True):
        with st.spinner("Eliminando todos los documentos..."):
            st.session_state.chatbot.reset_vector_store()
            st.session_state.chatbot.clear_conversation()
            st.session_state.messages = []
        st.success("La base de conocimiento y la conversación han sido reseteadas.")
        st.rerun()

# --- Área de Chat ---
display_chat_history()

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    if "chatbot" in st.session_state:
        handle_user_input(prompt)
    else:
        st.error("El chatbot no está inicializado. Por favor, refresca la página.")
