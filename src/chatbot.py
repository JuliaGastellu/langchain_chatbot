import os
import yaml
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.document import Document
from src.document_processor import DocumentProcessor
from src.vector_store import AdvancedVectorStore
from src.memory_manager import AdvancedMemoryManager
import logging

logger = logging.getLogger(__name__)

class TechnicalDocumentChatbot:
    """
    Orquesta el chatbot para análisis de documentos técnicos, integrando todos los componentes.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Inicializa el chatbot cargando configuración y preparando los componentes.
        """
        self.config = self._load_config(config_path)
        self.llm = self._init_llm()
        self.document_processor = DocumentProcessor(
            chunk_size=self.config['vectorstore']['chunk_size'],
            chunk_overlap=self.config['vectorstore']['chunk_overlap']
        )
        self.vector_store = self._init_vector_store()
        self.memory_manager = AdvancedMemoryManager(
            memory_type=self.config['memory']['type'],
            k=self.config['memory']['k'],
            llm=self.llm
        )
        self.chain = self._create_chain()
        logger.info("Chatbot inicializado correctamente.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración desde un archivo YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Archivo de configuración no encontrado en: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Error cargando la configuración: {e}")
            raise

    def _init_llm(self) -> ChatOpenAI:
        """Inicializa el modelo de lenguaje (LLM) apuntando a OpenRouter."""
        logger.info(f"Inicializando LLM: {self.config['model']['name']}")
        return ChatOpenAI(
            model=self.config['model']['name'],
            temperature=self.config['model']['temperature'],
            max_tokens=self.config['model']['max_tokens'],
            streaming=self.config['model']['streaming'],
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_API_BASE"),
            default_headers={
                "HTTP-Referer": os.getenv("YOUR_SITE_URL"),
                "X-Title": os.getenv("YOUR_SITE_NAME"),
            }
        )

    def _init_vector_store(self) -> AdvancedVectorStore:
        """Inicializa el almacén de vectores."""
        return AdvancedVectorStore(
            embedding_config=self.config['embedding'],
            store_type=self.config['vectorstore']['store_type'],
            collection_name=self.config['vectorstore']['collection_name']
        )

    def _create_chain(self) -> ConversationalRetrievalChain:
        """Crea la cadena de recuperación conversacional."""
        custom_prompt_template = """
        Eres un asistente de IA experto en analizar documentos técnicos.
        Usa los siguientes fragmentos de contexto para responder la pregunta al final.
        Si la respuesta no se encuentra en el contexto, di amablemente que no tienes esa información.
        Proporciona respuestas claras, concisas y en español.

        Contexto:
        {context}

        Historial de la conversación:
        {chat_history}

        Pregunta: {question}
        Respuesta útil en español:
        """
        
        PROMPT = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "chat_history", "question"]
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": self.config['vectorstore']['top_k']}
            ),
            memory=self.memory_manager.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def process_documents(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """Procesa y almacena los documentos subidos."""
        logger.info(f"Procesando {len(uploaded_files)} archivos.")
        documents = self.document_processor.process_uploaded_files(uploaded_files)
        if not documents:
            return {"status": "warning", "message": "No se pudieron procesar los documentos."}
        
        result = self.vector_store.add_documents(documents)
        logger.info(f"Documentos añadidos al almacén: {result}")
        return result

    def ask(self, query: str) -> Dict[str, Any]:
        """Realiza una pregunta al chatbot."""
        logger.info(f"Recibida pregunta: '{query}'")
        if self.vector_store.get_document_count() == 0:
            return {
                "answer": "Por favor, sube y procesa algunos documentos antes de hacer una pregunta.",
                "source_documents": []
            }
        
        result = self.chain({"question": query})
        logger.info(f"Respuesta generada para: '{query}'")
        return result

    def clear_conversation(self):
        """Limpia el historial de la conversación."""
        self.memory_manager.clear_memory()
        logger.info("Historial de conversación limpiado.")

    def reset_vector_store(self):
        """Elimina todos los documentos del almacén de vectores."""
        self.vector_store.delete_collection()
        logger.info("Almacén de vectores reseteado.")