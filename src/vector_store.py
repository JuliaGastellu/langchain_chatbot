from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AdvancedVectorStore:
    """
    Almacén de vectores avanzado que soporta ChromaDB y FAISS, con embeddings
    configurables de HuggingFace.
    """
    def __init__(self, 
                 embedding_config: Dict[str, Any],
                 store_type: str = "chroma",
                 persist_directory: str = "data/vector_db",
                 collection_name: str = "technical_docs"):
        """
        Inicializa el almacén de vectores.

        Args:
            embedding_config (Dict[str, Any]): Configuración para el modelo de embeddings.
            store_type (str): Tipo de almacén a usar ('chroma' o 'faiss').
            persist_directory (str): Directorio para persistir los datos.
            collection_name (str): Nombre de la colección en ChromaDB.
        """
        self.store_type = store_type
        self.persist_directory = os.path.join(persist_directory, store_type)
        self.collection_name = collection_name
        
        logger.info(f"Inicializando embeddings con el modelo: {embedding_config.get('model_name')}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            model_kwargs=embedding_config.get("model_kwargs", {}),
            encode_kwargs=embedding_config.get("encode_kwargs", {})
        )

        self.vector_store = None
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Configura el almacén de vectores según el tipo especificado."""
        os.makedirs(self.persist_directory, exist_ok=True)
        logger.info(f"Configurando almacén de vectores tipo '{self.store_type}' en '{self.persist_directory}'")
        
        try:
            if self.store_type == "chroma":
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            elif self.store_type == "faiss":
                faiss_index_path = os.path.join(self.persist_directory, "index.faiss")
                if os.path.exists(faiss_index_path):
                    self.vector_store = FAISS.load_local(
                        self.persist_directory,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    logger.warning(f"No se encontró un índice FAISS en {self.persist_directory}. Creando uno nuevo y vacío.")
                    # FAISS necesita ser inicializado con al menos un documento
                    dummy_texts = ["Inicialización"]
                    self.vector_store = FAISS.from_texts(dummy_texts, self.embeddings)
                    self.vector_store.save_local(self.persist_directory)
        except Exception as e:
            logger.error(f"Error al configurar el almacén de vectores: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Añade una lista de documentos al almacén de vectores."""
        if not documents:
            logger.warning("Se intentó añadir una lista vacía de documentos.")
            return {"status": "warning", "message": "No se proporcionaron documentos para añadir."}

        try:
            for doc in documents:
                doc.metadata['added_at'] = datetime.now().isoformat()
            
            logger.info(f"Añadiendo {len(documents)} documentos al almacén '{self.store_type}'.")
            if self.store_type == "chroma":
                self.vector_store.add_documents(documents)
            elif self.store_type == "faiss":
                self.vector_store.add_documents(documents)
                self.vector_store.save_local(self.persist_directory)
            
            return {
                "status": "success",
                "added_documents": len(documents),
                "total_documents": self.get_document_count()
            }
        except Exception as e:
            logger.error(f"Error al añadir documentos: {e}")
            return {"status": "error", "message": str(e)}
    
    def as_retriever(self, search_kwargs: Dict = None) -> Any:
        """Devuelve el almacén de vectores como un retriever de LangChain."""
        if not search_kwargs:
            search_kwargs = {"k": 5}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def get_document_count(self) -> int:
        """Obtiene el número total de documentos en el almacén."""
        try:
            if self.store_type == "chroma":
                return self.vector_store._collection.count()
            elif self.store_type == "faiss":
                return self.vector_store.index.ntotal
            return 0
        except Exception as e:
            logger.warning(f"No se pudo obtener el conteo de documentos: {e}")
            return 0

    def delete_collection(self):
        """Elimina la colección o los archivos del índice."""
        logger.warning(f"Eliminando datos del almacén de vectores en '{self.persist_directory}'")
        try:
            if self.store_type == "chroma":
                self.vector_store.delete_collection()
                logger.info("Colección de ChromaDB eliminada.")
            elif self.store_type == "faiss":
                faiss_index_path = os.path.join(self.persist_directory, "index.faiss")
                faiss_pkl_path = os.path.join(self.persist_directory, "index.pkl")
                if os.path.exists(faiss_index_path):
                    os.remove(faiss_index_path)
                if os.path.exists(faiss_pkl_path):
                    os.remove(faiss_pkl_path)
                logger.info("Archivos de índice FAISS eliminados.")
            
            # Re-inicializar un almacén vacío
            self._setup_vector_store()
            return {"status": "success", "message": "Almacén de vectores limpiado."}
        except Exception as e:
            logger.error(f"Error al eliminar la colección: {e}")
            return {"status": "error", "message": str(e)}
