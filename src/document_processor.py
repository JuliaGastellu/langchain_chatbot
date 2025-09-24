import os
from typing import List, Any
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import logging

# Configurar logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Procesa documentos de varios formatos para su posterior uso en un sistema RAG.
    Utiliza Unstructured para extraer texto de archivos como PDF, DOCX, TXT.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        """
        Inicializa el procesador de documentos.

        Args:
            chunk_size (int): El tamaño máximo de cada fragmento de texto.
            chunk_overlap (int): El número de caracteres que se solapan entre fragmentos.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        self.temp_dir = "temp_docs"
        os.makedirs(self.temp_dir, exist_ok=True)

    def process_uploaded_files(self, uploaded_files: List[Any]) -> List[Document]:
        """
        Procesa una lista de archivos subidos (generalmente desde Streamlit).

        Guarda temporalmente cada archivo, lo carga con Unstructured, lo divide en
        fragmentos y añade metadatos relevantes.

        Args:
            uploaded_files (List[Any]): Una lista de objetos de archivo subidos.

        Returns:
            List[Document]: Una lista de documentos procesados y divididos, listos
                            para ser insertados en un almacén de vectores.
        """
        all_documents = []
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(self.temp_dir, uploaded_file.name)
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                loader = UnstructuredFileLoader(
                    temp_path,
                    mode="single",
                    strategy="fast",
                )
                documents = loader.load()

                for doc in documents:
                    doc.metadata["file_name"] = uploaded_file.name

                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error procesando el archivo {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if not all_documents:
            return []

        split_documents = self.text_splitter.split_documents(all_documents)
        logger.info(f"Procesados {len(uploaded_files)} archivos, resultando en {len(split_documents)} fragmentos.")

        return split_documents

    def process_text(self, text: str, source_name: str = "texto_directo") -> List[Document]:
        """
        Procesa texto directo sin necesidad de archivo.
        
        Args:
            text (str): El texto a procesar
            source_name (str): Nombre identificativo del origen del texto
            
        Returns:
            List[Document]: Lista de documentos procesados y divididos
        """
        if not text or not text.strip():
            logger.warning("Se proporcionó texto vacío para procesar")
            return []
            
        try:
            # Crear documento base
            doc = Document(
                page_content=text,
                metadata={"source": source_name, "file_name": source_name}
            )
            
            # Dividir en fragmentos
            split_documents = self.text_splitter.split_documents([doc])
            logger.info(f"Texto procesado en {len(split_documents)} fragmentos")
            
            return split_documents
            
        except Exception as e:
            logger.error(f"Error procesando texto directo: {e}")
            return []
