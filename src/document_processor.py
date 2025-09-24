import os
import concurrent.futures
from typing import List, Any
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Procesador optimizado de documentos con procesamiento paralelo y carga selectiva.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, max_workers: int = 4):
        """
        Inicializa el procesador de documentos optimizado.
        
        Args:
            chunk_size (int): Tamaño máximo de cada fragmento
            chunk_overlap (int): Solapamiento entre fragmentos
            max_workers (int): Número de hilos para procesamiento paralelo
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        self.temp_dir = "temp_docs"
        self.max_workers = max_workers
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Mapeo de extensiones a loaders optimizados
        self.loader_mapping = {
            '.pdf': self._load_pdf_fast,
            '.txt': self._load_text_fast,
            '.docx': self._load_unstructured_fast,
            '.doc': self._load_unstructured_fast
        }

    def _get_file_extension(self, filename: str) -> str:
        """Obtiene la extensión del archivo en minúsculas."""
        return Path(filename).suffix.lower()

    def _load_pdf_fast(self, file_path: str) -> List[Document]:
        """Carga PDFs usando PyPDFLoader que es más rápido."""
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.warning(f"Error con PyPDFLoader, usando UnstructuredFileLoader: {e}")
            return self._load_unstructured_fast(file_path)

    def _load_text_fast(self, file_path: str) -> List[Document]:
        """Carga archivos de texto directamente."""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        except UnicodeDecodeError:
            try:
                loader = TextLoader(file_path, encoding='latin-1')
                return loader.load()
            except Exception as e:
                logger.error(f"Error cargando archivo de texto {file_path}: {e}")
                return []
        except Exception as e:
            logger.error(f"Error cargando archivo de texto {file_path}: {e}")
            return []

    def _load_unstructured_fast(self, file_path: str) -> List[Document]:
        """Carga usando UnstructuredFileLoader con configuración optimizada."""
        try:
            loader = UnstructuredFileLoader(
                file_path,
                mode="single",
                strategy="fast",  # Estrategia más rápida
            )
            return loader.load()
        except Exception as e:
            logger.error(f"Error cargando con Unstructured {file_path}: {e}")
            return []

    def _process_single_file(self, uploaded_file) -> List[Document]:
        """Procesa un solo archivo de forma optimizada."""
        temp_path = os.path.join(self.temp_dir, uploaded_file.name)
        documents = []
        
        try:
            # Escribir archivo temporal
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Seleccionar loader optimizado según extensión
            file_ext = self._get_file_extension(uploaded_file.name)
            loader_func = self.loader_mapping.get(file_ext, self._load_unstructured_fast)
            
            # Cargar documento
            docs = loader_func(temp_path)
            
            # Añadir metadatos
            for doc in docs:
                doc.metadata.update({
                    "file_name": uploaded_file.name,
                    "file_type": file_ext,
                    "file_size": len(uploaded_file.getvalue())
                })
            
            documents.extend(docs)
            logger.info(f"Procesado {uploaded_file.name}: {len(docs)} documentos")
            
        except Exception as e:
            logger.error(f"Error procesando {uploaded_file.name}: {e}")
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"No se pudo eliminar archivo temporal {temp_path}: {e}")
        
        return documents

    def process_uploaded_files(self, uploaded_files: List[Any]) -> List[Document]:
        """
        Procesa archivos en paralelo para mayor velocidad.
        """
        if not uploaded_files:
            return []
        
        logger.info(f"Iniciando procesamiento paralelo de {len(uploaded_files)} archivos")
        all_documents = []
        
        # Procesamiento paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self._process_single_file, file): file 
                            for file in uploaded_files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Error en procesamiento paralelo de {file.name}: {e}")
        
        if not all_documents:
            logger.warning("No se pudieron procesar documentos")
            return []
        
        # Dividir en fragmentos (esto también se puede paralelizar si es necesario)
        logger.info(f"Dividiendo {len(all_documents)} documentos en fragmentos")
        split_documents = self.text_splitter.split_documents(all_documents)
        
        logger.info(f"Procesamiento completado: {len(uploaded_files)} archivos → {len(split_documents)} fragmentos")
        return split_documents

    def process_text(self, text: str, source_name: str = "texto_directo") -> List[Document]:
        """Procesa texto directo de forma optimizada."""
        if not text or not text.strip():
            logger.warning("Texto vacío proporcionado")
            return []
            
        try:
            doc = Document(
                page_content=text,
                metadata={"source": source_name, "file_name": source_name}
            )
            
            split_documents = self.text_splitter.split_documents([doc])
            logger.info(f"Texto procesado en {len(split_documents)} fragmentos")
            
            return split_documents
            
        except Exception as e:
            logger.error(f"Error procesando texto: {e}")
