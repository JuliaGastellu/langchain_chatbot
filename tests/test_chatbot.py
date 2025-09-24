import unittest
from unittest.mock import Mock, patch
import sys
import os

# Añadir el directorio padre al path para importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chatbot import TechnicalDocumentChatbot
from src.document_processor import DocumentProcessor
from langchain.schema import Document

class TestTechnicalDocumentChatbot(unittest.TestCase):
    """Tests para el chatbot técnico"""
    
    def setUp(self):
        """Configuración inicial para tests"""
        # Mock de la configuración
        self.mock_config = {
            'model': {
                'name': 'gpt-3.5-turbo',
                'temperature': 0.3,
                'max_tokens': 1000
            },
            'vectorstore': {
                'top_k': 3
            },
            'memory': {
                'k': 5
            }
        }
        
    @patch('src.chatbot.yaml.safe_load')
    @patch('builtins.open')
    def test_config_loading(self, mock_open, mock_yaml):
        """Test carga de configuración"""
        mock_yaml.return_value = self.mock_config
        
        chatbot = TechnicalDocumentChatbot()
        
        self.assertEqual(chatbot.config['model']['name'], 'gpt-3.5-turbo')
        self.assertEqual(chatbot.config['vectorstore']['top_k'], 3)
    
    def test_document_processor(self):
        """Test procesador de documentos"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Test con texto simple
        text = "Este es un documento de prueba. " * 10
        documents = processor.process_text(text, "test_source")
        
        self.assertIsInstance(documents, list)
        self.assertTrue(len(documents) > 0)
        self.assertIsInstance(documents[0], Document)
    
    @patch('src.chatbot.ChatOpenAI')
    @patch('src.chatbot.OpenAIEmbeddings')
    @patch('src.chatbot.Chroma')
    def test_chatbot_initialization(self, mock_chroma, mock_embeddings, mock_llm):
        """Test inicialización del chatbot"""
        with patch('src.chatbot.TechnicalDocumentChatbot._load_config', return_value=self.mock_config):
            chatbot = TechnicalDocumentChatbot()
            
            # Verificar que los componentes se inicializaron
            self.assertIsNotNone(chatbot.llm)
            self.assertIsNotNone(chatbot.embeddings)
            self.assertIsNotNone(chatbot.vector_store)
            self.assertIsNotNone(chatbot.memory)

class TestDocumentProcessor(unittest.TestCase):
    """Tests para el procesador de documentos"""
    
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_process_text(self):
        """Test procesamiento de texto"""
        text = "Este es un texto de prueba muy largo que debería ser dividido en chunks más pequeños para poder ser procesado correctamente por el sistema."
        
        documents = self.processor.process_text(text)
        
        self.assertIsInstance(documents, list)
        self.assertTrue(len(documents) >= 1)
        
        for doc in documents:
            self.assertIsInstance(doc, Document)
            self.assertTrue(len(doc.page_content) <= 120)  # chunk_size + algo de overlap
    
    def test_empty_text(self):
        """Test con texto vacío"""
        documents = self.processor.process_text("")
        
        # Debería manejar texto vacío graciosamente
        self.assertIsInstance(documents, list)

if __name__ == '__main__':
    # Crear directorios necesarios para tests
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/vector_db', exist_ok=True)
    
    unittest.main()

