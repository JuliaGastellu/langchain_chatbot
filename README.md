# Chatbot Inteligente para Análisis de Documentos Técnicos con LangChain

## Descripción del Proyecto:

Un chatbot avanzado que combina múltiples capacidades de LangChain para analizar, procesar y responder preguntas sobre documentos técnicos, con memoria conversacional, búsqueda semántica y generación de resúmenes inteligentes.

## Características Principales:

RAG (Retrieval Augmented Generation) para documentos técnicos                   
Memoria conversacional persistente                               
Análisis multimodal (texto, PDFs, imágenes)                                     
Generación de resúmenes ejecutivos                              
Búsqueda semántica avanzada                    
Interfaz web interactiva                                                      
Sistema de feedback para mejorar respuestas                                               

## Tecnologías Utilizadas:

LangChain                             
OpenAI GPT-4                                                     
ChromaDB (Vector Database)                                           
Streamlit (Interfaz)                                         
PyPDF2 / Unstructured                                              
FAISS para búsqueda vectorial     

## Estructura del Proyecto:

langchain_chatbot/                        
├── src/                                         
│   ├── __init__.py                                
│   ├── chatbot.py                                
│   ├── document_processor.py                              
│   ├── memory_manager.py                               
│   ├── vector_store.py                       
│   └── utils.py                                    
├── data/                                            
│   ├── documents/                                      
│   └── vector_db/                                
├── config/                                       
│   └── config.yaml                                        
├── app.py                                       
├── requirements.txt                                   
├── README.md                                            
└── .env                                                    

## Instalación:

### Clonar el proyecto
git clone <repository-url>
cd langchain_chatbot

### Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

### Instalar dependencias
pip install -r requirements.txt

### Configurar variables de entorno
cp .env.example .env

### Editar .env con tus API keys

## Configuración

.env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_key

config/config.yaml
model:
  name: "gpt-4"
  temperature: 0.3
  max_tokens: 2000

vectorstore:
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 5

memory:
  type: "conversation_buffer_window"
  k: 10

Uso:

## Ejecutar la aplicación
streamlit run app.py

Casos de Uso
Análisis de Manuales Técnicos
Revisión de Documentación de APIs
Procesamiento de Papers Académicos
Análisis de Contratos y Documentos Legales
Generación de Reportes Ejecutivos
