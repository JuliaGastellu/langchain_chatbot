import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import json

def setup_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/chatbot.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_environment():
    """Valida que las variables de entorno necesarias estén configuradas"""
    required_vars = ['OPENROUTER_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == "your_openrouter_api_key_here":
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(f"Faltan las siguientes variables de entorno requeridas o no están configuradas: {', '.join(missing_vars)}")
    
    return True

def save_conversation_log(conversation: List[Dict[str, Any]], filename: str = None):
    """Guarda el log de conversación en un archivo"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/conversation_{timestamp}.json"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)
    
    return filename

def calculate_document_stats(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula estadísticas de los documentos procesados"""
    if not documents:
        return {"total_documents": 0, "total_words": 0, "avg_length": 0}
    
    total_words = sum(len(doc.get('content', '').split()) for doc in documents)
    avg_length = total_words / len(documents) if documents else 0
    
    return {
        "total_documents": len(documents),
        "total_words": total_words,
        "avg_length": round(avg_length, 2),
        "file_types": list(set(doc.get('metadata', {}).get('file_type', 'unknown') 
                              for doc in documents))
    }

def format_response_for_display(response: Dict[str, Any]) -> str:
    """Formatea la respuesta para mostrar en la interfaz"""
    answer = response.get('answer', '')
    confidence = response.get('confidence_score', 0)
    
    confidence_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
    
    formatted = f"{answer}\n\n{confidence_emoji} **Confianza:** {confidence:.1%}"
    
    return formatted

def clean_text(text: str) -> str:
    """Limpia y normaliza texto"""
    import re
    
    # Remover caracteres especiales excesivos
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

