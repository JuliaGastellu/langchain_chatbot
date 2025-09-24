from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMessage
from typing import Dict, Any, List
import json
import os
from datetime import datetime

class AdvancedMemoryManager:
    """Gestor avanzado de memoria conversacional"""
    
    def __init__(self, memory_type: str = "buffer_window", k: int = 10, llm=None):
        self.memory_type = memory_type
        self.k = k
        self.llm = llm
        self.setup_memory()
        
    def setup_memory(self):
        """Configura el tipo de memoria según la configuración"""
        if self.memory_type == "buffer_window":
            self.memory = ConversationBufferWindowMemory(
                k=self.k,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        elif self.memory_type == "summary_buffer" and self.llm:
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                max_token_limit=2000
            )
        else:
            # Fallback to simple buffer
            self.memory = ConversationBufferWindowMemory(
                k=self.k,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
    
    def add_exchange(self, human_message: str, ai_message: str):
        """Añade un intercambio de conversación a la memoria"""
        self.memory.save_context(
            {"input": human_message},
            {"answer": ai_message}
        )
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Obtiene las variables de memoria actuales"""
        return self.memory.load_memory_variables({})
    
    def clear_memory(self):
        """Limpia la memoria"""
        self.memory.clear()
    
    def export_memory(self, filepath: str = None) -> str:
        """Exporta la memoria a un archivo JSON"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"exports/memory_export_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        memory_data = {
            "timestamp": datetime.now().isoformat(),
            "memory_type": self.memory_type,
            "buffer": self.memory.buffer_as_str,
            "messages": [
                {
                    "type": msg.type if hasattr(msg, 'type') else "unknown",
                    "content": msg.content if hasattr(msg, 'content') else str(msg)
                }
                for msg in getattr(self.memory, 'chat_memory', {}).get('messages', [])
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def get_conversation_summary(self) -> str:
        """Genera un resumen de la conversación actual"""
        if hasattr(self.memory, 'predict_new_summary'):
            # Para ConversationSummaryBufferMemory
            return self.memory.moving_summary_buffer
        else:
            # Para otros tipos de memoria, crear resumen básico
            buffer = self.memory.buffer_as_str
            if len(buffer) > 500:
                return buffer[:500] + "..."
            return buffer