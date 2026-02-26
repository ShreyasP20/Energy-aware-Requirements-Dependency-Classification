import tiktoken
import re
import json
from typing import List, Dict, Any  

def safe_token_count(text: str, model_name: str = "gpt-4") -> int:
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model_name)
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.split())

def chunk_list(data: List[Any], chunk_size: int):
    """Yield successive chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]