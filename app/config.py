from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

def _getenv(key:str, default=None):
    v = os.getenv(key, default)

    if v is None:
        return None
    
    if isinstance(default, bool):
        return str(v).lower() in ("1", "true", "yes", "on")
    
    if isinstance(default, int):
        try: 
            return int(v)
        except:
            return default
        
    if isinstance(default, float):
        try:
            return float(v)
        except:
            return default
        
    return v

class Settings(BaseModel):
    model_name: str = _getenv("MODEL_NAME", "microsoft/phi-2")
    device_map: str | None = _getenv("DEVICE_MAP", "auto")
    max_new_tokens: int = _getenv("MAX_NEW_TOKENS", 32)
    temperature: float | None = _getenv("TEMPERATURE", None)
    top_p: float | None= _getenv("TOP_P", None)
    do_sample: bool = _getenv("DO_SAMPLE", False)
    port: int = _getenv("PORT", 8000)

settings = Settings()