import numpy as np
from typing import List, Dict, Any
import os
import json

class Model:
    def __init__(
            self, 
            model_name: str,
            columns: List[str],
            target: str,
            other_params: Dict[str, Any] = None,
            ):
        self.model_name = model_name
        self.columns = columns
        self.target = target
        self.other_params = other_params or {}

    # override this
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __repr__(self):
        return f"{self.model_name}\n"+\
            f"{len(self.columns)} columns:\n{self.columns}\n"+\
            f"target: {self.target}"

    def save_metadata(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save metadata (model name, columns, target)
        metadata = {
            'model_name': self.model_name,
            'columns': self.columns,
            'target': self.target,
            'other_params': self.other_params
        }
        
        metadata_filename = f"{model_id}.meta.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f)
            
        print(f"Metadata saved to {metadata_filename}")

    @classmethod
    def load_metadata(cls, model_id: str):
        # Load metadata
        metadata_filename = f"{model_id}.meta.json"
        if not os.path.exists(metadata_filename):
            raise FileNotFoundError(f"Metadata file not found: {metadata_filename}")
            
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
            
        return metadata
