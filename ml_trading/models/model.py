import numpy as np
from typing import List
import os
import json

class Model:
    def __init__(
            self, 
            model_name: str,
            columns: List[str],
            target: str,
            ):
        self.model_name = model_name
        self.columns = columns
        self.target = target

    # override this
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __repr__(self):
        return f"{self.model_name}\n"+\
            f"{len(self.columns)} columns:\n{self.columns}\n"+\
            f"target: {self.target}"

    def save_metadata(self, filename: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save metadata (model name, columns, target)
        metadata = {
            'model_name': self.model_name,
            'columns': self.columns,
            'target': self.target
        }
        
        metadata_filename = f"{filename}.meta.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f)
            
        print(f"Metadata saved to {metadata_filename}")

    @classmethod
    def load_metadata(cls, filename: str):
        # Load metadata
        metadata_filename = f"{filename}.meta.json"
        if not os.path.exists(metadata_filename):
            raise FileNotFoundError(f"Metadata file not found: {metadata_filename}")
            
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
            
        return metadata
