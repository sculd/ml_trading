import numpy as np
from typing import List

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __repr__(self):
        return f"{self.model_name}\n"+\
            f"{len(self.columns)} columns:\n{self.columns}\n"+\
            f"target: {self.target}"
