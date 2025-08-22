import joblib
from typing import List, Any
import ml_trading.models.model
import os


class SingleModelSaveLoadMixin():
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        model: Any,
        ):
        self.model = model

    def save(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save the Random Forest model using joblib
        model_filename = f"{model_id}.pkl"
        joblib.dump(self.model, model_filename)
        print(f"Model saved to {model_filename}")
        self.save_metadata(model_id)

    @classmethod
    def load(cls, model_id: str):
        metadata = ml_trading.models.model.Model.load_metadata(model_id)
        # Load Random Forest model
        model_filename = f"{model_id}.pkl"
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
            
        model = joblib.load(model_filename)
        
        # Create and return RandomForestModel instance
        return cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            model=model
        )
