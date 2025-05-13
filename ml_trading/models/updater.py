import os
import time
import logging
from dataclasses import dataclass
import ml_trading.models.registry
from ml_trading.models.manager import LOCAL_MODEL_DIR_BASE

@dataclass
class ModelUpdaterParams:
    """Parameters for the ModelUpdaterParams."""
    # model id is the name of the model as it appears in the file path
    model_id: str
    # model registry label is the name of a model class as it appears in the registry
    model_registry_label: str


class ModelUpdater:
    def __init__(self, param: ModelUpdaterParams):
        """
        Initialize the model updater.
        
        Args:
            param: ModelUpdaterParams
        """
        self.model_id = param.model_id
        self.model_registry_label = param.model_registry_label
        self.last_modified_time = self._get_file_modified_time()
        self.model_class = ml_trading.models.registry.get_model_by_label(self.model_id)
        self.model = None
        
        # Load the initial model if file exists
        if self.last_modified_time is not None:
            self._load_model()

    def _get_model_file_path(self):
        """Get the path to the model file."""
        return os.path.join(LOCAL_MODEL_DIR_BASE, self.model_id, self.model_id)
    
    def _get_file_modified_time(self):
        """Get the last modified time of the model files."""
        # Check for the metadata file which should always exist
        meta_file = f"{self._get_model_file_path()}.meta.json"
        if os.path.exists(meta_file):
            return os.path.getmtime(meta_file)
        return None
    
    def _load_model(self):
        """Load the model from file."""
        try:
            logging.info(f"Loading model {self.model_id} from {self._get_model_file_path()}")
            model_class = ml_trading.models.registry.get_model_by_label(self.model_registry_label)
            self.model = model_class.load(self._get_model_file_path())
            logging.info(f"Successfully loaded model {self.model_id}")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def check_for_updates(self):
        """Check if the model file has been updated and reload if necessary."""
        current_mod_time = self._get_file_modified_time()
        
        if current_mod_time is None:
            logging.warning(f"Model file {self._get_model_file_path()} does not exist")
            return False
        
        if self.last_modified_time is None or current_mod_time > self.last_modified_time:
            logging.info(f"Model file has been updated (last: {self.last_modified_time}, current: {current_mod_time})")
            success = self._load_model()
            if success:
                self.last_modified_time = current_mod_time
                return True
        
        return False
    