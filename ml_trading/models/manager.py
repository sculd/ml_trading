"""
Model manager for saving and loading models to/from Google Cloud Storage.
    
Note: 
    - Requires the GOOGLE_APPLICATION_CREDENTIALS environment variable to be set
    - Default bucket is "ml_trading"
"""

import os
import json
import tempfile
import glob
import shutil
from typing import Optional, Dict, Any, Tuple, Union, Type, List
from google.cloud import storage
import logging

# Import the Model class directly
from ml_trading.models.model import Model as MLTradingModel

# Default Google Cloud Storage bucket name
_bucket_name = "ml_trading"

class ModelManager:
    """
    Handles saving and loading ML models to/from Google Cloud Storage.
    
    This manager:
    - Automatically detects model types by examining saved files
    - Stores both the model and metadata files
    - Provides utility functions for listing and deleting models
    - Handles serialization and deserialization using model's built-in methods
    """
    def __init__(self, bucket_name: str = _bucket_name, local_model_dir: str = "saved_models"):
        """
        Initialize the ModelManager.
        
        Args:
            bucket_name: Name of the GCS bucket to use (default: _bucket_name)
            local_model_dir: Local directory for temporarily storing models
        """
        self.bucket_name = bucket_name
        self.local_model_dir = local_model_dir
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Create local directory if it doesn't exist
        os.makedirs(local_model_dir, exist_ok=True)
        
        logging.info(f"ModelManager initialized with bucket: {bucket_name}")
    
    def upload_model(self, 
                    model: MLTradingModel, 
                    model_path: str) -> bool:
        """
        Upload a model and its metadata to GCS.
        
        Uses temporary directory to detect model type by examining which files are created
        when model is saved.
        
        Args:
            model: The model object to upload (must be an instance of ml_trading.models.model.Model)
            model_path: Path/name identifier for the model (e.g., 'xgboost/btc_1h')
            
        Returns:
            bool: True if upload successful, False otherwise
            
        Example:
            manager = ModelManager()
            model = MyTrainedModel()  # A subclass of ml_trading.models.model.Model
            success = manager.upload_model(model, "xgboost/btc_1h")
        """
        try:
            # Ensure the model path doesn't start with a slash
            if model_path.startswith('/'):
                model_path = model_path[1:]
            
            # Ensure model_path ends with a slash
            if not model_path.endswith('/'):
                model_path = f"{model_path}/"
            
            # Create a temporary directory to determine model type
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model in temporary directory to see what files it creates
                temp_base_path = os.path.join(temp_dir, "temp_model")
                
                # Use model's save method without checking if it exists
                model.save(temp_base_path)
                
                # Get a list of all files created during save
                created_files = glob.glob(f"{temp_base_path}*")
                logging.info(f"Created files: {created_files}")
                
                # Upload all files directly from temp directory to GCS
                for file_path in glob.glob(f"{temp_base_path}*"):
                    # Extract the suffix from the original filename
                    file_suffix = file_path.replace(temp_base_path, '')
                    
                    # Create the GCS blob path
                    blob_path = f"{model_path}{file_suffix}"
                    
                    # Upload to GCS
                    blob = self.bucket.blob(blob_path)
                    blob.upload_from_filename(file_path)
                    logging.info(f"Uploaded {file_path} to gs://{self.bucket_name}/{blob_path}")
                
                logging.info(f"Model successfully uploaded to gs://{self.bucket_name}/{model_path}")
                return True
            
        except Exception as e:
            logging.error(f"Error uploading model to GCS: {str(e)}")
            return False
    
    def download_model(self, model_path: str, model_class: Type[MLTradingModel], local_path: Optional[str] = None) -> Optional[MLTradingModel]:
        """
        Download a model from GCS.
        
        Args:
            model_path: Path/name identifier for the model in GCS
            model_class: Model class to use for instantiation (required)
            local_path: Optional local path to save the model to
            
        Returns:
            Optional[MLTradingModel]: The loaded model if successful, None otherwise
        """
        try:
            # Ensure the model path doesn't start with a slash and ends with a slash
            if model_path.startswith('/'):
                model_path = model_path[1:]
            if not model_path.endswith('/'):
                model_path = f"{model_path}/"
                
            # Set local path if not provided
            local_path = local_path or self.local_model_dir
                
            # Create local directory structure
            local_model_dir = os.path.join(local_path, os.path.dirname(model_path))
            os.makedirs(local_model_dir, exist_ok=True)
            
            # Local base path for model files
            local_model_base = os.path.join(local_path, model_path)
            
            # List all blobs with the model path prefix
            blobs = list(self.bucket.list_blobs(prefix=model_path))
            if not blobs:
                logging.error(f"No files found for model at gs://{self.bucket_name}/{model_path}")
                return None
            
            # Download all model files
            for blob in blobs:
                # Get file suffix from blob name
                file_suffix = blob.name[len(model_path):]
                local_file_path = f"{local_model_base}{file_suffix}"
                
                # Download the file
                blob.download_to_filename(local_file_path)
                logging.info(f"Downloaded {blob.name} to {local_file_path}")
            
            # Try to load the model using model_class.load
            try:
                model = model_class.load(local_model_base)
                logging.info(f"Model successfully loaded using {model_class.__name__}.load()")
                return model
            except Exception as e:
                logging.error(f"Could not load model using {model_class.__name__}.load(): {str(e)}")
                return None
            
        except Exception as e:
            logging.error(f"Error downloading model from GCS: {str(e)}")
            return None
