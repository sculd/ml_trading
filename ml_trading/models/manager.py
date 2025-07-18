"""
Model manager for saving and loading models to/from Google Cloud Storage.
    
Note: 
    - Requires the GOOGLE_APPLICATION_CREDENTIALS environment variable to be set
    - Default bucket is "ml_trading"
"""

import os
import tempfile
import glob
from typing import Optional, Dict, Any, Tuple, Union, Type, List
from google.cloud import storage
import logging

# Import the Model class directly
from ml_trading.models.model import Model as MLTradingModel

LOCAL_MODEL_DIR_BASE = "saved_models"

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
    def __init__(self, bucket_name: str = _bucket_name):
        """
        Initialize the ModelManager.
        
        Args:
            bucket_name: Name of the GCS bucket to use (default: _bucket_name)
            local_model_dir: Local directory for temporarily storing models
        """
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Create local directory if it doesn't exist
        os.makedirs(LOCAL_MODEL_DIR_BASE, exist_ok=True)
        
        logging.info(f"ModelManager initialized with bucket: {bucket_name}")

    def load_model_from_local(
            self, 
            model_id: str, 
            model_class: Type[MLTradingModel],
            ) -> Optional[MLTradingModel]:
        """
        Load a model from local directory.
        """
        try:
            local_path = os.path.join(LOCAL_MODEL_DIR_BASE, model_id, model_id)
            return model_class.load(local_path)
            
        except Exception as e:
            logging.error(f"Error loading model from local directory: {str(e)}")
            return None
    
    def save_model_to_local(
            self, 
            model_id: str, 
            model: MLTradingModel,
            ) -> str:
        """
        Save a model to local directory.
        
        Args:
            model_id: Path/name identifier for the model (e.g., 'xgboost_btc_1h')
            model: Model to save
            
        Returns:
            str: Path to the saved model
        """
        try:
            # Create local directory structure
            local_model_dir = os.path.join(LOCAL_MODEL_DIR_BASE, model_id)
            os.makedirs(local_model_dir, exist_ok=True)
            local_path = os.path.join(local_model_dir, model_id)
            model.save(local_path)
            return local_path

        except Exception as e:
            logging.error(f"Error downloading model from GCS: {str(e)}")
            return None

    def upload_model(
            self, 
            model_id: str,
            ) -> bool:
        """
        Upload all files from the local model directory to GCS.
        
        Args:
            model_id: Path/name identifier for the model (e.g., 'xgboost_btc_1h')
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            # Check if local model directory exists
            local_model_dir = os.path.join(LOCAL_MODEL_DIR_BASE, model_id)
            if not os.path.exists(local_model_dir):
                logging.error(f"Local model directory not found: {local_model_dir}")
                return False
            
            # Get all files in the model directory
            model_files = []
            for file_name in os.listdir(local_model_dir):
                file_path = os.path.join(local_model_dir, file_name)
                if os.path.isfile(file_path):
                    model_files.append(file_path)
            
            if not model_files:
                logging.error(f"No files found in model directory: {local_model_dir}")
                return False
            
            logging.info(f"Found {len(model_files)} files to upload: {[os.path.basename(f) for f in model_files]}")
            
            # Upload all files directly to GCS
            for file_path in model_files:
                # Extract the filename from the file path
                file_name = os.path.basename(file_path)
                
                # Create the GCS blob path
                blob_path = f"{model_id}/{file_name}"
                
                # Upload to GCS
                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(file_path)
                logging.info(f"Uploaded {file_path} to gs://{self.bucket_name}/{blob_path}")
            
            logging.info(f"Model successfully uploaded to gs://{self.bucket_name}/{model_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error uploading model to GCS: {str(e)}")
            return False

    def download_model(
            self, 
            model_id: str, 
            model_class: Type[MLTradingModel],
            ) -> Optional[MLTradingModel]:
        """
        Download a model from GCS.
        
        Args:
            model_id: Path/name identifier for the model (e.g., 'xgboost_btc_1h')
            model_class: Model class to use for instantiation
            
        Returns:
            Optional[MLTradingModel]: The loaded model if successful, None otherwise
        """
        try:
            # Create local directory structure
            local_model_dir = os.path.join(LOCAL_MODEL_DIR_BASE, model_id)
            os.makedirs(local_model_dir, exist_ok=True)
            
            # List all blobs with the model path prefix
            blobs = list(self.bucket.list_blobs(prefix=model_id))
            if not blobs:
                logging.error(f"No files found for model at gs://{self.bucket_name}/{model_id}")
                return None
            
            # Download all model files
            for blob in blobs:
                # Get file suffix from blob name
                file_suffix = os.path.basename(blob.name)
                local_file_path = f"{local_model_dir}/{file_suffix}"
                
                # Download the file
                blob.download_to_filename(local_file_path)
                logging.info(f"Downloaded {blob.name} to {local_file_path}")
            

            model = self.load_model_from_local(model_id, model_class)
            if model is not None:
                logging.info(f"Model successfully loaded using {model_class.__name__}.load()")
                return model
            
        except Exception as e:
            logging.error(f"Error downloading model from GCS: {str(e)}")
            return None

        return None
