#!/usr/bin/env python
import argparse
import asyncio
import logging
import os
import setup_env # needed for the environment variables
import ml_trading.models.manager
import ml_trading.models.registry

def main():
    # Create parser and add arguments
    parser = argparse.ArgumentParser(description="Model management tool")
    parser.add_argument("--action", type=str, choices=["list", "upload", "download"], default="list", 
                        help="Action to perform: list, upload, or download")
    parser.add_argument("--model-id", type=str, help="Name of the model from registry to use")
    parser.add_argument("--model-class-id", type=str, help="Model class identifier (e.g., 'xgboost', 'lightgbm') to use for model identification")
    
    # Parse arguments
    args = parser.parse_args()

    # Create model manager
    model_manager = ml_trading.models.manager.ModelManager()
    
    # Select model if specified
    model_class = None
    if args.model_class_id:
        model_class = ml_trading.models.registry.get_model_by_label(args.model_class_id)
        if not model_class:
            print(f"Error: Model '{args.model_class_id}' not found in registry.")
            return
        print(f"Selected model class: {args.model_class_id}")
    
    # Handle different actions
    if args.action == "list":
        print("Registered models:")
        registered_models = ml_trading.models.registry.list_registered_model_labels()
        if registered_models:
            for model_label in registered_models:
                print(f"  - {model_label}")
        else:
            print("No models registered.")
            
        print("\nLocal models:")
        local_models_dir = ml_trading.models.manager.LOCAL_MODEL_DIR_BASE
        if os.path.exists(local_models_dir):
            local_models = [d for d in os.listdir(local_models_dir) 
                          if os.path.isdir(os.path.join(local_models_dir, d))]
            if local_models:
                for model_id in local_models:
                    print(f"  - {model_id}")
            else:
                print("No local models found.")
        else:
            print("Local models directory not found.")

    elif args.action in ["upload", "download"]:
        # Check required arguments for non-list actions
        if not args.model_id:
            print("Error: For upload/download actions, --model-id is required")
            return
        if not args.model_class_id:
            print("Error: For upload/download actions, --model-class-id is required")
            return
            
        if args.action == "upload":
            # Handle upload action
            print(f"Performing upload action")
            model_manager.upload_model(args.model_id, model_class)
            
        elif args.action == "download":
            # Handle download action
            print(f"Performing download action")
            model_manager.download_model(args.model_id, model_class)
     
if __name__ == "__main__":
    main() 
