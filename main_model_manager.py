#!/usr/bin/env python
import argparse
import asyncio
import logging
import setup_env # needed for the environment variables
import ml_trading.models.manager
import ml_trading.models.registry

def main():
    # Create parser and add arguments
    parser = argparse.ArgumentParser(description="Model management tool")
    parser.add_argument("--action", type=str, choices=["list", "upload", "download"], default="list", 
                        help="Action to perform: list, upload, or download")
    parser.add_argument("--model_id", type=str, help="Name of the model from registry to use")
    
    # Parse arguments
    args = parser.parse_args()

    # Create model manager
    model_manager = ml_trading.models.manager.ModelManager()
    
    # Select model if specified
    model_class = None
    if args.model_id:
        model_class = ml_trading.models.registry.get_model_by_label(args.model_id)
        if not model_class:
            print(f"Error: Model '{args.model_id}' not found in registry.")
            return
        print(f"Selected model: {args.model_id}")
    
    # Handle different actions
    if args.action == "list":
        print("Registered models:")
        registered_models = ml_trading.models.registry.list_registered_model_labels()
        if registered_models:
            for model_label in registered_models:
                print(f"  - {model_label}")
        else:
            print("No models registered.")

    if args.action == "upload":
        # Handle upload action
        print(f"Performing upload action")
        if not args.model_id:
            print("Error: For upload action, --model_id is required")
            return
        # Implementation for model upload
        # You would need to create and train a model first
        model_manager.upload_model(args.model_id, model_class)
        
    elif args.action == "download":
        # Handle download action
        print(f"Performing download action")
        if not args.model_id:
            print("Error: For download action,  --model_id is required")
            return
        # Download model using the specified model class
        model_manager.download_model(args.model_id, model_class)
     
if __name__ == "__main__":
    main() 
