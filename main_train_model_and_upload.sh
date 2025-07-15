#!/bin/zsh

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${0:A}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_model_and_upload_$TIMESTAMP.log"

# Set up PATH for launchctl environment (more comprehensive than cron)
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:/opt/local/bin:$PATH"

# Set up environment for launchctl (doesn't inherit shell environment)
export HOME="${HOME:-/Users/$(whoami)}"
export USER="${USER:-$(whoami)}"
export SHELL="${SHELL:-/bin/zsh}"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to send notification (launchctl-friendly)
send_notification() {
    local status=$1
    local message=$2
    
    # Log the notification
    log_message "NOTIFICATION [$status]: $message"
    
    # For launchctl, we can use osascript but need to handle user context
    # Try GUI notification (works better with launchctl user agents)
    if command -v osascript >/dev/null 2>&1; then
        osascript -e "display notification \"$message\" with title \"Train Model and Upload\" subtitle \"$status\"" 2>/dev/null || true
    fi
    
    # Optional: Add email notification
    # echo "$message" | mail -s "Train Model and Upload - $status" your-email@example.com
    
    # Optional: Add webhook notification
    # curl -X POST -H "Content-Type: application/json" \
    #      -d "{\"status\":\"$status\",\"message\":\"$message\"}" \
    #      "YOUR_WEBHOOK_URL" 2>/dev/null || true
    
    # Optional: Write to system log (viewable with Console.app)
    logger -t "ml_trading" "[$status] $message"
}

# Function to validate prerequisites
validate_prerequisites() {
    log_message "Validating prerequisites..."
    
    # Check if required Python files exist
    if [[ ! -f "$SCRIPT_DIR/main_train_model.py" ]]; then
        log_message "ERROR: main_train_model.py not found"
        return 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/main_model_manager.py" ]]; then
        log_message "ERROR: main_model_manager.py not found"
        return 1
    fi
    
    # Check if setup_env.py exists (imported by the Python scripts)
    if [[ ! -f "$SCRIPT_DIR/setup_env.py" ]]; then
        log_message "ERROR: setup_env.py not found"
        return 1
    fi
    
    log_message "Prerequisites validated successfully"
    return 0
}

# Function to setup environment variables
setup_environment() {
    log_message "Setting up environment for launchctl..."
    log_message "Current USER: $USER"
    log_message "Current HOME: $HOME"
    log_message "Current PATH: $PATH"
    
    # For launchctl, explicitly set required environment variables
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    export PYTHONUNBUFFERED=1
    
    # Try to source environment from multiple sources
    local env_sourced=false
    
    # Try ~/.zshenv (most appropriate for launchctl)
    if [[ -f "$HOME/.zshenv" ]]; then
        source "$HOME/.zshenv" 2>/dev/null && env_sourced=true
        log_message "Sourced ~/.zshenv"
    fi
    
    # Try ~/.zshrc (for compatibility)
    if [[ -f "$HOME/.zshrc" ]]; then
        source "$HOME/.zshrc" 2>/dev/null && env_sourced=true
        log_message "Sourced ~/.zshrc"
    fi
    
    # Try a custom environment file (recommended for launchctl)
    # Note: Python scripts also load .env via setup_env.py, but shell may need some variables
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        source "$SCRIPT_DIR/.env" 2>/dev/null && env_sourced=true
        log_message "Sourced $SCRIPT_DIR/.env (Python scripts will also load this via setup_env.py)"
    fi
    
    if [[ "$env_sourced" == "false" ]]; then
        log_message "WARNING: No environment file sourced"
    fi
    
    log_message "Environment setup completed"
}

# Function to setup virtual environment
setup_virtual_environment() {
    log_message "Setting up virtual environment..."
    
    # Try multiple virtual environment sources
    local venv_activated=false
    
    # Try MARKET_VENV_BASE_DIR
    if [[ -n "$MARKET_VENV_BASE_DIR" && -f "$MARKET_VENV_BASE_DIR/bin/activate" ]]; then
        source "$MARKET_VENV_BASE_DIR/bin/activate"
        log_message "Activated virtual environment from $MARKET_VENV_BASE_DIR"
        venv_activated=true
    # Try local venv
    elif [[ -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/venv/bin/activate"
        log_message "Activated local virtual environment"
        venv_activated=true
    
    if [[ "$venv_activated" == "false" ]]; then
        log_message "WARNING: No virtual environment activated"
        log_message "Python location: $(which python)"
    else
        log_message "Virtual environment activated successfully"
        log_message "Python location: $(which python)"
        log_message "Python version: $(python --version)"
    fi
}

# Function to run training
run_training() {
    log_message "Starting model training..."
    
    # Run the training command
    python main_train_model.py \
        --target-column label_long_tp30_sl30_10m \
        --model-class-id random_forest_classification \
        --model-id rf_testrun \
        --resample-params close,0.1 \
        --time-period 100d
    
    log_message "Model training completed successfully"
}

# Function to upload model
upload_model() {
    log_message "Starting model upload..."
    
    python main_model_manager.py \
        --action upload \
        --model-id rf_testrun
    
    log_message "Model upload completed successfully"
}

# Main execution
main() {
    log_message "Starting train model and upload process (via launchctl)..."
    log_message "Script directory: $SCRIPT_DIR"
    log_message "Log file: $LOG_FILE"
    log_message "Process ID: $$"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Validate prerequisites
    if ! validate_prerequisites; then
        send_notification "Error" "Prerequisites validation failed"
        return 1
    fi
    
    # Setup environment
    setup_environment
    
    # Setup virtual environment
    setup_virtual_environment
    
    # Run training
    if ! run_training; then
        send_notification "Error" "Model training failed"
        return 1
    fi
    
    # Upload model
    if ! upload_model; then
        send_notification "Error" "Model upload failed"
        return 1
    fi
    
    log_message "Daily update process finished successfully"
    send_notification "Success" "Model training and upload completed successfully"
    return 0
}

# Run main function and capture any errors
if ! main; then
    log_message "❌ Train model and upload failed with critical error"
    send_notification "Error" "Train model and upload failed - check logs immediately"
    exit 1
fi

log_message "Script execution completed" 
