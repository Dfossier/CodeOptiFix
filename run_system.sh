#!/bin/bash

# Setup script for CodeOptiFix2 with DeepSeek integration

show_help() {
    echo "Usage: ./run_system.sh [OPTION] [ARGS]"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message and exit"
    echo "  --install           Install required dependencies and exit"
    echo "  --update            Update dependencies and exit"
    echo "  --test-api          Test DeepSeek API connection and exit"
    echo ""
    echo "Self-Improvement Loop options:"
    echo "  --goals N           Number of improvement goals to generate (default: 3)"
    echo "  --candidates N      Number of candidates per goal (default: 1)"
    echo "  --continuous        Run continuously instead of just once"
    echo "  --interval N        Seconds between cycles in continuous mode (default: 3600)"
    echo ""
    echo "Example:"
    echo "  ./run_system.sh --goals 5 --candidates 2    # Generate 5 goals with 2 candidates each"
    echo "  ./run_system.sh --test-api                  # Test the DeepSeek API connection"
    echo ""
    exit 0
}

# Check for help flag
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
fi

# Display a welcome message
echo "=== CodeOptiFix2 - Self-Improving AI Assistant ==="
echo "This script will help you set up and run the system."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

# Install dependencies if needed
if [ "$1" == "--install" ] || [ "$1" == "--update" ]; then
    echo "Installing dependencies..."
    python3 -m pip install -r requirements.txt
    echo "Dependencies installed."
    
    # If it was just an install/update command, exit after installing
    if [ "$1" == "--update" ]; then
        exit 0
    fi
fi

# Set up API configuration
echo "Setting up API configuration..."

# Check for .env file
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    echo "Found .env file. Loading environment variables..."
    # Use dotenv to load environment variables
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo "Loaded environment variables from .env file."
fi

# Ask for API key if not set (even after loading .env)
if [ -z "$DEEPSEEK_API_KEY" ]; then
    read -p "Enter your API key (leave empty to use demo mode): " api_key
    export DEEPSEEK_API_KEY="$api_key"
    
    if [ -z "$api_key" ]; then
        echo "Running in demo mode (no API key provided)"
    else
        echo "API key set."
    fi
else
    echo "Using API key from environment variables."
fi

# Ask for API URL if not set
if [ -z "$DEEPSEEK_API_URL" ]; then
    read -p "Enter API URL [default: https://api.deepseek.com]: " api_url
    if [ -z "$api_url" ]; then
        api_url="https://api.deepseek.com"
    fi
    # Make sure the URL is just the base domain without path segments
    if [[ $api_url == *"/v1"* || $api_url == *"/chat"* || $api_url == *"/completions"* ]]; then
        echo "Warning: The URL contains path segments that might cause issues."
        echo "The OpenAI client will automatically append '/v1/chat/completions'."
        read -p "Do you want to use just the base URL instead? (y/n) [default: y]: " fix_url
        if [ -z "$fix_url" ] || [ "$fix_url" == "y" ]; then
            # Extract just the domain part
            api_url=$(echo $api_url | awk -F/ '{print $1"//"$3}')
            echo "URL simplified to: $api_url"
        fi
    fi
    export DEEPSEEK_API_URL="$api_url"
    echo "API URL set to: $api_url"
fi

# Ask for LLM model if not set
if [ -z "$CODEOPTIFIX_LLM_MODEL" ]; then
    read -p "Enter LLM model name [default: deepseek-chat]: " model
    if [ -z "$model" ]; then
        model="deepseek-chat"
    fi
    export CODEOPTIFIX_LLM_MODEL="$model"
    echo "Model set to: $model"
fi

# Set other LLM parameters
export CODEOPTIFIX_LLM_TEMP="0.2"
export CODEOPTIFIX_LLM_MAX_TOKENS="4096"

# Check if we're testing the API connection
if [ "$1" == "--test-api" ]; then
    echo "Testing LLM API connection..."
    python3 test_api.py
    exit $?
fi

# Run the system with command-line arguments
echo "Running the self-improvement loop..."
python3 self_improvement_loop.py "$@"

echo "Execution completed."