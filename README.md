# CodeOptiFix

CodeOptiFix is an AI-driven tool that analyzes your codebase to understand, assess, and propose optimizations.

## Features

- **Code Analysis**: Automatically analyzes code files to understand their purpose and structure
- **Quality Assessment**: Identifies potential issues, complexity problems, and areas for improvement
- **Optimization Proposals**: Generates improved versions of your code while preserving functionality
- **Multiple Language Support**: Works with Python, JavaScript, and C++ files
- **API Integration**: Uses DeepSeek's powerful AI models via their API
- **Apply Changes**: Apply suggested improvements with automatic backup for rollback
- **Focused Improvements**: Target specific areas like performance, readability, or security
- **Rollback Support**: Easily revert to original code if you don't like the changes

## Setup

1. Clone the repository
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Configure your API key:
   - Create a `.env` file in the project root directory
   - Add your DeepSeek API key: `DEEPSEEK_API_KEY=your_api_key_here`

## Usage

Basic usage:
```bash
python main.py
```

Analyze a specific directory:
```bash
python main.py --dir /path/to/your/code
```

Analyze a single file:
```bash
python main.py --file /path/to/your/file.py
```

Apply suggested changes (modifies original files with backup):
```bash
python main.py --apply
```

Save suggested changes as new version files (doesn't modify originals):
```bash
python main.py --save-versions
```

Analyze with focus on a specific improvement area:
```bash
python main.py --focus performance
```

Rollback changes for a specific file:
```bash
python main.py --rollback /path/to/your/file.py
```

Rollback all applied changes:
```bash
python main.py --rollback-all
```

## Configuration

The tool's behavior can be customized by modifying the `config.py` file:

- `SUPPORTED_EXTENSIONS`: File types to analyze (`.py`, `.js`, `.cpp` by default)
- `MODEL_TYPE`: Choose between "local" and "deepseek" (API) modes
- `MODEL_ID`: Specify which DeepSeek model to use
- `PROMPTS`: Customize the prompts used for different analysis stages

## Focus Areas

You can focus the analysis on specific aspects of your code:

- `performance` - Optimize for speed and efficiency
- `readability` - Improve code clarity and maintainability
- `security` - Find and fix security vulnerabilities
- `complexity` - Reduce code complexity
- `memory` - Improve memory usage
- `testability` - Make code more testable

## Project Structure

- `main.py`: Entry point for the application
- `cli.py`: Command-line interface handling
- `config.py`: Configuration settings
- `core/`: Core functionality modules
  - `model.py`: Language model integration
  - `analyzer.py`: Code analysis logic
  - `file_handler.py`: File operations

## Requirements

- Python 3.10+
- dotenv
- httpx (for API requests)
- vllm (optional, for local model support)

## License

MIT License