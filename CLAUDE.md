# CodeOptiFix2 Guidelines

## Build/Test/Lint Commands
- Run tests: `python -m unittest discover`
- Run single test: `python -m unittest tests.path.to.test_file.TestClass.test_method`
- Run linter: `pylint ./path/to/module`
- Format code: `black ./path/to/module`
- Type check: `mypy ./path/to/module`

## Code Style Guidelines
- Python 3.9+ compatible syntax
- Use type hints for all function parameters and return values
- Follow PEP 8 naming conventions (snake_case for variables/functions, CamelCase for classes)
- Group imports: standard library, third-party, local modules (alphabetically within groups)
- Line length: 100 characters max
- Document modules, classes, and functions with docstrings
- Error handling: use try/except blocks with specific exceptions
- Use f-strings for string formatting
- Include unit tests for all new functionality
- Log errors and significant operations with appropriate levels