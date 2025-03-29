def validate_config(config: dict):
    """Validate configuration settings."""
    if not config.get('DEEPSEEK_API_KEY'):
        raise ValueError("DEEPSEEK_API_KEY is required in config")