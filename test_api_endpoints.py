import os
import sys
import json
import logging
import traceback
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


def test_endpoint_response(
    endpoint: str, expected_status: int, auth_token: str = None
) -> bool:
    pass


def validate_response_schema(response: dict, schema: dict) -> bool:
    pass


def load_test_data(file_path: Path) -> dict:
    pass
