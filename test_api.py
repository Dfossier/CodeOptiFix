import structlog
import requests
import json
from typing import Dict, Any

structlog.configure(
    processors=[structlog.processors.JSONRenderer(indent=2, sort_keys=True)],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


def test_api_endpoint():
    url = "https://api.example.com/test"
    test_data = {"key": "value"}
    logger.info(
        "test_started", endpoint=url, test_data=test_data, test_name="test_api_endpoint"
    )
    try:
        response = requests.post(url, json=test_data)
        response_time = response.elapsed.total_seconds()
        logger.info(
            "api_response",
            status_code=response.status_code,
            response_time=response_time,
            response_body=response.json(),
            test_name="test_api_endpoint",
        )
        assert response.status_code == 200
        assert "expected_key" in response.json()
        logger.info(
            "test_passed",
            test_name="test_api_endpoint",
            assertions=["status_code_200", "response_contains_key"],
        )
    except Exception as e:
        logger.error(
            "test_failed",
            test_name="test_api_endpoint",
            error=str(e),
            error_type=type(e).__name__,
            traceback=str(e.__traceback__),
        )
        raise
