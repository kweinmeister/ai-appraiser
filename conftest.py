from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import the app from main
from main import app


@pytest.fixture
def mock_google_cloud_clients_and_app():
    """A fixture that patches Google Cloud clients and provides a TestClient
    with a properly managed lifespan.
    """
    with patch("main.storage.Client", autospec=True) as MockStorageClient, patch(
        "main.genai.Client",
        autospec=True,
    ) as MockGenaiClient, TestClient(app) as test_client:
        yield test_client, MockStorageClient.return_value, MockGenaiClient.return_value
