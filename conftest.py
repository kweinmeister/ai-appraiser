import pytest
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def mock_google_cloud_clients():
    # Mock storage.Client
    storage_patcher = patch("main.storage.Client", autospec=True)
    mock_storage_client = storage_patcher.start()

    # Mock genai.Client
    genai_patcher = patch("main.genai.Client", autospec=True)
    mock_genai_client = genai_patcher.start()

    yield mock_storage_client, mock_genai_client

    storage_patcher.stop()
    genai_patcher.stop()
