import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import (
    ValuationResponse,
    app,
    estimate_value,
    get_data_url,
    upload_image_to_gcs,
)


# --- Test Helper Functions ---
def test_get_data_url():
    # Create a custom mock for UploadFile
    file_content = b"fake image content"
    mock_file = MagicMock()
    mock_file.filename = "test.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.read.return_value = file_content
    contents = file_content
    data_url = get_data_url(mock_file, contents)
    assert data_url == "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50"


@patch("main.client.models.generate_content")
def test_estimate_value_with_image_uri(mock_generate_content):
    # Mock the first generate_content call (valuation)
    mock_response_with_search = MagicMock()
    mock_response_with_search.candidates = [MagicMock()]
    mock_response_with_search.candidates[0].content.parts = [
        MagicMock(
            text="Estimated value: $100, Reasoning: Looks good, Search URLs: [example.com]"
        )
    ]
    mock_generate_content.return_value = mock_response_with_search

    # Mock the second generate_content call (parsing)
    mock_response_for_parsing = MagicMock()
    mock_response_for_parsing.text = '{"estimated_value": 100.0, "currency": "USD", "reasoning": "Looks good", "search_urls": ["example.com"]}'
    mock_generate_content.return_value = mock_response_for_parsing

    response = estimate_value(
        image_uri="gs://some_bucket/some_image.jpg", description="A test item"
    )
    assert response.estimated_value == 100.0
    assert response.currency == "USD"
    assert response.reasoning == "Looks good"
    assert response.search_urls == ["example.com"]


@patch("main.client.models.generate_content")
def test_estimate_value_with_image_data(mock_generate_content):
    # Mock the first generate_content call (valuation)
    mock_response_with_search = MagicMock()
    mock_response_with_search.candidates = [MagicMock()]
    mock_response_with_search.candidates[0].content.parts = [
        MagicMock(
            text="Estimated value: $100, Reasoning: Looks good, Search URLs: [example.com]"
        )
    ]
    mock_generate_content.return_value = mock_response_with_search

    # Mock the second generate_content call (parsing)
    mock_response_for_parsing = MagicMock()
    mock_response_for_parsing.text = '{"estimated_value": 100.0, "currency": "USD", "reasoning": "Looks good", "search_urls": ["example.com"]}'
    mock_generate_content.return_value = mock_response_for_parsing

    image_data = b"fake image data"
    response = estimate_value(
        image_uri=None,
        description="Test",
        image_data=image_data,
        mime_type="image/jpeg",
    )
    assert response.estimated_value == 100.0
    assert response.currency == "USD"
    assert response.reasoning == "Looks good"
    assert response.search_urls == ["example.com"]


@patch("main.estimate_value")
def test_estimate_value_raises_exception_no_image(mock_estimate_value):
    mock_estimate_value.side_effect = ValueError(
        "Must provide either image_uri or image_data"
    )
    with pytest.raises(ValueError) as exc_info:
        estimate_value(
            image_uri=None, description="Test", image_data=None, mime_type=None
        )
    assert str(exc_info.value) == "Must provide either image_uri or image_data"


@patch("main.storage_client.bucket")
def test_upload_image_to_gcs(mock_bucket):
    # Mock Google Cloud Storage calls
    mock_blob = MagicMock()
    mock_bucket.return_value.blob.return_value = mock_blob

    # Create a custom mock for UploadFile
    file_content = b"fake image content"
    mock_file = MagicMock()
    mock_file.filename = "test.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.file = io.BytesIO(
        file_content
    )  # Still use BytesIO for the file-like object

    # Call the function
    gcs_uri = upload_image_to_gcs(mock_file)

    # Assertions
    assert gcs_uri.startswith("gs://")
    assert "test.jpg" in gcs_uri
    mock_bucket.assert_called_once()
    mock_bucket.return_value.blob.assert_called()
    mock_blob.upload_from_file.assert_called_once_with(
        mock_file.file, content_type="image/jpeg"
    )


client = TestClient(app)


@patch("main.STORAGE_BUCKET", "test-bucket")
@patch("main.upload_image_to_gcs")
def test_upload_image_success(mock_upload_image_to_gcs):
    mock_upload_image_to_gcs.return_value = "gs://test-bucket/test.jpg"
    test_image_content = b"fake image content"
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.jpg", test_image_content, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "data_url": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
        "gcs_uri": "gs://test-bucket/test.jpg",
        "content_type": "image/jpeg",
    }
    mock_upload_image_to_gcs.assert_called_once()


def test_upload_image_invalid_file_type():
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid image file type. Please upload an image."
    }


@patch("main.estimate_value")
def test_estimate_item_value_success(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=123.45,
        currency="USD",
        reasoning="Looks nice",
        search_urls=["http://example.com"],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item",
            "image_data": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_type": "image/jpeg",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "estimated_value": 123.45,
        "currency": "USD",
        "reasoning": "Looks nice",
        "search_urls": ["http://example.com"],
    }
    mock_estimate_value.assert_called_once()


def test_estimate_item_value_no_image():
    response = client.post("/value", data={"description": "A test item"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Either image_url or image_data is required."}


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
