import base64
import io
from unittest.mock import MagicMock, patch
from freezegun import freeze_time

import pytest
from fastapi.testclient import TestClient

from main import (
    Currency,
    ValuationResponse,
    app,
    estimate_value,
    get_data_url,
    upload_image_to_gcs,
)


# --- Test Helper Functions ---
from main import DEFAULT_CURRENCY


def test_get_data_url_correct_format():
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
def test_estimate_value_image_uri_success_eur(mock_generate_content):
    # Simulate the final JSON output from the second (parsing) call
    mock_final_response = MagicMock()
    mock_final_response.text = '{"estimated_value": 100.0, "currency": "EUR", "reasoning": "Looks good", "search_urls": ["example.com"]}'

    # The first call can be a simple mock, we don't need to inspect its output anymore
    mock_valuation_response = MagicMock()
    mock_valuation_response.candidates = [MagicMock()]
    mock_valuation_response.candidates[0].content.parts = [MagicMock(text="Some text")]

    mock_generate_content.side_effect = [
        mock_valuation_response,
        mock_final_response,
    ]

    response = estimate_value(
        image_uri="gs://some_bucket/some_image.jpg",
        description="A test item",
        currency=Currency.EUR,
    )

    assert response.estimated_value == 100.0
    assert response.currency == Currency.EUR
    assert response.reasoning == "Looks good"
    assert response.search_urls == ["example.com"]

    # We still expect two calls, but we don't need to inspect the prompts as deeply
    assert mock_generate_content.call_count == 2


@patch("main.client.models.generate_content")
def test_estimate_value_image_data_success(mock_generate_content):
    # Simulate the final JSON output from the second (parsing) call
    mock_final_response = MagicMock()
    mock_final_response.text = f'{{"estimated_value": 100.0, "currency": "{DEFAULT_CURRENCY}", "reasoning": "Looks good", "search_urls": ["example.com"]}}'

    # The first call can be a simple mock
    mock_valuation_response = MagicMock()
    mock_valuation_response.candidates = [MagicMock()]
    mock_valuation_response.candidates[0].content.parts = [MagicMock(text="Some text")]

    mock_generate_content.side_effect = [
        mock_valuation_response,
        mock_final_response,
    ]

    image_data = b"fake image data"
    response = estimate_value(
        image_uri=None,
        description="Test item with data",
        image_data=image_data,
        mime_type="image/jpeg",
    )

    assert response.estimated_value == 100.0
    assert response.currency == Currency(DEFAULT_CURRENCY)
    assert response.reasoning == "Looks good"
    assert response.search_urls == ["example.com"]
    assert mock_generate_content.call_count == 2


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


@patch("main.client.models.generate_content")
def test_estimate_value_valuation_api_error(mock_generate_content):
    # Mock the first generate_content call (valuation) to raise an API error
    from google.api_core.exceptions import GoogleAPIError

    mock_generate_content.side_effect = GoogleAPIError("Gemini API error")

    with pytest.raises(GoogleAPIError) as exc_info:
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg", description="A test item"
        )
    assert str(exc_info.value) == "Gemini API error"


@patch("main.client.models.generate_content")
def test_estimate_value_parsing_api_error(mock_generate_content):
    # Mock the first generate_content call (valuation)
    mock_response_with_search = MagicMock()
    mock_response_with_search.candidates = [MagicMock()]
    mock_response_with_search.candidates[0].content.parts = [
        MagicMock(
            text="Estimated value: $100, Reasoning: Looks good, Search URLs: [example.com]"
        )
    ]

    # Mock the second generate_content call (parsing) to raise an API error
    from google.api_core.exceptions import GoogleAPIError

    mock_generate_content.side_effect = [
        mock_response_with_search,
        GoogleAPIError("Gemini API error during parsing"),
    ]

    with pytest.raises(GoogleAPIError) as exc_info:
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg", description="A test item"
        )
    assert str(exc_info.value) == "Gemini API error during parsing"


@patch("main.client.models.generate_content")
def test_estimate_value_malformed_json_response(mock_generate_content):
    # Simulate the final parsing call returning malformed JSON
    mock_final_response = MagicMock()
    mock_final_response.text = (
        '{"wrong_field": "some value", "currency": "USD"}'  # Missing required fields
    )

    # The first call can be a simple mock
    mock_valuation_response = MagicMock()
    mock_valuation_response.candidates = [MagicMock()]
    mock_valuation_response.candidates[0].content.parts = [MagicMock(text="Some text")]

    mock_generate_content.side_effect = [
        mock_valuation_response,
        mock_final_response,
    ]

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg", description="A test item"
        )


@patch("main.client.models.generate_content")
def test_estimate_value_invalid_search_urls(mock_generate_content):
    # Simulate the final parsing call returning invalid search_urls type
    mock_final_response = MagicMock()
    mock_final_response.text = '{"estimated_value": 100.0, "currency": "USD", "reasoning": "Looks good", "search_urls": "not-a-list"}'

    # The first call can be a simple mock
    mock_valuation_response = MagicMock()
    mock_valuation_response.candidates = [MagicMock()]
    mock_valuation_response.candidates[0].content.parts = [MagicMock(text="Some text")]

    mock_generate_content.side_effect = [
        mock_valuation_response,
        mock_final_response,
    ]

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg", description="A test item"
        )


@patch("main.client.models.generate_content")
def test_estimate_value_invalid_estimated_value(mock_generate_content):
    # Simulate the final parsing call returning invalid estimated_value type
    mock_final_response = MagicMock()
    mock_final_response.text = '{"estimated_value": "not-a-number", "currency": "USD", "reasoning": "Looks good", "search_urls": ["example.com"]}'

    # The first call can be a simple mock
    mock_valuation_response = MagicMock()
    mock_valuation_response.candidates = [MagicMock()]
    mock_valuation_response.candidates[0].content.parts = [MagicMock(text="Some text")]

    mock_generate_content.side_effect = [
        mock_valuation_response,
        mock_final_response,
    ]

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg", description="A test item"
        )


@freeze_time("2023-01-01 12:00:00")
@patch("main.STORAGE_BUCKET", "test-bucket")
@patch("main.storage_client.bucket")
def test_upload_image_to_gcs(mock_bucket_method):
    """
    Tests the image upload functionality to Google Cloud Storage.

    The `mock_bucket_method` is a patch on `main.storage_client.bucket`,
    which is called with the bucket name. It returns a mock bucket object
    that can be further inspected for calls to `blob` and `upload_from_file`.
    """
    mock_bucket = MagicMock()
    mock_bucket.name = "test-bucket"
    mock_bucket_method.return_value = mock_bucket

    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    file_content = b"fake image content"
    mock_file = MagicMock()
    mock_file.filename = "test.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.file = io.BytesIO(file_content)

    gcs_uri = upload_image_to_gcs(mock_file)

    # With frozen time, we can assert the exact filename
    expected_filename = "20230101120000000000_test.jpg"
    expected_uri = f"gs://{mock_bucket.name}/{expected_filename}"

    assert gcs_uri == expected_uri
    mock_bucket_method.assert_called_once_with("test-bucket")
    mock_bucket.blob.assert_called_once_with(expected_filename)
    mock_blob.upload_from_file.assert_called_once_with(
        mock_file.file, content_type="image/jpeg"
    )


client = TestClient(app)


@patch("main.STORAGE_BUCKET", "test-bucket")
@patch("main.upload_image_to_gcs")
def test_upload_image_endpoint_success_with_gcs(mock_upload_image_to_gcs):
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


@patch("main.STORAGE_BUCKET", None)
@patch("main.upload_image_to_gcs")  # Still mock to ensure it's NOT called
def test_upload_image_no_storage_bucket(mock_upload_image_to_gcs_not_called):
    test_image_content = b"fake image content"
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.jpg", test_image_content, "image/jpeg")},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["gcs_uri"] is None, (
        "gcs_uri should be None when STORAGE_BUCKET is not set"
    )
    assert (
        response_json["data_url"] == "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50"
    )
    assert response_json["content_type"] == "image/jpeg"
    mock_upload_image_to_gcs_not_called.assert_not_called()


@patch("main.STORAGE_BUCKET", "test-bucket")  # Ensure STORAGE_BUCKET is set
@patch("main.upload_image_to_gcs")
def test_upload_image_gcs_upload_fails(mock_upload_image_to_gcs):
    mock_upload_image_to_gcs.side_effect = Exception("GCS upload failed")
    test_image_content = b"fake image content"
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.jpg", test_image_content, "image/jpeg")},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Error uploading image: GCS upload failed"}
    mock_upload_image_to_gcs.assert_called_once()


def test_upload_image_invalid_type():
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid image file type. Please upload an image."
    }


@patch("main.estimate_value")
def test_value_endpoint_success_gbp(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=123.45,
        currency=Currency.GBP,  # Test with GBP
        reasoning="Looks nice",
        search_urls=["http://example.com"],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item",
            "image_data": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_type": "image/jpeg",
            "currency": "GBP",  # Pass GBP in form data
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "estimated_value": 123.45,
        "currency": "GBP",
        "reasoning": "Looks nice",
        "search_urls": ["http://example.com"],
    }
    mock_estimate_value.assert_called_once_with(
        image_uri=None,
        description="A test item",
        image_data=b"fake image content",
        mime_type="image/jpeg",
        currency=Currency.GBP,
    )


@patch("main.estimate_value")
def test_value_endpoint_success_image_url(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=123.45,
        currency=Currency.USD,
        reasoning="Looks nice from URL",
        search_urls=["http://example.com/url_image"],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item from URL",
            "image_url": "gs://test-bucket/test_image.jpg",
            # No image_data
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "estimated_value": 123.45,
        "currency": "USD",
        "reasoning": "Looks nice from URL",
        "search_urls": ["http://example.com/url_image"],
    }
    mock_estimate_value.assert_called_once_with(
        image_uri="gs://test-bucket/test_image.jpg",
        description="A test item from URL",
        image_data=None,
        mime_type=None,
        currency=Currency(DEFAULT_CURRENCY),
    )


@patch("main.estimate_value")
def test_value_endpoint_empty_url_prioritizes_image_data(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=50.0,
        currency=Currency.CAD,
        reasoning="Data with empty URL",
        search_urls=[],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item with empty URL",
            "image_url": "",  # Empty image_url
            "image_data": "data:image/png;base64,ZmFrZSBpbWFnZSBkYXRh",  # Fake png data
            "content_type": "image/png",
            "currency": "CAD",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "estimated_value": 50.0,
        "currency": "CAD",
        "reasoning": "Data with empty URL",
        "search_urls": [],
    }
    mock_estimate_value.assert_called_once_with(
        image_uri="",
        description="A test item with empty URL",
        image_data=b"fake image data",
        mime_type="image/png",
        currency=Currency.CAD,
    )


@patch("main.estimate_value")
def test_value_endpoint_both_inputs_prioritizes_url(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=200.0,
        currency=Currency.JPY,
        reasoning="URL should be prioritized",
        search_urls=["http://example.com/both"],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item with both URL and data",
            "image_url": "gs://test-bucket/preferred_image.jpg",
            "image_data": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",  # Minimal GIF
            "content_type": "image/gif",  # Relates to image_data
            "currency": "JPY",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "estimated_value": 200.0,
        "currency": "JPY",
        "reasoning": "URL should be prioritized",
        "search_urls": ["http://example.com/both"],
    }
    # Decode the image data from the data URL to ensure the bytes match exactly
    image_data_str = (
        "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    )
    image_data_bytes = base64.b64decode(image_data_str.split(",", 1)[1])

    mock_estimate_value.assert_called_once_with(
        image_uri="gs://test-bucket/preferred_image.jpg",
        description="A test item with both URL and data",
        image_data=image_data_bytes,
        mime_type="image/gif",
        currency=Currency.JPY,
    )


@patch("main.estimate_value")
def test_value_endpoint_estimate_value_exception(mock_estimate_value):
    mock_estimate_value.side_effect = Exception("Something went wrong")
    response = client.post(
        "/value",
        data={
            "description": "A test item that causes an error",
            "image_data": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_type": "image/jpeg",
        },
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Error: Something went wrong"}
    mock_estimate_value.assert_called_once()


def test_value_endpoint_no_image_provided():
    response = client.post("/value", data={"description": "A test item"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Either image_url or image_data is required."}


def _assert_html_contains_currency(response, currency):
    """Helper to check for currency in the root HTML response."""
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert f"let defaultCurrency = '{currency}';" in response.text


def test_read_root_serves_html_with_default_currency():
    # Temporarily patch DEFAULT_CURRENCY for this test to ensure we check a specific value
    with patch("main.DEFAULT_CURRENCY", "XYZ"):
        response = client.get("/")
        _assert_html_contains_currency(response, "XYZ")


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_value_endpoint_invalid_currency():
    response = client.post(
        "/value",
        data={
            "description": "A test item",
            "image_data": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_type": "image/jpeg",
            "currency": "INVALID_CURRENCY",
        },
    )
    assert (
        response.status_code == 422
    )  # Unprocessable Entity for Pydantic validation error
    assert "Input should be 'USD', 'EUR', 'GBP', 'JPY' or 'CAD'" in response.text


@patch("main.client.models.generate_content")
def test_value_endpoint_integration_style(mock_generate_content):
    """
    Tests the /value endpoint all the way down to the Gemini client mock,
    without mocking the intermediate estimate_value function.
    """
    # Mock the final JSON response from the parsing call
    mock_final_response = MagicMock()
    mock_final_response.text = '{"estimated_value": 99.99, "currency": "USD", "reasoning": "Integration test success", "search_urls": []}'

    # Mock the initial valuation call
    mock_valuation_response = MagicMock()
    mock_valuation_response.candidates = [MagicMock()]
    mock_valuation_response.candidates[0].content.parts = [
        MagicMock(text="Some valuation text")
    ]

    mock_generate_content.side_effect = [
        mock_valuation_response,
        mock_final_response,
    ]

    response = client.post(
        "/value",
        data={
            "description": "An integration test item",
            "image_data": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_type": "image/jpeg",
            "currency": "USD",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "estimated_value": 99.99,
        "currency": "USD",
        "reasoning": "Integration test success",
        "search_urls": [],
    }
    assert mock_generate_content.call_count == 2
