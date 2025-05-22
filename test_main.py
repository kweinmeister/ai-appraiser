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
def test_estimate_value_success_with_image_uri_and_eur_currency(mock_generate_content):
    # Mock the first generate_content call (valuation)
    mock_response_with_search = MagicMock()
    mock_response_with_search.candidates = [MagicMock()]
    mock_response_with_search.candidates[0].content.parts = [
        MagicMock(
            text="Estimated value: EUR100, Reasoning: Looks good, Search URLs: [example.com]" # Currency in text
        )
    ]

    # Mock the second generate_content call (parsing)
    mock_response_for_parsing = MagicMock()
    # Ensure currency in the parsed output matches the requested "EUR"
    mock_response_for_parsing.text = '{"estimated_value": 100.0, "currency": "EUR", "reasoning": "Looks good", "search_urls": ["example.com"]}'

    # Set up side_effect for the two calls
    mock_generate_content.side_effect = [
        mock_response_with_search,
        mock_response_for_parsing,
    ]

    response = estimate_value(
        image_uri="gs://some_bucket/some_image.jpg",
        description="A test item",
        currency="EUR", # Requesting EUR
    )
    assert response.estimated_value == 100.0
    assert response.currency == "EUR" # Expecting EUR
    assert response.reasoning == "Looks good"
    assert response.search_urls == ["example.com"]

    # Check calls to generate_content
    assert mock_generate_content.call_count == 2

    # Call 1: Valuation
    valuation_call_args = mock_generate_content.call_args_list[0]
    valuation_contents = valuation_call_args[1]["contents"]
    assert "gs://some_bucket/some_image.jpg" in str(valuation_contents)
    assert "A test item" in str(valuation_contents)
    assert "EUR" in str(valuation_contents) # Check currency in valuation prompt
    assert "google_search" in str(valuation_call_args[1]["config"].tools)

    # Call 2: Parsing
    parsing_call_args = mock_generate_content.call_args_list[1]
    parsing_contents = parsing_call_args[1]["contents"]
    # The valuation_text from the first call (including "EUR100") should be in the parsing prompt
    assert "Estimated value: EUR100" in parsing_contents
    assert ValuationResponse.model_json_schema() in parsing_contents # Schema should be in prompt
    assert "EUR" in parsing_contents # Currency should be in parsing prompt
    assert parsing_call_args[1]["config"].response_mime_type == "application/json"


@patch("main.client.models.generate_content")
def test_estimate_value_success_with_image_data(mock_generate_content):
    # Mock the first generate_content call (valuation)
    mock_response_with_search = MagicMock()
    mock_response_with_search.candidates = [MagicMock()]
    mock_response_with_search.candidates[0].content.parts = [
        MagicMock(
            text="Estimated value: $100, Reasoning: Looks good, Search URLs: [example.com]" # Using $ for USD
        )
    ]

    # Mock the second generate_content call (parsing)
    mock_response_for_parsing = MagicMock()
    # Ensure currency in the parsed output matches the default (USD)
    mock_response_for_parsing.text = f'{{"estimated_value": 100.0, "currency": "{DEFAULT_CURRENCY}", "reasoning": "Looks good", "search_urls": ["example.com"]}}'

    # Set up side_effect for the two calls
    mock_generate_content.side_effect = [
        mock_response_with_search,
        mock_response_for_parsing,
    ]

    image_data = b"fake image data"
    response = estimate_value(
        image_uri=None,
        description="Test item with data",
        image_data=image_data,
        mime_type="image/jpeg",
        # currency will use DEFAULT_CURRENCY
    )
    assert response.estimated_value == 100.0
    assert response.currency == DEFAULT_CURRENCY
    assert response.reasoning == "Looks good"
    assert response.search_urls == ["example.com"]

    # Check calls to generate_content
    assert mock_generate_content.call_count == 2

    # Call 1: Valuation
    valuation_call_args = mock_generate_content.call_args_list[0]
    valuation_contents = valuation_call_args[1]["contents"]
    assert "Part.from_bytes(data=b'fake image data', mime_type='image/jpeg')" in str(valuation_contents)
    assert "Test item with data" in str(valuation_contents)
    assert DEFAULT_CURRENCY in str(valuation_contents) # Check currency in valuation prompt
    assert "google_search" in str(valuation_call_args[1]["config"].tools)

    # Call 2: Parsing
    parsing_call_args = mock_generate_content.call_args_list[1]
    parsing_contents = parsing_call_args[1]["contents"]
    assert "Estimated value: $100" in parsing_contents # From first call's text
    assert ValuationResponse.model_json_schema() in parsing_contents
    assert DEFAULT_CURRENCY in parsing_contents # Currency should be in parsing prompt
    assert parsing_call_args[1]["config"].response_mime_type == "application/json"


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
def test_estimate_value_malformed_valuation_text(mock_generate_content):
    # Mock the first generate_content call (valuation)
    mock_response_with_search = MagicMock()
    mock_response_with_search.candidates = [MagicMock()]
    mock_response_with_search.candidates[0].content.parts = [
        MagicMock(
            text="This is some malformed text without the expected fields."
        )
    ]

    # Mock the second generate_content call (parsing) to return malformed JSON
    mock_response_for_parsing = MagicMock()
    mock_response_for_parsing.text = '{"wrong_field": "some value"}'  # Malformed JSON

    mock_generate_content.side_effect = [
        mock_response_with_search,
        mock_response_for_parsing,
    ]

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg", description="A test item"
        )


@patch("main.client.models.generate_content")
def test_estimate_value_invalid_search_urls(mock_generate_content):
    # Mock the first generate_content call (valuation)
    mock_response_with_search = MagicMock()
    mock_response_with_search.candidates = [MagicMock()]
    mock_response_with_search.candidates[0].content.parts = [
        MagicMock(
            text="Estimated value: $100, Reasoning: Looks good, Search URLs: not-a-list"
        )
    ]

    # Mock the second generate_content call (parsing) to return JSON with invalid search_urls
    mock_response_for_parsing = MagicMock()
    mock_response_for_parsing.text = '{"estimated_value": 100.0, "currency": "USD", "reasoning": "Looks good", "search_urls": "not-a-list"}'

    mock_generate_content.side_effect = [
        mock_response_with_search,
        mock_response_for_parsing,
    ]

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg", description="A test item"
        )


@patch("main.client.models.generate_content")
def test_estimate_value_invalid_estimated_value(mock_generate_content):
    # Mock the first generate_content call (valuation)
    mock_response_with_search = MagicMock()
    mock_response_with_search.candidates = [MagicMock()]
    mock_response_with_search.candidates[0].content.parts = [
        MagicMock(
            text="Estimated value: not-a-number, Reasoning: Looks good, Search URLs: [example.com]"
        )
    ]

    # Mock the second generate_content call (parsing) to return JSON with invalid estimated_value
    mock_response_for_parsing = MagicMock()
    mock_response_for_parsing.text = '{"estimated_value": "not-a-number", "currency": "USD", "reasoning": "Looks good", "search_urls": ["example.com"]}'

    mock_generate_content.side_effect = [
        mock_response_with_search,
        mock_response_for_parsing,
    ]

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg", description="A test item"
        )


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

    import re
    # Assertions
    assert gcs_uri.startswith(f"gs://{mock_bucket.name}/")
    # Ensure the filename is in the format <timestamp>_test.jpg
    file_part = gcs_uri.split("/")[-1]
    assert re.match(r"\d{20}_test\.jpg", file_part)
    mock_bucket_factory.assert_called_once_with(STORAGE_BUCKET) # Assert factory was called with bucket name
    mock_bucket.blob.assert_called_once() # Assert blob method was called on the bucket instance
    # The blob name should be file_part
    assert mock_bucket.blob.call_args[0][0] == file_part
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
@patch("main.upload_image_to_gcs") # Still mock to ensure it's NOT called
def test_upload_image_no_storage_bucket(mock_upload_image_to_gcs_not_called):
    test_image_content = b"fake image content"
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.jpg", test_image_content, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "data_url": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
        "gcs_uri": None,
        "content_type": "image/jpeg",
    }
    mock_upload_image_to_gcs_not_called.assert_not_called()


@patch("main.STORAGE_BUCKET", "test-bucket") # Ensure STORAGE_BUCKET is set
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


def test_upload_image_endpoint_invalid_file_type():
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid image file type. Please upload an image."
    }


@patch("main.estimate_value")
def test_value_endpoint_success_with_image_data_and_gbp_currency(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=123.45,
        currency="GBP", # Test with GBP
        reasoning="Looks nice",
        search_urls=["http://example.com"],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item",
            "image_data": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_type": "image/jpeg",
            "currency": "GBP", # Pass GBP in form data
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "estimated_value": 123.45,
        "currency": "GBP", # Expect GBP in response
        "reasoning": "Looks nice",
        "search_urls": ["http://example.com"],
    }
    mock_estimate_value.assert_called_once()
    # Check that estimate_value was called with image_data and None for image_url
    args, kwargs = mock_estimate_value.call_args
    assert kwargs.get("image_data") is not None
    assert kwargs.get("image_url") is None
    assert kwargs.get("description") == "A test item"
    assert kwargs.get("mime_type") == "image/jpeg"
    assert kwargs.get("currency") == "GBP" # Ensure currency is passed to estimate_value


@patch("main.estimate_value")
def test_value_endpoint_success_with_image_url(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=123.45,
        currency="USD",
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
    mock_estimate_value.assert_called_once()
    # Check that estimate_value was called with image_url and None for image_data
    args, kwargs = mock_estimate_value.call_args
    assert kwargs.get("image_url") == "gs://test-bucket/test_image.jpg"
    assert kwargs.get("image_data") is None
    assert kwargs.get("description") == "A test item from URL"
    assert kwargs.get("mime_type") is None # mime_type is None as it's not from form for URL
    assert kwargs.get("currency") == DEFAULT_CURRENCY # Should default


@patch("main.estimate_value")
def test_value_endpoint_success_with_empty_url_and_image_data(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=50.0,
        currency="CAD",
        reasoning="Data with empty URL",
        search_urls=[],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item with empty URL",
            "image_url": "", # Empty image_url
            "image_data": "data:image/png;base64,ZmFrZSBpbWFnZSBkYXRh", # Fake png data
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
    mock_estimate_value.assert_called_once()
    args, kwargs = mock_estimate_value.call_args
    assert kwargs.get("image_url") == "" # image_url is passed as empty string
    assert kwargs.get("image_data") is not None
    assert kwargs.get("description") == "A test item with empty URL"
    assert kwargs.get("mime_type") == "image/png"
    assert kwargs.get("currency") == "CAD"


@patch("main.estimate_value")
def test_value_endpoint_success_with_both_url_and_data_url_prioritized(mock_estimate_value):
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=200.0,
        currency="JPY",
        reasoning="URL should be prioritized",
        search_urls=["http://example.com/both"],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item with both URL and data",
            "image_url": "gs://test-bucket/preferred_image.jpg",
            "image_data": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7", # Minimal GIF
            "content_type": "image/gif", # Relates to image_data
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
    mock_estimate_value.assert_called_once()
    args, kwargs = mock_estimate_value.call_args
    assert kwargs.get("image_url") == "gs://test-bucket/preferred_image.jpg" # image_url is used
    # image_data is decoded by the endpoint but estimate_value prioritizes image_url
    assert kwargs.get("image_data") is not None # It's decoded by the endpoint
    # However, the actual call to client.models.generate_content inside estimate_value
    # will use the image_uri (from image_url) not the image_data bytes.
    assert kwargs.get("description") == "A test item with both URL and data"
    assert kwargs.get("mime_type") == "image/gif" # This is from the form, associated with image_data
    assert kwargs.get("currency") == "JPY"


@patch("main.estimate_value")
def test_value_endpoint_fail_on_estimate_value_exception(mock_estimate_value):
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


def test_value_endpoint_fail_no_image_provided():
    response = client.post("/value", data={"description": "A test item"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Either image_url or image_data is required."}


def test_read_root_serves_html_with_default_currency():
    # Temporarily patch DEFAULT_CURRENCY for this test to ensure we check a specific value
    with patch("main.DEFAULT_CURRENCY", "XYZ"):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Check that the specific currency is embedded in the JS block
        assert "let defaultCurrency = 'XYZ';" in response.text


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
