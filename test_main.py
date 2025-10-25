import base64
import io
from unittest.mock import ANY, MagicMock, patch

import pytest
from freezegun import freeze_time
from google.api_core.exceptions import GoogleAPIError
from pydantic import ValidationError

from main import (
    DEFAULT_CURRENCY,
    Currency,
    ValuationResponse,
    estimate_value,
    get_data_url,
    upload_image_to_gcs,
)


def test_get_data_url_correct_format() -> None:
    # Create a custom mock for UploadFile
    file_content = b"fake image content"
    mock_file = MagicMock()
    mock_file.filename = "test.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.read.return_value = file_content
    contents = file_content
    data_url = get_data_url(mock_file, contents)
    assert data_url == "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50"


def test_estimate_value_image_uri_success_eur(
    mock_google_cloud_clients_and_app,
) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_models = mock_genai_client.models
    mock_models.generate_content.side_effect = [
        MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(parts=[MagicMock(text="Some valuation text")]),
                ),
            ],
        ),
        MagicMock(
            text='{"estimated_value": 100.0, "currency": "EUR", "reasoning": "Looks good", "search_urls": ["example.com"]}',
        ),
    ]

    response = estimate_value(
        image_uri="gs://some_bucket/some_image.jpg",
        description="A test item",
        currency=Currency.EUR,
        client=mock_genai_client,
    )

    assert response.estimated_value == 100.0
    assert response.currency == Currency.EUR
    assert response.reasoning == "Looks good"
    assert response.search_urls == ["example.com"]
    assert mock_models.generate_content.call_count == 2


def test_estimate_value_image_data_success(mock_google_cloud_clients_and_app) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_models = mock_genai_client.models
    mock_models.generate_content.side_effect = [
        MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(parts=[MagicMock(text="Some valuation text")]),
                ),
            ],
        ),
        MagicMock(
            text=f'{{"estimated_value": 100.0, "currency": "{DEFAULT_CURRENCY}", "reasoning": "Looks good", "search_urls": ["example.com"]}}',
        ),
    ]

    image_data = b"fake image data"
    response = estimate_value(
        image_uri=None,
        description="Test item with data",
        image_data=image_data,
        mime_type="image/jpeg",
        client=mock_genai_client,
    )

    assert response.estimated_value == 100.0
    assert response.currency == Currency(DEFAULT_CURRENCY)
    assert response.reasoning == "Looks good"
    assert response.search_urls == ["example.com"]
    assert mock_models.generate_content.call_count == 2


@patch("main.estimate_value")
def test_estimate_value_raises_exception_no_image(mock_estimate_value) -> None:
    mock_estimate_value.side_effect = ValueError(
        "Must provide either image_uri or image_data",
    )
    with pytest.raises(ValueError) as exc_info:
        estimate_value(
            image_uri=None,
            description="Test",
            image_data=None,
            mime_type=None,
            client=MagicMock(),
        )
    assert str(exc_info.value) == "Must provide either image_uri or image_data"


def test_estimate_value_valuation_api_error(mock_google_cloud_clients_and_app) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = GoogleAPIError(
        "Gemini API error",
    )

    with pytest.raises(GoogleAPIError) as exc_info:
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg",
            description="A test item",
            client=mock_genai_client,
        )
    assert str(exc_info.value) == "Gemini API error"


def test_estimate_value_parsing_api_error(mock_google_cloud_clients_and_app) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = [
        MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(parts=[MagicMock(text="Some valuation text")]),
                ),
            ],
        ),
        GoogleAPIError("Gemini API error during parsing"),
    ]

    with pytest.raises(GoogleAPIError) as exc_info:
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg",
            description="A test item",
            client=mock_genai_client,
        )
    assert str(exc_info.value) == "Gemini API error during parsing"


def test_estimate_value_malformed_json_response(
    mock_google_cloud_clients_and_app,
) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = [
        MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(parts=[MagicMock(text="Some valuation text")]),
                ),
            ],
        ),
        MagicMock(text='{"wrong_field": "some value", "currency": "USD"}'),
    ]

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg",
            description="A test item",
            client=mock_genai_client,
        )


def test_estimate_value_invalid_search_urls(mock_google_cloud_clients_and_app) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = [
        MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(parts=[MagicMock(text="Some valuation text")]),
                ),
            ],
        ),
        MagicMock(
            text='{"estimated_value": 100.0, "currency": "USD", "reasoning": "Looks good", "search_urls": "not-a-list"}',
        ),
    ]

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg",
            description="A test item",
            client=mock_genai_client,
        )


def test_estimate_value_invalid_estimated_value(
    mock_google_cloud_clients_and_app,
) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = [
        MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(parts=[MagicMock(text="Some valuation text")]),
                ),
            ],
        ),
        MagicMock(
            text='{"estimated_value": "not-a-number", "currency": "USD", "reasoning": "Looks good", "search_urls": ["example.com"]}',
        ),
    ]

    with pytest.raises(ValidationError):
        estimate_value(
            image_uri="gs://some_bucket/some_image.jpg",
            description="A test item",
            client=mock_genai_client,
        )


@freeze_time("2023-01-01 12:00:00")
def test_upload_image_to_gcs(mock_google_cloud_clients_and_app) -> None:
    """Tests the image upload functionality to Google Cloud Storage."""
    _, mock_storage_client, _ = mock_google_cloud_clients_and_app
    mock_bucket = mock_storage_client.bucket.return_value
    mock_bucket.name = "test-bucket"
    mock_blob = mock_bucket.blob.return_value

    with patch("main.STORAGE_BUCKET", "test-bucket"):
        file_content = b"fake image content"
        mock_file = MagicMock()
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.file = io.BytesIO(file_content)

        gcs_uri = upload_image_to_gcs(mock_file, mock_storage_client)

        expected_filename = "20230101120000000000_test.jpg"
        expected_uri = f"gs://{mock_bucket.name}/{expected_filename}"

        assert gcs_uri == expected_uri
        mock_storage_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with(expected_filename)
        mock_blob.upload_from_file.assert_called_once_with(
            mock_file.file,
            content_type="image/jpeg",
        )


@patch("main.STORAGE_BUCKET", "test-bucket")
@patch("main.upload_image_to_gcs")
def test_upload_image_endpoint_success_with_gcs(
    mock_upload_image_to_gcs,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
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
def test_upload_image_no_storage_bucket(
    mock_upload_image_to_gcs_not_called,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
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
def test_upload_image_gcs_upload_fails(
    mock_upload_image_to_gcs,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_upload_image_to_gcs.side_effect = Exception("GCS upload failed")
    test_image_content = b"fake image content"
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.jpg", test_image_content, "image/jpeg")},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Error uploading image: GCS upload failed"}
    mock_upload_image_to_gcs.assert_called_once()


def test_upload_image_invalid_type(mock_google_cloud_clients_and_app) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid image file type. Please upload an image.",
    }


@patch("main.estimate_value")
def test_value_endpoint_success_gbp(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = ValuationResponse(
        estimated_value=123.45,
        currency=Currency.GBP,
        reasoning="Looks nice",
        search_urls=["http://example.com"],
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item",
            "image_data": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_type": "image/jpeg",
            "currency": "GBP",
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
        client=ANY,
        image_data=b"fake image content",
        mime_type="image/jpeg",
        currency=Currency.GBP,
    )


@patch("main.estimate_value")
def test_value_endpoint_success_image_url(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
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
        client=ANY,
        image_data=None,
        mime_type=None,
        currency=Currency(DEFAULT_CURRENCY),
    )


@patch("main.estimate_value")
def test_value_endpoint_empty_url_prioritizes_image_data(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
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
            "image_url": "",
            "image_data": "data:image/png;base64,ZmFrZSBpbWFnZSBkYXRh",
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
        client=ANY,
        image_data=b"fake image data",
        mime_type="image/png",
        currency=Currency.CAD,
    )


@patch("main.estimate_value")
def test_value_endpoint_both_inputs_prioritizes_url(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
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
            "image_data": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
            "content_type": "image/gif",
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
        client=ANY,
        image_data=image_data_bytes,
        mime_type="image/gif",
        currency=Currency.JPY,
    )


@patch("main.estimate_value")
def test_value_endpoint_estimate_value_exception(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
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


def test_value_endpoint_no_image_provided(mock_google_cloud_clients_and_app) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    response = client.post("/value", data={"description": "A test item"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Either image_url or image_data is required."}


def _assert_html_contains_currency(response, currency) -> None:
    """Helper to check for currency in the root HTML response."""
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert f"let defaultCurrency = '{currency}';" in response.text


def test_read_root_serves_html_with_default_currency(
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    with patch("main.DEFAULT_CURRENCY", "XYZ"):
        response = client.get("/")
        _assert_html_contains_currency(response, "XYZ")


def test_health_check(mock_google_cloud_clients_and_app) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_value_endpoint_invalid_currency(mock_google_cloud_clients_and_app) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
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


def test_value_endpoint_integration_style(mock_google_cloud_clients_and_app) -> None:
    """Tests the /value endpoint all the way down to the Gemini client mock,
    without mocking the intermediate estimate_value function.
    """
    client, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_models = mock_genai_client.models
    mock_models.generate_content.side_effect = [
        MagicMock(
            candidates=[
                MagicMock(
                    content=MagicMock(parts=[MagicMock(text="Some valuation text")]),
                ),
            ],
        ),
        MagicMock(
            text='{"estimated_value": 99.99, "currency": "USD", "reasoning": "Integration test success", "search_urls": []}',
        ),
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
    assert mock_models.generate_content.call_count == 2
