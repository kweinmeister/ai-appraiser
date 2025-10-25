from __future__ import annotations

import base64
import datetime
import logging
import os
import pathlib
from contextlib import asynccontextmanager
from enum import Enum
from mimetypes import guess_type

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.cloud import storage
from google.genai.types import GenerateContentConfig, GoogleSearch, Part, Tool
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from werkzeug.utils import secure_filename

# --- Configuration ---
logging.basicConfig(level=logging.INFO)

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.5-flash")
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET")
DEFAULT_CURRENCY = os.environ.get(
    "CURRENCY",
    "USD",
)
if not STORAGE_BUCKET:
    logging.warning(
        "STORAGE_BUCKET environment variable not set. Image uploads to GCS will be skipped.",
    )


# The GCS object name limit is 1024 bytes. The timestamp prefix is 26 bytes.
# We truncate the user-provided filename to a safe length to avoid exceeding the limit.
GCS_FILENAME_MAX_LEN = 220


# --- Data Models ---
class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"


class ValuationResponse(BaseModel):
    estimated_value: float
    currency: Currency = Field(
        Currency(DEFAULT_CURRENCY),
        description="Currency code (ISO 4217)",
    )
    reasoning: str
    search_urls: list[str]


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the Google Cloud clients on startup and stores them in the app state."""
    app.state.storage_client = storage.Client(project=PROJECT_ID)
    app.state.client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    yield


# --- FastAPI App ---
app = FastAPI(
    title="Item Valuation API",
    description="API to estimate item value based on image and text.",
    lifespan=lifespan,
)


# --- Dependencies ---
def get_storage_client(request: Request) -> storage.Client:
    return request.app.state.storage_client


def get_genai_client(request: Request) -> genai.Client:
    return request.app.state.client


# --- Helper Functions ---
def upload_image_to_gcs(file: UploadFile, storage_client: storage.Client) -> str:
    """Uploads an image file to Google Cloud Storage and returns the GCS URI."""
    assert storage_client is not None
    bucket = storage_client.bucket(STORAGE_BUCKET)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S%f")
    # Ensure filename is not None before securing it
    filename_to_secure = file.filename or "unknown_file"
    safe_filename = secure_filename(filename_to_secure)
    # Truncate filename to prevent object names from exceeding GCS limits.
    truncated_filename = safe_filename[:GCS_FILENAME_MAX_LEN]
    filename = f"{timestamp}_{truncated_filename}"
    blob = bucket.blob(filename)

    try:
        blob.upload_from_file(file.file, content_type=file.content_type)
        return f"gs://{STORAGE_BUCKET}/{filename}"
    except Exception:
        logging.exception("Error uploading image to Cloud Storage")
        raise


def get_data_url(file: UploadFile, contents: bytes) -> str:
    """Creates a data URL for the image."""
    encoded_image = base64.b64encode(contents).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded_image}"


def estimate_value(
    *,
    image_uri: str | None,
    description: str,
    client: genai.Client,
    image_data: bytes | None = None,
    mime_type: str | None = None,
    currency: Currency = Currency(DEFAULT_CURRENCY),
) -> ValuationResponse:
    """Calls Gemini API with Search Tool to estimate item value, then parses the result into a ValuationResponse."""
    assert client is not None
    valuation_prompt = f"""You are a professional appraiser, adept at determining the value of items based on their description and market data.
Here is additional information provided by the user: {description}.
Your task is to estimate the item's fair market value.

To do this, you must use your built-in Search Tool to find comparable items currently for sale and recent auction results.
Analyze the item description, user information, and the search results carefully.

Provide a reasoned estimate of the item's value (or a price range) in {currency.value}.
Justify your estimate based on the condition of the item, its characteristics, and the market prices of similar items.
Consider details such as:
- Condition (e.g., new, used, excellent, poor)
- Branding (if any)
- Year or age (if known)
- Any other relevant characteristics that would help in determining its value.
Include the URLs of the most relevant search results you used to arrive at your valuation.

Return a text response only, not an executable code response.
"""
    config_with_search = GenerateContentConfig(
        tools=[Tool(google_search=GoogleSearch())],
    )

    if image_uri:
        response_with_search = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                Part.from_uri(
                    file_uri=image_uri,
                    mime_type=guess_type(image_uri)[0] or "image/jpeg",
                ),
                valuation_prompt,
            ],
            config=config_with_search,
        )
    elif image_data:
        response_with_search = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                Part.from_bytes(data=image_data, mime_type=mime_type or "image/jpeg"),
                valuation_prompt,
            ],
            config=config_with_search,
        )
    else:
        msg = "Must provide either image_uri or image_data"
        raise ValueError(msg)

    # Use final part of search results with answer
    valuation_text = None
    if (
        response_with_search
        and response_with_search.candidates
        and response_with_search.candidates[0].content
        and response_with_search.candidates[0].content.parts
    ):
        for part in response_with_search.candidates[0].content.parts:
            if part.text:
                valuation_text = part.text
                break

    if not valuation_text:
        msg = "Failed to get a text response from the valuation model."
        raise ValueError(msg)

    # Second Gemini call to parse the valuation string into a ValuationResponse
    config_for_parsing = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ValuationResponse,
    )
    parsing_prompt = f"""Here is the valuation text: {valuation_text}
Your task is to parse this text into a JSON object that adheres to the ValuationResponse schema.
Provide detailed reasoning without linking that reasoning to the source information, such as 'based on the image'.
The ValuationResponse schema is: {ValuationResponse.model_json_schema()}
Ensure the JSON is valid and contains the estimated_value, currency (using ISO 4217 currency code): {currency.value}, reasoning, and search_urls fields."""
    response_for_parsing = client.models.generate_content(
        model=MODEL_ID,
        contents=parsing_prompt,
        config=config_for_parsing,
    )
    valuation_response_text = (
        response_for_parsing.text if response_for_parsing else None
    )
    if not valuation_response_text:
        msg = "Failed to get a valid JSON response from the parsing model."
        raise ValueError(msg)

    validated_response = ValuationResponse.model_validate_json(valuation_response_text)
    if validated_response.currency != currency:
        logging.warning(
            "Model returned currency '%s' but '%s' was requested.",
            validated_response.currency.value,
            currency.value,
        )
    return validated_response


# --- API Endpoints ---
@app.post("/upload-image")
async def upload_image(
    image_file: Annotated[UploadFile, File()],
    storage_client: Annotated[storage.Client, Depends(get_storage_client)],
):
    """Uploads an image, returns a Data URL for preview, and stores the GCS URI."""
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file type. Please upload an image.",
        )

    try:
        contents = await image_file.read()
        await image_file.seek(0)  # Reset for GCS upload
        image_uri = (
            upload_image_to_gcs(image_file, storage_client) if STORAGE_BUCKET else None
        )
        data_url = get_data_url(image_file, contents)

        return JSONResponse(
            {
                "data_url": data_url,
                "gcs_uri": image_uri,
                "content_type": image_file.content_type,
            },
        )
    except Exception:
        logging.exception("Error uploading image")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while uploading the image.",
        )


@app.post("/value", response_model=ValuationResponse)
async def estimate_item_value(
    description: Annotated[str, Form()],
    image_url: Annotated[str | None, Form()] = None,
    image_data: Annotated[str | None, Form()] = None,
    content_type: Annotated[str | None, Form()] = None,
    currency: Annotated[Currency, Form()] = Currency(DEFAULT_CURRENCY),
    client: genai.Client = Depends(get_genai_client),
):
    """Estimates the value of an item based on an image and text input."""
    if not image_url and not image_data:
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image_data is required.",
        )

    try:
        decoded_image_data = None
        # Only decode image_data if image_url is not provided.
        if not image_url and image_data:
            try:
                # Split data URL and decode base64 content
                _, encoded_data = image_data.split(",", 1)
                decoded_image_data = base64.b64decode(encoded_data)
            except Exception as e:
                # Raise 400 on malformed image_data
                raise HTTPException(
                    status_code=400, detail="Invalid image_data format.",
                ) from e

        response_data = estimate_value(
            image_uri=image_url,
            description=description,
            client=client,
            image_data=decoded_image_data,
            mime_type=content_type,
            currency=currency,
        )
        return JSONResponse(content=response_data.model_dump())
    except Exception:
        logging.exception("Internal server error in /value")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during valuation.",
        )


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(pathlib.Path(__file__).parent / "index.html") as f:
        html_content = f.read()
    html_content = html_content.replace(
        "let defaultCurrency = 'USD';",
        f"let defaultCurrency = '{DEFAULT_CURRENCY}';",
    )

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
