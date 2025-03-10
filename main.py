import base64
import datetime
import logging
import os
from mimetypes import guess_type

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.cloud import storage
from google.genai.types import GenerateContentConfig, GoogleSearch, Part, Tool
from pydantic import BaseModel, Field

# --- Configuration ---
logging.basicConfig(level=logging.INFO)

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.0-flash-001")
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET")
DEFAULT_CURRENCY = os.environ.get("CURRENCY", "usd")
if not STORAGE_BUCKET:
    logging.warning(
        "STORAGE_BUCKET environment variable not set. Image uploads to GCS will be skipped."
    )

storage_client = storage.Client(project=PROJECT_ID)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# --- Data Models ---
class ValuationRequest(BaseModel):
    description: str = Form(...)


class ValuationResponse(BaseModel):
    estimated_value: float
    currency: str = Field(DEFAULT_CURRENCY, description="Currency code (ISO 4217)")
    reasoning: str
    search_urls: list[str]


# --- FastAPI App ---
app = FastAPI(
    title="Item Valuation API",
    description="API to estimate item value based on image and text.",
)


# --- Helper Functions ---
def upload_image_to_gcs(file: UploadFile) -> str:
    """Uploads an image file to Google Cloud Storage and returns the GCS URI."""
    bucket = storage_client.bucket(STORAGE_BUCKET)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S%f")
    filename = f"{timestamp}_{file.filename}"
    blob = bucket.blob(filename)

    try:
        blob.upload_from_file(file.file, content_type=file.content_type)
        return f"gs://{STORAGE_BUCKET}/{filename}"
    except Exception as e:
        logging.error(f"Error uploading image to Cloud Storage: {e}")
        raise


def get_data_url(file: UploadFile, contents: bytes) -> str:
    """Creates a data URL for the image."""
    encoded_image = base64.b64encode(contents).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded_image}"


def estimate_value(
    image_uri: str | None,
    description: str,
    image_data: bytes | None = None,
    mime_type: str | None = None,
    currency: str = DEFAULT_CURRENCY,
) -> ValuationResponse:
    """
    Calls Gemini API with Search Tool to estimate item value, then parses the result into a ValuationResponse.
    """

    valuation_prompt = f"""You are a professional appraiser, adept at determining the value of items based on their description and market data.
Here is additional information provided by the user: {description}.
Your task is to estimate the item's fair market value.

To do this, you must use your built-in Search Tool to find comparable items currently for sale and recent auction results.
Analyze the item description, user information, and the search results carefully.

Provide a reasoned estimate of the item's value (or a price range) in {currency}.
Justify your estimate based on the condition of the item, its characteristics, and the market prices of similar items.
Consider details such as:
- Condition (e.g., new, used, excellent, poor)
- Branding (if any)
- Year or age (if known)
- Any other relevant characteristics that would help in determining its value.
Include the URLs of the most relevant search results you used to arrive at your valuation.

Return a text response only, not an executable code response.
"""
    google_search_tool = Tool(google_search=GoogleSearch())
    config_with_search = GenerateContentConfig(tools=[google_search_tool])

    if image_uri:
        response_with_search = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                Part.from_uri(file_uri=image_uri, mime_type=guess_type(image_uri)[0]),
                valuation_prompt,
            ],
            config=config_with_search,
        )
    elif image_data:
        response_with_search = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                Part.from_bytes(data=image_data, mime_type=mime_type),
                valuation_prompt,
            ],
            config=config_with_search,
        )
    else:
        raise ValueError("Must provide either image_uri or image_data")

    # Use final part of search results with answer
    valuation_text = "Error estimating value: no text response."
    for part in response_with_search.candidates[0].content.parts:
        if part.text:
            valuation_text = part.text

    # Second Gemini call to parse the valuation string into a ValuationResponse
    config_for_parsing = GenerateContentConfig(
        response_mime_type="application/json", response_schema=ValuationResponse
    )
    parsing_prompt = f"""Here is the valuation text: {valuation_text}
Your task is to parse this text into a JSON object that adheres to the ValuationResponse schema.
Provide detailed reasoning without linking that reasoning to the source information, such as 'based on the image'.
The ValuationResponse schema is: {ValuationResponse.model_json_schema()}
Ensure the JSON is valid and contains the estimated_value, currency (using ISO 4217 currency code): {currency}, reasoning, and search_urls fields."""
    response_for_parsing = client.models.generate_content(
        model=MODEL_ID, contents=parsing_prompt, config=config_for_parsing
    )
    valuation_response = response_for_parsing.text

    return ValuationResponse.model_validate_json(valuation_response)


# --- API Endpoints ---
@app.post("/upload-image")
async def upload_image(image_file: UploadFile = File(...)):
    """Uploads an image, returns a Data URL for preview, and stores the GCS URI."""
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid image file type. Please upload an image."
        )

    try:
        contents = await image_file.read()
        await image_file.seek(0)  # Reset for GCS upload
        image_uri = upload_image_to_gcs(image_file) if STORAGE_BUCKET else None
        data_url = get_data_url(image_file, contents)

        return JSONResponse(
            {
                "data_url": data_url,
                "gcs_uri": image_uri,
                "content_type": image_file.content_type,
            }
        )
    except Exception as e:
        logging.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {e}")


@app.post("/value", response_model=ValuationResponse)
async def estimate_item_value(
    description: str = Form(...),
    image_url: str = Form(None),
    image_data: str = Form(None),
    content_type: str = Form(None),
    currency: str = Form(DEFAULT_CURRENCY),
):
    """Estimates the value of an item based on an image and text input."""

    if not image_url and not image_data:
        raise HTTPException(
            status_code=400, detail="Either image_url or image_data is required."
        )

    try:
        response_data = estimate_value(
            image_url,
            description,
            image_data=(
                base64.b64decode(image_data.split(",", 1)[1]) if image_data else None
            ),
            mime_type=content_type,
            currency=currency,
        )
        return JSONResponse(content=response_data.model_dump())
    except Exception as e:
        logging.error(f"Internal server error in /value: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
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
