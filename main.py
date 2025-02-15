from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Part,
    Tool,
)
from pydantic import BaseModel
import os
import datetime
from mimetypes import guess_type
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL_ID = "gemini-2.0-flash-001"
CLOUD_STORAGE_BUCKET_NAME = os.environ.get("CLOUD_STORAGE_BUCKET_NAME")
DATASTORE_KIND = "valuation_transactions"  # For Datastore logging

storage_client = storage.Client(project=PROJECT_ID)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# --- Data Models ---
class ValuationRequest(BaseModel):
    description: str = Form(...)


class ValuationResponse(BaseModel):
    estimated_value: float
    currency: str
    reasoning: str
    search_urls: list[str]


# --- FastAPI App ---
app = FastAPI(
    title="Item Valuation API",
    description="API to estimate item value based on image and text.",
)

# --- CORS Configuration ---
origins = ["*"]  # Allow all origins for local development

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def upload_image_to_gcs(file: UploadFile) -> str:
    """Uploads an image file to Google Cloud Storage and returns the GCS URI."""
    bucket = storage_client.bucket(CLOUD_STORAGE_BUCKET_NAME)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    filename = f"{timestamp}_{file.filename}"
    blob = bucket.blob(filename)

    try:
        blob.upload_from_file(file.file, content_type=file.content_type)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail=f"Error uploading image to Cloud Storage: {e}"
        )


def estimate_value(image_uri: str, description: str) -> ValuationResponse:
    """Calls Gemini API with Search Tool to estimate item value, then parses the result into a ValuationResponse."""
    valuation_prompt = f"""You are a professional appraiser, adept at determining the value of items based on their description and market data.
Here is additional information provided by the user: {description}.
Your task is to estimate the item's fair market value.

To do this, you must use your built-in Search Tool to find comparable items currently for sale and recent auction results.
Analyze the item description, user information, and the search results carefully.

Provide a reasoned estimate of the item's value (or a price range) in USD.
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

    try:
        response_with_search = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                Part.from_uri(file_uri=image_uri, mime_type=guess_type(image_uri)[0]),
                valuation_prompt,
            ],
            config=config_with_search,
        )
        try:
            valuation_text = response_with_search.text
        except:
            valuation_text = "Error estimating value: no text response."
        print(valuation_text)

        # Second Gemini call to parse the valuation string into a ValuationResponse
        config_for_parsing = GenerateContentConfig(
            response_mime_type="application/json", response_schema=ValuationResponse
        )
        parsing_prompt = f"""Here is the valuation text: {valuation_text}
Your task is to parse this text into a JSON object that adheres to the ValuationResponse schema.
Provide detailed reasoning without linking that reasoning to the source information, such as 'based on the image'.
The ValuationResponse schema is: {ValuationResponse.schema_json()}
Ensure the JSON is valid and contains the estimated_value, reasoning, and search_urls fields."""

        response_for_parsing = client.models.generate_content(
            model=MODEL_ID, contents=parsing_prompt, config=config_for_parsing
        )
        valuation_response = response_for_parsing.text
        print(valuation_response)

        return ValuationResponse.model_validate_json(valuation_response)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error estimating value with search: {e}"
        )


# --- API Endpoints ---
@app.post("/value", response_model=ValuationResponse)
async def estimate_item_value(
    description: str = Form(...),
    image_url: str = Form(...),
):
    """Estimates the value of an item based on an image and text input."""

    if not image_url:
        raise HTTPException(status_code=400, detail="Image URL is required.")

    transaction_id = datetime.datetime.utcnow().strftime(
        "%Y%m%d%H%M%S%f"
    )  # Simple transaction ID

    try:
        # item_description = generate_item_description(image_url, description)
        # response_data = estimate_value_with_search(item_description, description)
        response_data = estimate_value(image_url, description)
        print(response_data)
        formatted_value = (
            f"{response_data.estimated_value:.2f}"  # Format to two decimal places
        )
        html_content = f"""
        <div class="bg-white p-4 rounded shadow-md">
            <p><strong>Estimated Value:</strong> ${formatted_value}</p>
            <br>
            <p class="text-sm">{response_data.reasoning}</p>"""

        if (
            response_data.search_urls
            and response_data.search_urls != ["N/A"]
            and any(response_data.search_urls)
        ):
            html_content += f"""
            <br>
            <p><strong>Sources:</strong></p>
            <ul class="text-sm list-disc list-inside ml-4">
                {"".join(f"<li><a href='{url}' target='_blank'>{url}</a></li>" for url in response_data.search_urls)}
            </ul>
            """

        html_content += """
        </div>
        """
        return HTMLResponse(content=html_content, media_type="text/html")

    except HTTPException as http_exc:
        print(http_exc)
        raise http_exc
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail="Internal server error during valuation."
        )


@app.post("/upload-image")
async def upload_image(image_file: UploadFile = File(...)):
    """Uploads an image, returns an HTML img tag to display it, and sets the image URL in a hidden field."""
    print(f"Incoming request: {image_file}")
    print(f"Image file filename: {image_file.filename}")
    print(f"Image file content type: {image_file.content_type}")

    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid image file type. Please upload an image."
        )

    try:
        print("Before upload to GCS")
        image_uri = upload_image_to_gcs(image_file)
        print("After upload to GCS")
        # Include setting the hidden input's value in the response
        return HTMLResponse(
            content=f"""
            <img src='{image_uri}' class='w-full'>
            <script>
                document.getElementById('image_url').value = '{image_uri}';
            </script>
        """
        )
    except HTTPException as http_exc:
        print(http_exc)
        raise http_exc
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error uploading image: {e}")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
