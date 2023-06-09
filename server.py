import io

from PIL import Image
from fastapi import FastAPI, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, Response

from decensor import decensor

app = FastAPI(title="deepcreampy-onnx-docker")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


def send_image(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return Response(content=image_bytes.getvalue(), media_type="image/png")


@app.get("/", include_in_schema=False)
async def route_index():
    return RedirectResponse("/docs")


@app.post("/deepcreampy-bar", summary="Decensor bars from an image.")
async def route_bar(image: bytes = File()) -> Response:
    image = Image.open(io.BytesIO(image))
    result = decensor(image, image, is_mosaic=False)
    return send_image(result)


@app.post("/deepcreampy-mosaic", summary="Decensor mosaics from an image.")
async def route_bar(image: bytes = File(), masked: bytes = File()) -> Response:
    image = Image.open(io.BytesIO(image))
    masked = Image.open(io.BytesIO(masked))
    result = decensor(image, masked, is_mosaic=True)
    return send_image(result)
