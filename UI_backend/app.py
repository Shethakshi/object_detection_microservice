from fastapi import FastAPI, UploadFile, File
import httpx  # better async support than requests

app = FastAPI()

AI_BACKEND_URL = "http://ai_backend:5000/detect/"  # service name + internal port

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    files = {"file": (file.filename, await file.read(), file.content_type)}
    
    try:
        # Use httpx for better error reporting
        async with httpx.AsyncClient() as client:
            response = await client.post(AI_BACKEND_URL, files=files, timeout=30)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": "AI backend returned error", "details": e.response.text}
    except httpx.RequestError as e:
        return {"error": "Could not connect to AI backend", "details": str(e)}
    except Exception as e:
        return {"error": "Unexpected error", "details": str(e)}
