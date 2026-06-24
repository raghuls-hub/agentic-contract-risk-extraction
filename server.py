import os
import tempfile
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from main import LegalPipelineOrchestrator

app = FastAPI(title="Agentic Contract Risk API")

# Enable CORS for React frontend running typically on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the orchestrator globally so models are loaded once
orchestrator = LegalPipelineOrchestrator()

@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        return {"error": "File must be a PDF", "success": False}
        
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_pdf_path = tmp_file.name
            
        # 1. Extract Full Document Content
        full_text = orchestrator.chunker.extract_text_from_pdf(tmp_pdf_path)
        
        # Encode PDF for native rendering in frontend
        with open(tmp_pdf_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
            
        # 2. Run Pipeline
        results = orchestrator.execute_pipeline(tmp_pdf_path)
        
        # Ensure results are serializable
        serialized_results = []
        if results:
            for r in results:
                if hasattr(r, "model_dump"):
                    serialized_results.append(r.model_dump())
                elif hasattr(r, "__dict__"):
                    serialized_results.append(r.__dict__)
                else:
                    serialized_results.append(r)
                    
        return {
            "success": True,
            "full_text": full_text,
            "pdf_base64": pdf_base64,
            "results": serialized_results
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}
    finally:
        if 'tmp_pdf_path' in locals() and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
