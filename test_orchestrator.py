import tempfile
import os
from main import LegalPipelineOrchestrator

orchestrator = LegalPipelineOrchestrator()

with open(r"c:\Users\theba\OneDrive\Desktop\agentic-contract-risk-extraction\input_documents\ma_contract_high_risk.pdf", "rb") as f:
    pdf_bytes = f.read()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(pdf_bytes)
    tmp_pdf_path = tmp_file.name

print(f"Testing with temp file: {tmp_pdf_path}")
results = orchestrator.execute_pipeline(tmp_pdf_path)

print("Results length:", len(results) if results else 0)

os.remove(tmp_pdf_path)
