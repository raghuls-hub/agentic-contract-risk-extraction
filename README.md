# Agentic Contract Risk Extraction

<<<<<<< HEAD
An end-to-end AI-powered legal risk analysis pipeline and dashboard for contracts. This system intelligently chunks legal documents, identifies high-risk clauses using hybrid machine learning, and quantitatively evaluates the negotiation balance between Buyers and Sellers using game-theoretic principles.

## 🚀 Features

- **Automated Document Processing (Agent 1)**: Ingests PDF contracts, extracts raw text, cleans formatting artifacts, and creates structural chunks.
- **Hybrid Risk Routing (Agent 2)**: Analyzes document chunks using `maticzav/legal-bert-embedding`, a pre-trained base risk classifier (`legal_risk_classifier.pkl`), and dynamic LLM threshold calibration to identify high-risk clauses.
- **Quantitative Game-Theoretic Analysis (Agent 3)**: Evaluates flagged clauses as a zero-sum negotiation game between Company A (Buyer) and Company B (Target). Computes mathematical advantage scores across six critical legal dimensions (financial indemnity, operational lockdown, IP vulnerability, etc.) to determine the dominant party and suggest compromises.
- **Modern Web Dashboard**: A Vite + React frontend providing a clean user interface to upload contracts, view the original PDF, and interactively explore the AI-generated risk scorecard and reasoning.

## 🏗️ Architecture

- **Backend**: Python-based pipeline orchestrated via `main.py` and served as a REST API using FastAPI (`server.py`).
- **AI / LLM Integration**: Uses the Groq API (powered by `llama-3.3-70b-versatile`) for high-speed, high-quality legal reasoning and calibration.
- **Frontend**: Built with React, Vite, and Lucide Icons for a responsive, interactive user experience.

## ⚙️ Prerequisites

- **Python 3.8+**
- **Node.js 18+**
- **Groq API Keys**: Required for LLM reasoning steps (Agents 2 & 3).

## 🛠️ Setup & Installation

### 1. Environment Configuration

In the root directory, create a `.env` file containing your Groq API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_API_KEY_AGENT2=your_groq_api_key_here

```

### 2. Backend (Python API)

1. Navigate to the project root directory.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI backend server:
   ```bash
   python server.py
   ```
   *The API will start running on `http://127.0.0.1:8000`.*

### 3. Frontend (React Dashboard)

1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install the Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The application UI will start running on `http://localhost:5173`.*

## 📖 Usage

1. Open your browser and navigate to `http://localhost:5173`.
2. Upload a standard M&A Contract in PDF format.
3. Click "Run AI Analysis".
4. The system will process the contract in the background through the 3-agent pipeline.
5. Review the results:
   - **Risk Clauses Panel**: See a summary of all identified risky clauses.
   - **Original PDF Tab**: View the source document.
   - **Scorecard Panel**: Analyze the mathematical balance, legal reasoning, and suggested compromise for the selected clause.

## 💻 CLI Orchestrator Usage

You can also run the full pipeline without the web interface using the monolithic orchestrator:

```bash
python main.py path/to/your/contract.pdf
```

The final quantitative analysis will be saved locally to `final_pipeline_analysis_output.json`.
=======
Agentic Contract Risk Extraction is an end-to-end legal document analysis system for M&A contracts. It ingests contract PDFs, splits them into structured chunks, scores each clause for risk, calibrates risk thresholds dynamically, and then performs a deeper clause-level analysis to explain why a clause is risky and how the parties could negotiate a compromise.

The project combines classical NLP, a trained risk classifier, sentence embeddings, and LLM-based reasoning to produce a multi-stage contract review pipeline.

## What the project does

1. **Extracts text from contract PDFs**
   - Reads legal documents page by page.
   - Cleans layout artifacts such as broken lines and hyphenated words.
   - Splits the document into overlapping chunks for downstream analysis.

2. **Routes clauses by risk**
   - Uses a Legal-BERT embedding model and a trained classifier to assign a baseline risk score to each chunk.
   - Selects high, medium, and low anchor clauses.
   - Uses an LLM to self-calibrate scores and adjust the final routing threshold.

3. **Performs clause-level risk analysis**
   - Sends flagged clauses to a quantitative M&A analysis engine.
   - Produces structured outputs including clause risk explanation, dominant party, compromise suggestion, and weighted mathematical balance.

4. **Serves results via API**
   - Exposes a FastAPI endpoint for uploading PDF files.
   - Returns extracted text, the encoded PDF, and structured analysis results for frontend use.

5. **Includes a React frontend**
   - The repository contains a frontend directory built with React + Vite.
   - It is intended for document upload and result presentation.

## Architecture overview

The repository is organized as a three-agent workflow:

### Agent 1: Document Chunking
- File: `agent_1.py`
- Purpose: Extract and clean PDF text, then split it into contract-sized chunks.
- Output: A list of chunk objects with `chunk_id`, `source_file`, and `chunk_text`.

### Agent 2: Risk Routing
- File: `agent_2.py`
- Purpose: Score chunks using embeddings + a classifier, then use an LLM for calibration and threshold adjustment.
- Output: Two groups:
  - `flagged` chunks
  - `safe` chunks

### Agent 3: Quantitative Risk Analysis
- File: `agent_3.py`
- Purpose: Analyze flagged clauses in depth using a structured LLM prompt and deterministic post-processing.
- Output: Structured analysis objects with weighted scores and computed balance metrics.

### Orchestrator
- File: `main.py`
- Purpose: Connect all agents into a single pipeline.
- Behavior: Loads models, processes a PDF, routes risky clauses, analyzes them, and saves the final JSON output.

### API Layer
- File: `server.py`
- Purpose: Wrap the pipeline in a FastAPI service.
- Endpoint: `POST /api/analyze`

## Technologies used

### Backend and AI
- **Python**
- **FastAPI** – API layer
- **Uvicorn** – ASGI server
- **PyPDF** – PDF text extraction
- **LangChain Text Splitters** – chunking logic
- **Sentence-Transformers** – Legal-BERT embeddings
- **Torch** – ML runtime dependency
- **Joblib** – loads the trained classifier pickle
- **Groq / OpenAI-compatible API** – LLM calls
- **Pydantic** – structured schemas and validation
- **python-dotenv** – environment variable loading
- **python-multipart** – file upload support

### Frontend
- **JavaScript**
- **React**
- **Vite**
- **CSS**
- **HTML**

### Model / artifact files
- `legal_risk_classifier.pkl` – trained risk classifier
- `final_pipeline_analysis_output.json` – pipeline output artifact

## Repository contents

- `main.py` – full pipeline orchestrator
- `server.py` – FastAPI server
- `agent_1.py` – PDF chunking and cleaning
- `agent_2.py` – classifier-based risk routing and calibration
- `agent_3.py` – quantitative clause analysis engine
- `frontend/` – React + Vite UI
- `input_documents/` – sample input PDFs
- `requirements.txt` – Python dependencies
- `test_orchestrator.py` – local pipeline test script

## How the pipeline works

### Step 1: Ingest and chunk the contract
The chunker extracts text from a PDF and normalizes layout artifacts before splitting the document into overlapping chunks.

### Step 2: Score and route clauses
Each chunk is embedded using Legal-BERT, scored by a classifier, and then adjusted by an LLM-based calibration step. Clauses are separated into safe and flagged sets.

### Step 3: Analyze flagged clauses in detail
Flagged clauses are passed to the quantitative analysis agent, which returns a structured breakdown of the legal risk, leverage, and likely negotiation compromise.

### Step 4: Persist and return results
The final result is serialized to JSON and also returned through the API for frontend consumption.

## Environment variables

The pipeline expects Groq API keys in the environment.

### Required for `main.py`
- `GROQ_API_KEY_AGENT2`
- `GROQ_API_KEY_AGENT3`

### Required for `agent_3.py` when used directly
- `GROQ_API_KEY`

## Installation

```bash
pip install -r requirements.txt
```

## Running the project

### 1. Run the full pipeline from the terminal

```bash
python main.py
```

You will be prompted to enter the path to a contract PDF.

### 2. Run Agent 1 only

```bash
python agent_1.py --input input_documents/your_contract.pdf --output cleaned_chunks.json
```

### 3. Run Agent 2 only

```bash
python agent_2.py --inputs cleaned_chunks.json --output routed_results.json --model legal_risk_classifier.pkl
```

### 4. Run Agent 3 only

```bash
python agent_3.py --input Agent-2-output.json --output results.json
```

### 5. Start the API server

```bash
python server.py
```

The API will run on `http://127.0.0.1:8000`.

### 6. Run the frontend

From the `frontend/` directory:

```bash
npm install
npm run dev
```

## API usage

### `POST /api/analyze`
Upload a PDF file and receive structured analysis output.

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "file=@input_documents/ma_contract_high_risk.pdf"
```

### Response shape
The response includes:
- `success`
- `full_text`
- `pdf_base64`
- `results`

## Output format

The final pipeline output is saved as JSON and contains:
- clause identifiers
- clause summaries
- risk explanations
- dominant party determination
- compromise suggestions
- weighted mathematical balance metrics

## Notes

- The project is designed specifically for M&A contract analysis.
- Agent 2 uses a classifier plus LLM calibration to reduce false negatives.
- Agent 3 computes balance metrics deterministically from per-parameter scores.
- The repository currently appears to be a working prototype with both CLI and API entry points.

## Sample workflow

```text
PDF contract
  → Agent 1 chunking
  → Agent 2 risk routing
  → Agent 3 quantitative analysis
  → JSON output / API response / frontend display
```

## License

No license file was found in the repository.

---

If you want, I can also turn this into a polished README with badges, project screenshots section, usage examples, and a cleaner professional tone.
>>>>>>> 113f353b0a8b41a45c407b12fbae0d2d4ace9117
