# Agentic Contract Risk Extraction

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
