"""
main.py — Monolithic Legal M&A End-to-End Orchestrator

Workflow:
  1. Eagerly mounts models/keys.
  2. Interactively asks user for a PDF path via terminal input.
  3. Executes Agent 1 (Chunking) -> Agent 2 (Routing) -> Agent 3 (Quantitative Engine).
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# =====================================================================
# GLOBAL PIPELINE CONFIGURATION & CONSTANTS
# =====================================================================
BASE_CLASSIFIER_PATH = r"D:\VS_code\Python\Agentic AI\Lawc\Project\Agent-2\legal_risk_classifier.pkl"

ROUTING_CALIBRATION_MODEL = "llama-3.3-70b-versatile"
QUANTITATIVE_ANALYSIS_MODEL = "llama-3.3-70b-versatile"

CHUNK_WINDOW_SIZE = 600
CHUNK_SLIDING_OVERLAP = 150
BASELINE_RISK_THRESHOLD = 0.3
MISMATCH_DELTA_THRESHOLD = 0.20
AGENT_3_API_DELAY_SECONDS = 0.4

FINAL_JSON_OUTPUT_DESTINATION = "./final_pipeline_analysis_output.json"

# Load environment configuration variables early
load_dotenv()

# Setup tracking layout logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PipelineOrchestrator")

# Verify component file visibility before initialization hooks
try:
    from agent_1 import SmartDocumentChunker
    from agent_2 import RiskRoutingAgent
    from agent_3 import Agent3, Agent3Config, save_results
except ImportError as e:
    logger.critical(f"Dependency Resolution Error: Ensure agent-1.py, agent-2.py, and agent-3.py are in this path. Detail: {e}")
    sys.exit(1)


# =====================================================================
# CORE PIPELINE PIPELINE ORCHESTRATOR
# =====================================================================
class LegalPipelineOrchestrator:
    def __init__(self):
        """
        Condition Met: Core infrastructure, ML model loading, 
        embeddings models, and separate API allocations happen eagerly here.
        """
        print("\n" + "=" * 80)
        print("🚀 INITIALIZING LEGAL M&A PIPELINE CORE ENGINE")
        print("=" * 80)
        
        # 1. Gather isolated Groq keys from environment space
        self.key_agent2 = os.environ.get("GROQ_API_KEY_AGENT2")
        self.key_agent3 = os.environ.get("GROQ_API_KEY_AGENT3")
        
        if not self.key_agent2 or not self.key_agent3:
            raise EnvironmentError(
                "Missing multi-key configuration targets in your local .env file.\n"
                "Please configure 'GROQ_API_KEY_AGENT2' and 'GROQ_API_KEY_AGENT3'."
            )

        # 2. Pre-load Layer 1 (Document Fragmentation Core)
        print("[*] Configuring Layer 1 Splitting Windows...")
        self.chunker = SmartDocumentChunker(
            chunk_size=CHUNK_WINDOW_SIZE, 
            chunk_overlap=CHUNK_SLIDING_OVERLAP
        )

        # 3. Pre-load Layer 2 (Legal-BERT Weights + Base Classifier Pickle)
        print("[*] Instantiating Layer 2: Matrix Weights & Sentence Transformers...")
        # Inject Key 2 into the environment briefly so Agent 2 hooks into it seamlessly during setup
        os.environ["GROQ_API_KEY"] = self.key_agent2
        self.router = RiskRoutingAgent(
            model_path=BASE_CLASSIFIER_PATH,
            calibration_model=ROUTING_CALIBRATION_MODEL,
            baseline_threshold=BASELINE_RISK_THRESHOLD,
            mismatch_threshold=MISMATCH_DELTA_THRESHOLD
        )

        # 4. Pre-load Layer 3 (Quantitative Valuation Platform Engine)
        print("[*] Spinning Up Layer 3: Game-Theoretic Framework Structure...")
        agent3_config = Agent3Config(
            api_key=self.key_agent3,
            model_name=QUANTITATIVE_ANALYSIS_MODEL,
            request_delay_s=AGENT_3_API_DELAY_SECONDS
        )
        self.analyzer = Agent3(config=agent3_config)
        
        print("\n[✓] ALL MODELS EAGERLY LOADED. System ready for document path processing.\n")

    def execute_pipeline(self, target_pdf_path: str):
        """
        Executes sequential pipeline calculation blocking synchronously until finished.
        """
        if not os.path.exists(target_pdf_path):
            print(f"\n❌ Execution Error: Target file path does not exist: '{target_pdf_path}'")
            return

        print("\n" + "─" * 80)
        print(f"📁 Processing File Target: {os.path.basename(target_pdf_path)}")
        print("─" * 80)

        # -----------------------------------------------------------------
        # LAYER 1: Text Ingestion & Pattern Structural Fragmentation
        # -----------------------------------------------------------------
        print("\n👉 [RUNNING LAYER 1] Ingesting PDF Layout & Cleaning Text Elements...")
        raw_chunks = self.chunker.process_document(pdf_path=target_pdf_path)
        
        if not raw_chunks:
            print("⚠️ Layer 1 returned empty data layout blocks. Pipeline terminating.")
            return
        print(f"✨ [LAYER 1 SUCCESS] Extracted {len(raw_chunks)} normalized chunk text structures.")

        # -----------------------------------------------------------------
        # LAYER 2: Model Self-Scoring & Matrix Routing
        # -----------------------------------------------------------------
        print("\n👉 [RUNNING LAYER 2] Computing Legal-BERT Vectors & Executing LLM Self-Calibration...")
        # Ensure correct Groq token context is present before hitting router layer execution
        os.environ["GROQ_API_KEY"] = self.key_agent2
        routing_results = self.router.process_document(raw_chunks)
        
        flagged_elements = routing_results.get("flagged", [])
        safe_elements = routing_results.get("safe", [])
        
        print(f"✨ [LAYER 2 SUCCESS] Router Evaluation Complete: {len(flagged_elements)} Flagged Chunks | {len(safe_elements)} Safe Chunks.")
        
        if not flagged_elements:
            print("\n🎉 Run Concluded: Zero contract risk factors found above dynamic evaluation thresholds.")
            return

        # -----------------------------------------------------------------
        # LAYER 3: Microeconomic Game Evaluation & Payoff Balancing
        # -----------------------------------------------------------------
        print(f"\n👉 [RUNNING LAYER 3] Pushing {len(flagged_elements)} Risk Clauses to Game-Theoretic Framework Matrix...")
        
        # Connect the structured dictionary outputs safely:
        # Agent 2 labels processed text as 'text', Agent 3 ingests a dict list looking for 'text'
        formatted_flagged_inputs = []
        for element in flagged_elements:
            formatted_flagged_inputs.append({
                "chunk_id": element.get("chunk_id"),
                "text": element.get("text", "")
            })

        # verbose=True forces Agent 3 to print live updates and its final scorecard summary in terminal
        final_quantitative_payloads = self.analyzer.run(
            flagged_chunks=formatted_flagged_inputs, 
            verbose=True
        )

        # -----------------------------------------------------------------
        # DISK PERSISTENCE STORAGE
        # -----------------------------------------------------------------
        if final_quantitative_payloads:
            print(f"\n💾 Serializing complete quantitative results to layout: {FINAL_JSON_OUTPUT_DESTINATION}")
            save_results(final_quantitative_payloads, FINAL_JSON_OUTPUT_DESTINATION)
            print("\n" + "=" * 80)
            print("🎉 WORKFLOW ORCHESTRATION PIPELINE RUN RECOVERY COMPLETED")
            print("=" * 80 + "\n")
        else:
            print("⚠️ Zero output data blocks returned from quantitative parameter breakdown.")


# =====================================================================
# SYSTEM RUNTIME INTERACTIVE TERMINAL LOOP
# =====================================================================
if __name__ == "__main__":
    try:
        # Step 1: Eagerly instantiate core pipelines (Loads models, classifiers, transformers first)
        orchestrator = LegalPipelineOrchestrator()
        
        # Step 2: Interactive loop entry
        while True:
            print("💡 Enter the absolute or relative path to your M&A Contract PDF file.")
            print("   (Or type 'exit' or 'quit' to terminate the session context.)")
            user_input = input("\n📥 PDF File Path Entry: ").strip()
            
            if user_input.lower() in ["exit", "quit", ""]:
                print("\n👋 Closing Legal Workflow Core Orchestrator. Session Ended.\n")
                break
                
            # Clean string quotation marks wrapper if added via drag-drop mechanisms
            cleaned_path = user_input.strip("'\"")
            
            # Step 3: Trigger pipeline block (Calculations block sequentially)
            orchestrator.execute_pipeline(cleaned_path)
            print("\n" + "═" * 80 + "\n")
            
    except Exception as initialization_fault:
        logger.critical(f"Fatal System Pipeline Crash: {initialization_fault}", exc_info=True)
        sys.exit(1)