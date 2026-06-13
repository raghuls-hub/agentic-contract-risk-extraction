# =====================================================================
# 1. EMERGENCE CRITICAL FIX (Prevents PyTorch circular import)
# =====================================================================
import sys
import torch
import torch.fx
import torch.ao.quantization

# =====================================================================
# 2. STANDARD IMPORTS
# =====================================================================
import os
import json
import joblib
import math
import logging
import argparse
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Setup module-level logger
logger = logging.getLogger("RiskRoutingAgent")


class RiskRoutingAgent:
    """
    APPROACH 3 — Score + Anchor Chunks with LLM Self-Scoring & Score Correction
    -----------------------------------------------------------------------------
    Production-grade agent designed to calibrate text classification scores 
    dynamically using an LLM to mitigate false-negative risks.
    """

    def __init__(
        self, 
        model_path: str = "legal_risk_classifier.pkl",
        calibration_model: str = "llama-3.3-70b-versatile",
        baseline_threshold: float = 0.3,
        mismatch_threshold: float = 0.20
    ):
        """
        Initializes the agent with configurable pipeline components.
        """
        self.baseline_threshold = baseline_threshold
        self.mismatch_threshold = mismatch_threshold
        self.calibration_model = calibration_model
        
        logger.info("Initializing RiskRoutingAgent...")
        logger.info("Loading Legal-BERT Embedding Model...")
        self.encoder = SentenceTransformer("maticzav/legal-bert-embedding")

        logger.info(f"Loading Base Classifier from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier model file not found at: {model_path}")
        self.base_classifier = joblib.load(model_path)

        logger.info("Authenticating Groq Client...")
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Environment variable 'GROQ_API_KEY' is missing.")

        self.llm_client = Groq(api_key=groq_api_key)
        logger.info("RiskRoutingAgent Initialization Complete.")

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------
    def process_document(self, document_chunks: list) -> dict:
        """
        Processes document chunks, scores them, runs LLM calibration,
        and routes them into flagged or safe buckets.
        """
        total_chunks = len(document_chunks)
        logger.info(f"Processing document: {total_chunks} chunks received.")

        if total_chunks == 0:
            return {"flagged": [], "safe": []}

        texts = [chunk["chunk_text"] for chunk in document_chunks]

        # ── STEP 1: Mathematical Base Scoring ─────────────────────────
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        probabilities = self.base_classifier.predict_proba(embeddings)[:, 1]

        scored_chunks = [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["chunk_text"],
                "score": float(prob),
                "corrected_score": float(prob),
                "original_index": i,
            }
            for i, (chunk, prob) in enumerate(zip(document_chunks, probabilities))
        ]
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        for x in scored_chunks:
            logger.debug(f"Chunk score tracking -> {x['chunk_id']}: {x['score']:.4f}")

        # ── STEP 2: Select Anchor Chunks ──────────────────────────────
        highest_chunk, mid_chunk, lowest_chunk = self._select_anchor_chunks(scored_chunks)
        logger.info(f"Selected Anchors -> High: {highest_chunk['chunk_id']}, Mid: {mid_chunk['chunk_id']}, Low: {lowest_chunk['chunk_id']}")

        # ── STEP 3: LLM Self-Scoring + Mismatch Analysis ──────────────
        llm_result = self._llm_self_score_and_compare(
            scored_chunks, highest_chunk, mid_chunk, lowest_chunk
        )

        mismatches = llm_result["mismatches"]
        raw_threshold = llm_result["raw_threshold"]

        # Log details about self-assigned scores vs classifier scores
        for cid, llm_sc in llm_result["llm_scores"].items():
            clf_sc = next(c["score"] for c in scored_chunks if c["chunk_id"] == cid)
            diff = round(llm_sc - clf_sc, 2)
            status = "MISMATCH" if abs(diff) >= self.mismatch_threshold else "OK"
            logger.info(f"Anchor comparison [{status}] -> {cid}: Classifier={clf_sc:.4f}, LLM={llm_sc:.4f}, Diff={diff:+.2f}")

        # ── STEP 4: Score Correction ───────────────────────────────────
        corrected_chunks = self._apply_score_correction(scored_chunks, mismatches)

        # ── STEP 5: Compute Final Threshold ───────────────────────────
        final_threshold = self._compute_final_threshold(
            scored_chunks, raw_threshold, llm_result["llm_scores"]
        )
        logger.info(f"Calculated dynamic system threshold: {final_threshold:.4f}")

        # ── STEP 6: Final Filter & Route ──────────────────────────────
        flagged_chunks, safe_chunks = [], []
        for chunk in corrected_chunks:
            if chunk["corrected_score"] >= final_threshold:
                flagged_chunks.append(chunk)
            else:
                safe_chunks.append(chunk)

        flagged_chunks.sort(key=lambda x: x["original_index"])
        safe_chunks.sort(key=lambda x: x["original_index"])

        return {"flagged": flagged_chunks, "safe": safe_chunks}

    # ------------------------------------------------------------------
    # PRIVATE: ANCHOR CHUNK SELECTION
    # ------------------------------------------------------------------
    def _select_anchor_chunks(self, scored_chunks: list) -> tuple:
        highest_chunk = scored_chunks[0]
        lowest_chunk = scored_chunks[-1]

        scores = [c["score"] for c in scored_chunks]
        median_score = sorted(scores)[len(scores) // 2]
        mid_chunk = min(scored_chunks, key=lambda c: abs(c["score"] - median_score))

        used_ids = {highest_chunk["chunk_id"], lowest_chunk["chunk_id"]}
        if mid_chunk["chunk_id"] in used_ids:
            candidates = [c for c in scored_chunks if c["chunk_id"] not in used_ids]
            if candidates:
                mid_chunk = min(candidates, key=lambda c: abs(c["score"] - median_score))
            else:
                mid_chunk = scored_chunks[len(scored_chunks) // 2]

        return highest_chunk, mid_chunk, lowest_chunk

    # ------------------------------------------------------------------
    # PRIVATE: LLM SELF-SCORING + MISMATCH ANALYSIS
    # ------------------------------------------------------------------
    def _llm_self_score_and_compare(
        self,
        scored_chunks: list,
        highest_chunk: dict,
        mid_chunk: dict,
        lowest_chunk: dict,
    ) -> dict:
        score_lines = "\n".join(f"  {c['chunk_id']}: {c['score']:.4f}" for c in scored_chunks)
        scores = [c["score"] for c in scored_chunks]
        
        prompt = f"""
You are a threshold calibration engine for a legal M&A contract risk detection system.

═══════════════════════════════════════════════════════════
HOW THESE SCORES WERE GENERATED
═══════════════════════════════════════════════════════════
Each score was produced by this pipeline:
  Legal-BERT Embeddings → Trained Risk Classifier → Risk Score (0.0 – 1.0)

- Score closer to 1.0 = classifier is highly confident this clause is RISKY
- Score closer to 0.0 = classifier is highly confident this clause is SAFE

═══════════════════════════════════════════════════════════
SCORE DISTRIBUTION STATISTICS
═══════════════════════════════════════════════════════════
  Total chunks : {len(scored_chunks)}
  Maximum score: {max(scores):.4f}
  Minimum score: {min(scores):.4f}
  Mean score   : {sum(scores) / len(scores):.4f}

═══════════════════════════════════════════════════════════
ALL CHUNK SCORES (chunk_id : classifier_risk_score)
═══════════════════════════════════════════════════════════
{score_lines}

═══════════════════════════════════════════════════════════
ANCHOR CLAUSE 1 — HIGHEST SCORED
[{highest_chunk['chunk_id']} | Classifier Score: {highest_chunk['score']:.4f}]
═══════════════════════════════════════════════════════════
The classifier considers this the MOST RISKY clause in the document.

Clause text:
"{highest_chunk['text']}"

YOUR TASK FOR ANCHOR 1:
Read this clause as a legal expert. Assign your own independent risk score between 0.0 and 1.0.

═══════════════════════════════════════════════════════════
ANCHOR CLAUSE 2 — MID SCORED (BOUNDARY REGION)
[{mid_chunk['chunk_id']} | Classifier Score: {mid_chunk['score']:.4f}]
═══════════════════════════════════════════════════════════
The classifier places this clause at the BOUNDARY between risky and safe.

Clause text:
"{mid_chunk['text']}"

YOUR TASK FOR ANCHOR 2:
Assign your own independent risk score.

═══════════════════════════════════════════════════════════
ANCHOR CLAUSE 3 — LOWEST SCORED
[{lowest_chunk['chunk_id']} | Classifier Score: {lowest_chunk['score']:.4f}]
═══════════════════════════════════════════════════════════
The classifier considers this the SAFEST clause in the document.

Clause text:
"{lowest_chunk['text']}"

YOUR TASK FOR ANCHOR 3:
Assign your own independent risk score.

═══════════════════════════════════════════════════════════
CRITICAL THRESHOLD DETERMINATION RULES
═══════════════════════════════════════════════════════════
When deciding the final value for "raw_threshold", you must follow these directives:
1. The standard system baseline threshold is {self.baseline_threshold}. 
2. You must give intentional importance/bias to the LEFT SIDE of the spectrum (closer to 0). 
   By skewing the threshold lower, you expand the safety net to ensure that subtle or borderline 
   risky clauses are NEVER accidentally leaked into the "safe area." A lower threshold maximizes risk catch rate.

═══════════════════════════════════════════════════════════
YOUR RESPONSE FORMAT
═══════════════════════════════════════════════════════════
Respond ONLY with a valid JSON object with exactly these keys:

  "scenario"      : "A", "B", or "C"
  "reasoning"     : one sentence explaining your threshold choice
  "llm_scores"    : object mapping chunk_id → your independent score (0.0–1.0)
  "mismatches"    : array of mismatch objects
  "raw_threshold" : your suggested threshold float BEFORE any score correction.

Example response:
{{
  "scenario": "B",
  "reasoning": "Biased threshold lower toward 0.35 to prioritize safety-net capture of borderline clauses.",
  "llm_scores": {{
    "{highest_chunk['chunk_id']}": 0.85,
    "{mid_chunk['chunk_id']}": 0.60,
    "{lowest_chunk['chunk_id']}": 0.20
  }},
  "mismatches": [],
  "raw_threshold": 0.35
}}
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.calibration_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            scenario = parsed.get("scenario", "B")
            reasoning = parsed.get("reasoning", "No reasoning provided.")
            llm_scores = parsed.get("llm_scores", {})
            mismatches = parsed.get("mismatches", [])
            raw_threshold = parsed.get("raw_threshold", 0.50)

            llm_scores = {k: max(0.0, min(1.0, float(v))) for k, v in llm_scores.items()}
            significant_mismatches = [
                m for m in mismatches if float(m.get("magnitude", 0.0)) >= self.mismatch_threshold
            ]

            return {
                "scenario": scenario,
                "reasoning": reasoning,
                "llm_scores": llm_scores,
                "mismatches": significant_mismatches,
                "raw_threshold": float(raw_threshold),
            }

        except Exception as e:
            logger.error(f"LLM Self-Scoring failed: {e}. Falling back to system defaults.")
            return {
                "scenario": "B",
                "reasoning": "LLM fallback triggered.",
                "llm_scores": {},
                "mismatches": [],
                "raw_threshold": 0.50,
            }

    # ------------------------------------------------------------------
    # PRIVATE: SCORE CORRECTION
    # ------------------------------------------------------------------
    def _apply_score_correction(self, scored_chunks: list, mismatches: list) -> list:
        if not mismatches:
            return scored_chunks

        corrected = []
        for chunk in scored_chunks:
            clf_score = chunk["score"]
            total_delta = 0.0
            total_weight = 0.0

            for mismatch in mismatches:
                anchor_clf_score = mismatch["classifier_score"]
                magnitude = float(mismatch["magnitude"])
                direction = mismatch["direction"]

                sigma = 0.15
                distance = abs(clf_score - anchor_clf_score)
                weight = math.exp(-(distance ** 2) / (2 * sigma ** 2))

                signed_delta = -magnitude if direction == "llm_lower" else +magnitude
                total_delta += signed_delta * weight
                total_weight += weight

            if total_weight > 0:
                avg_delta = total_delta / total_weight
                corrected_score = max(0.0, min(1.0, clf_score + avg_delta))
            else:
                corrected_score = clf_score

            corrected.append({**chunk, "corrected_score": corrected_score})

        return corrected

    # ------------------------------------------------------------------
    # PRIVATE: FINAL THRESHOLD COMPUTATION
    # ------------------------------------------------------------------
    def _compute_final_threshold(self, scored_chunks: list, raw_threshold: float, llm_scores: dict) -> float:
        if raw_threshold < self.baseline_threshold:
            threshold = self.baseline_threshold
            logger.debug(f"LLM raw threshold ({raw_threshold:.4f}) is below baseline. Enforcing baseline floor: {self.baseline_threshold}")
        else:
            threshold = raw_threshold

        computed_score = 0.0
        for cid, llm_sc in llm_scores.items():
            clf_sc = next((c["score"] for c in scored_chunks if c["chunk_id"] == cid), None)
            if clf_sc is not None:
                diff = llm_sc - clf_sc
                computed_score += round(diff, 2)

        if computed_score < 0:
            adjustment = abs(computed_score) / 2.0
            threshold -= adjustment
        else:
            threshold += computed_score

        return max(0.0, min(1.0, threshold))


# =====================================================================
# 3. STANDALONE TESTING AND DEBUGGING SUITE (Via CLI Arguments)
# =====================================================================
if __name__ == "__main__":
    # Configure logging for direct console feedback during local testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Set up argument parser to receive inputs/outputs from the terminal dynamically
    parser = argparse.ArgumentParser(description="Standalone CLI Runner for RiskRoutingAgent Calibration Pipeline.")
    parser.add_argument(
        "-i", "--inputs", 
        nargs="+", 
        required=True, 
        help="Space-separated paths to one or more input JSON data chunk files."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Base path output destination file name (e.g., path/to/output.json)."
    )
    parser.add_argument(
        "-m", "--model", 
        default="Project\Agent-2\legal_risk_classifier.pkl", 
        help="Path to the validation classifier pickle checkpoint."
    )

    args = parser.parse_args()
    
    for file in args.inputs:
        if not os.path.exists(file):
            print(f"❌ Error: {file} not found.")
            sys.exit(1)

        print(f"📥 Loading document chunks from {file}...")
        with open(file, "r", encoding="utf-8") as f:
            document_chunks = json.load(f)

        # Initialize agent dynamically using the terminal specified model parameter
        agent = RiskRoutingAgent(model_path=args.model)
        results = agent.process_document(document_chunks)

        print("================ FINAL OUTPUT SUMMARY ================")
        flagged_ids = [c["chunk_id"] for c in results["flagged"]]
        safe_ids = [c["chunk_id"] for c in results["safe"]]
        print(f"File processed: {os.path.basename(file)}")
        print(f"Flagged : {', '.join(flagged_ids) or 'None'} ({len(flagged_ids)} chunks)")
        print(f"Safe    : {', '.join(safe_ids)    or 'None'} ({len(safe_ids)} chunks)")
        print("======================================================")

        # Create suffix-split outputs so file loops don't destroy each other's processing data
        base_dir = os.path.dirname(args.output) or "."
        file_name_part = os.path.splitext(os.path.basename(file))[0]
        base_out_name = os.path.splitext(os.path.basename(args.output))[0]
        
        dynamic_output = os.path.join(base_dir, f"{base_out_name}_{file_name_part}.json")

        print(f"💾 Saving output to {dynamic_output}...")
        with open(dynamic_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print("✅ Done!\n")