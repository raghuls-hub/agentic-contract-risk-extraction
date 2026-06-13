"""
agent3.py — Quantitative M&A Risk Analysis Engine (Agent 3)
============================================================
IMPORT USAGE (from main pipeline):
    from agent3 import Agent3, Agent3Config, ExpandedTwoSidedRiskSchema, load_flagged_chunks

    config = Agent3Config(api_key="...", model_name="llama-3.3-70b-versatile")
    agent  = Agent3(config)
    results: list[ExpandedTwoSidedRiskSchema] = agent.run(flagged_chunks)

STANDALONE USAGE (CLI for testing/debugging):
    python agent3.py --input Agent-2-output.json
    python agent3.py --input Agent-2-output.json --output results.json
    python agent3.py --input Agent-2-output.json --output results.json --model llama-3.3-70b-versatile
    python agent3.py --input Agent-2-output.json --chunk CK-03          # single-chunk debug
    python agent3.py --input Agent-2-output.json --no-save              # print only, no file written
    python agent3.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging — library-safe: NullHandler by default, caller configures if needed
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

load_dotenv()

# =====================================================================
# PARAMETER WEIGHTS
# Reflects real-world M&A legal significance.
# Adjust per deal type without touching any other logic.
# =====================================================================
PARAMETER_WEIGHTS: Dict[str, float] = {
    "financial_indemnity_exposure":   0.25,  # Highest  — direct cash exposure
    "operational_lockdown":           0.20,  # High     — controls target's freedom
    "ip_asset_vulnerability":         0.20,  # High     — often the core deal asset
    "termination_remedial_asymmetry": 0.15,  # Medium   — escape valve dynamics
    "dispute_regulatory_burden":      0.10,  # Medium   — procedural leverage
    "consequential_damage_exposures": 0.10,  # Medium   — tail-risk multiplier
}
assert abs(sum(PARAMETER_WEIGHTS.values()) - 1.0) < 1e-9, "PARAMETER_WEIGHTS must sum to 1.0"

# =====================================================================
# 1. DATA MODELS
# =====================================================================

class AdvancedCompanyScoreCard(BaseModel):
    """Per-party advantage scores across 6 M&A risk dimensions."""

    financial_indemnity_exposure: int = Field(
        ...,
        description=(
            "Score 0-5: ADVANTAGE this party gains from indemnity caps, baskets, survival limits. "
            "5=fully protected/uncapped upside. 0=fully exposed, no benefit."
        ),
    )
    operational_lockdown: int = Field(
        ...,
        description=(
            "Score 0-5: CONTROL or FREEDOM this party gains from operating covenants. "
            "5=maximum control over the other party or full freedom to operate. 0=fully restricted."
        ),
    )
    ip_asset_vulnerability: int = Field(
        ...,
        description=(
            "Score 0-5: IP OWNERSHIP or PROTECTION this party gains. "
            "5=acquires full IP rights. 0=loses all IP rights."
        ),
    )
    termination_remedial_asymmetry: int = Field(
        ...,
        description=(
            "Score 0-5: TERMINATION POWER or REMEDY PROTECTION this party holds. "
            "5=can exit freely and collect fees. 0=locked in with no remedy."
        ),
    )
    dispute_regulatory_burden: int = Field(
        ...,
        description=(
            "Score 0-5: How FAVORABLE dispute resolution terms are for this party. "
            "5=home forum, full fee recovery, full control. 0=hostile forum, fee exposure."
        ),
    )
    consequential_damage_exposures: int = Field(
        ...,
        description=(
            "Score 0-5: DAMAGE RECOVERY RIGHTS this party retains. "
            "5=can claim all consequential/lost-profit damages. 0=waived all damage rights."
        ),
    )
    advantages_gained: str = Field(
        ...,
        description="Specific leverage and structural protections this party gains from this clause.",
    )

    @field_validator(
        "financial_indemnity_exposure",
        "operational_lockdown",
        "ip_asset_vulnerability",
        "termination_remedial_asymmetry",
        "dispute_regulatory_burden",
        "consequential_damage_exposures",
        mode="before",
    )
    @classmethod
    def clamp_score(cls, v: object) -> int:
        """Silently clamp LLM-returned values to the valid [0, 5] range."""
        return max(0, min(5, int(v)))

    def weighted_score(self) -> float:
        """
        Compute a single weighted advantage score normalised to [0, 100].

        Formula
        -------
        raw = Σ (param_score × param_weight)   →  [0.0, 5.0]
        score = raw × 20                        →  [0.0, 100.0]
        """
        raw = (
            self.financial_indemnity_exposure   * PARAMETER_WEIGHTS["financial_indemnity_exposure"]
            + self.operational_lockdown         * PARAMETER_WEIGHTS["operational_lockdown"]
            + self.ip_asset_vulnerability       * PARAMETER_WEIGHTS["ip_asset_vulnerability"]
            + self.termination_remedial_asymmetry * PARAMETER_WEIGHTS["termination_remedial_asymmetry"]
            + self.dispute_regulatory_burden    * PARAMETER_WEIGHTS["dispute_regulatory_burden"]
            + self.consequential_damage_exposures * PARAMETER_WEIGHTS["consequential_damage_exposures"]
        )
        return round(raw * 20, 4)


class LLMRiskPayload(BaseModel):
    """
    Fields returned by the LLM.
    mathematical_balance is intentionally absent — it is computed
    deterministically in ExpandedTwoSidedRiskSchema.from_llm_payload().
    """

    clause_causing_risk: str
    reason_for_risk: str
    clause_summary: str
    dominant_party: str
    company_a_buyer: AdvancedCompanyScoreCard
    company_b_target: AdvancedCompanyScoreCard
    suggested_compromise: str


class ExpandedTwoSidedRiskSchema(LLMRiskPayload):
    """
    Full analysis result including the deterministically computed
    mathematical_balance. This is the public return type of Agent3.run().
    """

    chunk_id: str = ""                                        # injected after LLM call
    mathematical_balance: Dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_llm_payload(
        cls,
        payload: LLMRiskPayload,
        chunk_id: str = "",
    ) -> "ExpandedTwoSidedRiskSchema":
        """
        Build the full result by computing mathematical_balance from the
        LLM's per-parameter scores — never from self-reported LLM estimates.

        Why deterministic?
        ------------------
        Asking the LLM to self-report percentages produces values that drift
        from its own parameter scores due to hallucination. Deriving the
        balance from the scored parameters guarantees:
          (a) mathematical consistency with the per-parameter scores
          (b) reproducibility given the same scores
          (c) freedom from float rounding / estimation drift

        Formula
        -------
        score_X   = party.weighted_score()              →  [0, 100]
        total     = score_a + score_b
        fav_a_pct = score_a / total × 100               (50 / 50 when total == 0)
        fav_b_pct = 100 − fav_a_pct                     (guaranteed exact sum)
        delta     = |score_a − score_b|
        """
        score_a = payload.company_a_buyer.weighted_score()
        score_b = payload.company_b_target.weighted_score()
        total_pool = score_a + score_b

        if total_pool == 0:
            fav_a = fav_b = 50.0
        else:
            fav_a = round(score_a / total_pool * 100, 2)
            fav_b = round(100 - fav_a, 2)   # computed, not LLM-supplied

        delta = round(abs(score_a - score_b), 2)

        # Dominant party derived from numbers — overrides LLM's qualitative label
        if delta < 5.0:
            dominant_party = "Balanced"
        elif score_a > score_b:
            dominant_party = "Company A (Buyer)"
        else:
            dominant_party = "Company B (Target)"

        payload_dict = payload.model_dump()
        payload_dict["dominant_party"] = dominant_party

        return cls(
            **payload_dict,
            chunk_id=chunk_id,
            mathematical_balance={
                "score_company_a":            score_a,
                "score_company_b":            score_b,
                "total_pool":                 round(total_pool, 2),
                "company_a_favorability_pct": fav_a,
                "company_b_favorability_pct": fav_b,
                "negotiation_delta_pct":      delta,
            },
        )


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

@dataclass
class Agent3Config:
    """
    All tuneable parameters for Agent 3.

    Parameters
    ----------
    api_key     : Groq API key. Falls back to GROQ_API_KEY env var if None.
    model_name  : Groq model identifier.
    temperature : LLM sampling temperature (low = more deterministic scoring).
    request_delay_s : Sleep between API calls to avoid rate-limiting.
    balanced_threshold_pts : Score delta below which result is 'Balanced'.
    """

    api_key: Optional[str] = None
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.05
    request_delay_s: float = 0.4
    balanced_threshold_pts: float = 5.0


# =====================================================================
# 3. CORE AGENT CLASS
# =====================================================================

class Agent3:
    """
    Quantitative M&A risk analysis engine.

    Designed for two usage patterns:
      1. Imported by a pipeline — call agent.run(flagged_chunks)
      2. Run standalone via CLI — python agent3.py --input ... --output ...

    Example (import)
    ----------------
    from agent3 import Agent3, Agent3Config

    agent   = Agent3(Agent3Config(api_key="gsk_..."))
    results = agent.run(flagged_chunks)          # list[ExpandedTwoSidedRiskSchema]
    """

    # LLM schema string built once at class init, not per-call
    _LLM_SCHEMA: str = json.dumps(LLMRiskPayload.model_json_schema(), indent=2)

    def __init__(self, config: Optional[Agent3Config] = None) -> None:
        self.config = config or Agent3Config()
        resolved_key = self.config.api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Pass it via Agent3Config(api_key=...) or set the environment variable."
            )
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=resolved_key,
        )
        logger.info("Agent3 initialised — model=%s", self.config.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        flagged_chunks: List[dict],
        *,
        verbose: bool = False,
    ) -> List[ExpandedTwoSidedRiskSchema]:
        """
        Analyse every flagged chunk and return structured results.

        Parameters
        ----------
        flagged_chunks : List of dicts with at minimum 'chunk_id' and 'text' keys.
        verbose        : When True, prints the per-clause live scorecard to stdout.
                         Always False when used as a library; True when run via CLI.

        Returns
        -------
        List[ExpandedTwoSidedRiskSchema] — one entry per successfully analysed chunk.
        Failed chunks are logged as errors and skipped (never raise).
        """
        results: List[ExpandedTwoSidedRiskSchema] = []
        total_start = time.time()

        if verbose:
            _print_banner()

        for chunk in flagged_chunks:
            chunk_id: str = chunk.get("chunk_id", "UNKNOWN")
            chunk_text: str = chunk.get("text", "")

            if not chunk_text.strip():
                logger.warning("Chunk %s has empty text — skipping.", chunk_id)
                continue

            if verbose:
                print(f"\n▶  Analyzing Clause: {chunk_id}")
                print(f"   [Text length]: {len(chunk_text)} characters")

            t0 = time.time()
            try:
                result = self._evaluate_clause(chunk_text, chunk_id=chunk_id)
                latency = round(time.time() - t0, 3)
                result.__dict__["_latency"] = latency   # attach timing metadata non-intrusively
                results.append(result)
                logger.info("Chunk %s analysed in %.2fs", chunk_id, latency)

                if verbose:
                    _print_clause_result(result, latency)

            except Exception as exc:
                logger.error("Chunk %s failed: %s", chunk_id, exc, exc_info=True)
                if verbose:
                    print(f"   ❌ Pipeline fault on {chunk_id}: {exc}")

            if verbose:
                print("-" * 120)

            time.sleep(self.config.request_delay_s)

        total_duration = round(time.time() - total_start, 2)
        logger.info(
            "Agent3 complete — %d/%d chunks succeeded in %.2fs",
            len(results), len(flagged_chunks), total_duration,
        )

        if verbose:
            _print_scorecard(results, total_duration)

        return results

    def evaluate_single(self, chunk_id: str, text: str) -> ExpandedTwoSidedRiskSchema:
        """
        Analyse one clause directly. Useful for unit tests and one-off debugging.

        Example
        -------
        result = agent.evaluate_single("CK-03", "Section 2.1 IP Vesting...")
        print(result.mathematical_balance)
        """
        return self._evaluate_clause(text, chunk_id=chunk_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert quantitative M&A corporate attorney risk engine.\n\n"

            "## TASK\n"
            "Analyze the contract clause as a zero-sum negotiation game between:\n"
            "  • Company A = Buyer / Acquirer\n"
            "  • Company B = Target / Seller\n\n"

            "## SCORING MODEL — ADVANTAGE-BASED\n"
            "Score each of the 6 parameters from 0 to 5 for BOTH parties INDEPENDENTLY.\n"
            "Scores represent the ADVANTAGE or POWER each party gains — NOT their exposure or risk.\n\n"
            "  5 = this party is the clear WINNER on this parameter\n"
            "  3 = moderate advantage\n"
            "  1 = slight advantage or neutral\n"
            "  0 = this party is the clear LOSER on this parameter\n\n"

            "## CRITICAL — ZERO-SUM RULE\n"
            "These scores are ZERO-SUM per parameter.\n"
            "If one party scores 5, the other must score 0 or 1.\n"
            "If one party scores 4, the other must score 0–2.\n"
            "Scores of 4/4 or 5/5 on the same parameter are FORBIDDEN.\n\n"

            "## WORKED EXAMPLE\n"
            "Clause: 'Buyer may terminate at any time for any reason without paying a break fee.\n"
            "If Seller terminates, Seller pays 50% of escrow as a break fee to Buyer.'\n\n"
            "  termination_remedial_asymmetry:\n"
            "    Company A (Buyer)  = 5  ← can exit freely AND collect a fee\n"
            "    Company B (Target) = 0  ← cannot exit without paying a penalty\n\n"

            "## STEP-BY-STEP SCORING PROCESS\n"
            "For each of the 6 parameters:\n"
            "  1. Ask: 'Who benefits MORE from this clause on this dimension?'\n"
            "  2. Give that party a HIGH score (3–5)\n"
            "  3. Give the other party a LOW score (0–2) — inverse of the winner\n"
            "  4. If a parameter is genuinely irrelevant to this clause, both parties score 1\n\n"

            "## PARAMETERS TO SCORE\n"
            "  1. financial_indemnity_exposure   — Who benefits from indemnity caps, baskets, survival limits?\n"
            "  2. operational_lockdown           — Who gains control over operations or freedom to act?\n"
            "  3. ip_asset_vulnerability         — Who gains IP ownership, title, or licensing rights?\n"
            "  4. termination_remedial_asymmetry — Who has more power to exit and collect remedies?\n"
            "  5. dispute_regulatory_burden      — Whose forum, fee, and defense terms are more favorable?\n"
            "  6. consequential_damage_exposures — Who retains the right to claim consequential damages?\n\n"

            "## OUTPUT FORMAT\n"
            f"Output ONLY a valid JSON object matching this schema:\n{self._LLM_SCHEMA}\n\n"
            "Do NOT include 'mathematical_balance' — it is computed externally from your scores.\n"
            "The 'dominant_party' field: your qualitative call "
            "('Company A (Buyer)', 'Company B (Target)', or 'Balanced')."
        )

    def _evaluate_clause(
        self,
        chunk_text: str,
        chunk_id: str = "",
    ) -> ExpandedTwoSidedRiskSchema:
        """Single LLM call + deterministic post-processing."""
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user",   "content": f"Analyze this contract clause:\n{chunk_text}"},
            ],
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )
        raw = LLMRiskPayload.model_validate_json(response.choices[0].message.content)
        return ExpandedTwoSidedRiskSchema.from_llm_payload(raw, chunk_id=chunk_id)


# =====================================================================
# 4. I/O HELPERS  (used by CLI and importable for pipeline reuse)
# =====================================================================

def load_flagged_chunks(input_path: str | Path) -> List[dict]:
    """
    Load flagged chunks from an Agent 2 output JSON file.

    Parameters
    ----------
    input_path : Path to Agent-2-output.json

    Returns
    -------
    List of flagged chunk dicts, each with at least 'chunk_id' and 'text'.

    Raises
    ------
    FileNotFoundError : if the file does not exist
    ValueError        : if the JSON is missing the 'flagged' key
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "flagged" not in data:
        raise ValueError(
            f"'{path}' is missing the top-level 'flagged' key. "
            "Ensure Agent 2 ran successfully and produced the expected format."
        )

    chunks = data["flagged"]
    logger.info("Loaded %d flagged chunks from %s", len(chunks), path)
    return chunks


def save_results(
    results: List[ExpandedTwoSidedRiskSchema],
    output_path: str | Path,
) -> None:
    """
    Serialise Agent 3 results to a JSON file.

    Parameters
    ----------
    results     : The list returned by Agent3.run()
    output_path : Destination file path. Parent directories are created if needed.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "agent": "agent3",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_clauses": len(results),
        "results": [r.model_dump() for r in results],
    }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    logger.info("Results saved → %s", path.resolve())
    print(f"\n✅ Results saved → {path.resolve()}")


# =====================================================================
# 5. CLI DISPLAY HELPERS  (stdout only — never called by library users)
# =====================================================================

def _print_banner() -> None:
    print("\n" + "=" * 120)
    print("🚀 AGENT 3 — QUANTITATIVE RISK ANALYSIS ENGINE")
    print("   Balance computed deterministically from weighted 6-parameter scores")
    print("=" * 120)


def _print_clause_result(result: ExpandedTwoSidedRiskSchema, latency: float) -> None:
    mb = result.mathematical_balance
    print(f"   [Dominant Party]:    ★ {result.dominant_party} ★")
    print(f"   [Weighted Scores]:   Company A: {mb['score_company_a']:.1f}/100  |  Company B: {mb['score_company_b']:.1f}/100")
    print(f"   [Power Balance]:     A: {mb['company_a_favorability_pct']}%  |  B: {mb['company_b_favorability_pct']}%")
    print(f"   [Delta]:             {mb['negotiation_delta_pct']} pts")
    print(f"   [Risk Clause]:       \"{result.clause_causing_risk}\"")
    print(f"   [Legal Reason]:      {result.reason_for_risk}")
    print(f"   [Compromise]:        {result.suggested_compromise}")
    print(f"   [Latency]:           {latency}s")


def _print_scorecard(results: List[ExpandedTwoSidedRiskSchema], total_duration: float) -> None:
    avg_latency = (
        sum(getattr(r, "__dict__", {}).get("_latency", 0) for r in results) / len(results)
        if results else 0
    )

    W = 140
    print("\n\n" + "═" * W)
    print(" 📋 PRODUCTION SCORECARD — AGENT 3  [Deterministic Balance]")
    print("═" * W)
    print(
        f" 📊 Clauses analysed: {len(results)}"
        f"  |  Avg latency: {avg_latency:.2f}s"
        f"  |  Total time: {total_duration:.2f}s"
    )
    print("─" * W)
    print(
        f" {'ID':<10} | {'Clause Summary':<48} | {'Dominant':<22}"
        f" | {'Score A/B':<17} | {'Balance (A/B)':<15} | {'Δ pts':<8}"
    )
    print("─" * W)
    for r in results:
        mb = r.mathematical_balance
        score_col = f"{mb['score_company_a']:.1f} / {mb['score_company_b']:.1f}"
        bal_col   = f"{mb['company_a_favorability_pct']}% / {mb['company_b_favorability_pct']}%"
        print(
            f" {r.chunk_id:<10} | {r.clause_summary[:48]:<48} | {r.dominant_party:<22}"
            f" | {score_col:<17} | {bal_col:<15} | {mb['negotiation_delta_pct']:<8}"
        )
    print("─" * W)

    print("\n 📍 RISK CLAUSE EXTRACTION & COMPROMISE LOG:")
    print("─" * W)
    for r in results:
        excerpt = r.clause_causing_risk[:75] + "..."
        print(f" [{r.chunk_id}] Excerpt: \"{excerpt}\"")
        print(f" {' ' * len(r.chunk_id)}  → Compromise: {r.suggested_compromise}")
        print()
    print("═" * W + "\n")


# =====================================================================
# 6. CLI ENTRY POINT
# =====================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent3",
        description=(
            "Agent 3 — Quantitative M&A Risk Analysis Engine\n"
            "Analyses flagged contract clauses from Agent 2 output JSON.\n\n"
            "Examples:\n"
            "  python agent3.py --input Agent-2-output.json\n"
            "  python agent3.py --input Agent-2-output.json --output results.json\n"
            "  python agent3.py --input Agent-2-output.json --chunk CK-03\n"
            "  python agent3.py --input Agent-2-output.json --model llama-3.3-70b-versatile --no-save\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to Agent-2-output.json (required)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help=(
            "Path to write JSON results. "
            "Defaults to '<input_stem>-agent3-results.json' beside the input file."
        ),
    )
    parser.add_argument(
        "--model", "-m",
        default="llama-3.3-70b-versatile",
        metavar="MODEL",
        help="Groq model name (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--chunk", "-c",
        default=None,
        metavar="CHUNK_ID",
        help="Analyse a single chunk by ID for debugging (e.g. --chunk CK-03)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results to stdout only; do not write an output file.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        metavar="SECONDS",
        help="Sleep between API calls in seconds (default: 0.4)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point. Returns an exit code (0 = success, 1 = error).
    Separated from __main__ block so it can be called in tests.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Configure logging for standalone use
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Load input ---
    try:
        flagged_chunks = load_flagged_chunks(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n❌ Input error: {exc}", file=sys.stderr)
        return 1

    # --- Filter to single chunk if --chunk given ---
    if args.chunk:
        flagged_chunks = [c for c in flagged_chunks if c.get("chunk_id") == args.chunk]
        if not flagged_chunks:
            print(
                f"\n❌ Chunk '{args.chunk}' not found in flagged list.",
                file=sys.stderr,
            )
            return 1
        print(f"🔍 Debug mode — analysing single chunk: {args.chunk}")

    print(f"⚡ Booting Agent 3 — {len(flagged_chunks)} clause(s) queued | model: {args.model}")

    # --- Initialise agent ---
    try:
        config = Agent3Config(model_name=args.model, request_delay_s=args.delay)
        agent  = Agent3(config)
    except EnvironmentError as exc:
        print(f"\n❌ Config error: {exc}", file=sys.stderr)
        return 1

    # --- Run analysis ---
    results = agent.run(flagged_chunks, verbose=True)

    if not results:
        print("\n⚠️  No clauses were successfully analysed.", file=sys.stderr)
        return 1

    # --- Save output ---
    if not args.no_save:
        if args.output:
            out_path = Path(args.output)
        else:
            in_path  = Path(args.input)
            out_path = in_path.parent / f"{in_path.stem}-agent3-results.json"

        try:
            save_results(results, out_path)
        except OSError as exc:
            print(f"\n❌ Could not write output file: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())