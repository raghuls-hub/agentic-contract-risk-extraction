#===============================================================================================
# To test this script alone use :
# python document_chunker.py --input data/ma_contract_high_risk.pdf --output test_output.json --size 600
#===============================================================================================

#===============================================================================================
# To use it in the main pipeline :

# main.py code snippet

# def run_agent_pipeline(pdf_file_path):
#     # Initialize Layer 1 Chunker
#     chunker = SmartDocumentChunker(chunk_size=600, chunk_overlap=150)
    
#     # Process text and catch chunks directly in a variable memory block
#     chunks_data = chunker.process_document(pdf_path=pdf_file_path)
    
#     # Forward the text arrays into Agent 2 (The Hybrid Router)
#     print(f"Loaded {len(chunks_data)} records directly into routing vector pools.")
#     return chunks_data
#===============================================================================================

import os
import re
import json
import argparse
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SmartDocumentChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Symmetrical and structural hierarchy fallback
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        print(f"[*] Extracting raw text from: {pdf_path}")
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            return full_text
        except Exception as e:
            print(f"[-] Failed to read PDF: {e}")
            return ""

    def clean_and_stitch_text(self, raw_text: str) -> str:
        print("[*] Cleaning text and stitching broken lines...")

        # 1. Fix hyphenated words broken across lines
        text = re.sub(r'([A-Za-z]+)-\n([A-Za-z]+)', r'\1\2', raw_text)

        # 2. Fix sentences arbitrarily broken by layout newlines
        text = re.sub(r'(?<![.!?::;])\n+([a-z])', r' \1', text)

        # 3. Fix words broken by layout without hyphens (e.g., "f\nirst")
        text = re.sub(r'([a-z])\n([a-z])', r'\1\2', text)

        # 4. Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 5. Remove excess white spaces
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    def create_chunks(self, cleaned_text: str, filename: str) -> list:
        print("[*] Generating structural chunks...")
        raw_chunks = self.text_splitter.split_text(cleaned_text)

        formatted_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            formatted_chunks.append({
                "chunk_id": f"CK-{str(i+1).zfill(2)}",
                "source_file": filename,
                "chunk_text": chunk_text.strip()
            })

        return formatted_chunks

    def process_document(self, pdf_path: str, output_json_path: str = None) -> list:
        """
        Processes a document. If output_json_path is provided, saves to disk.
        Always returns the generated list of chunks for downstream agent consumption.
        """
        filename = os.path.basename(pdf_path)

        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            print("[-] Terminating. No text could be pulled.")
            return []

        cleaned_text = self.clean_and_stitch_text(raw_text)
        final_chunks = self.create_chunks(cleaned_text, filename)

        if output_json_path:
            print(f"[*] Saving {len(final_chunks)} perfectly stitched chunks to {output_json_path}...")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_chunks, f, indent=2, ensure_ascii=False)

        print("[====== SUCCESS ======] Document processing complete!")
        return final_chunks


# =====================================================================
# STANDALONE EXECUTION & DEBUGGING INTERFACE
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Document Chunker - Layer 1 Pipeline Utility")
    
    # Define parameters that can be handled explicitly through terminal execution
    parser.add_argument("--input", type=str, required=True, help="Path to the input target M&A contract PDF")
    parser.add_argument("--output", type=str, default="cleaned_chunks.json", help="Path to output target JSON results destination")
    parser.add_argument("--size", type=int, default=600, help="Chunk layout text allocation limit window size")
    parser.add_argument("--overlap", type=int, default=150, help="Sliding validation window overlap tracking metric balance")

    args = parser.parse_args()

    # Execute standalone debugging instance
    print("\n--- STANDALONE DEBUG MODE MODE ACTIVE ---")
    chunker = SmartDocumentChunker(chunk_size=args.size, chunk_overlap=args.overlap)
    chunker.process_document(args.input, args.output)
