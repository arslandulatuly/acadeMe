import sys
import os
import pdfplumber
import langextract as lx
import textwrap

def extract_educational_concepts(pdf_path, api_key):
    """
    Extracts structured educational concepts from a PDF using LangExtract.
    """
    # 1. Extract raw text from PDF using your existing method
    full_text = ""
    print(f"Extracting text from: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Add page markers for source grounding
                full_text += f"\n--- PAGE {i+1} ---\n{text}\n"
    
    if not full_text:
        print("No text could be extracted from the PDF.")
        return None
    
    # 2. Define the educational extraction task
    prompt = textwrap.dedent("""
        Extract key educational concepts from the text.
        For each concept, provide:
        1. The concept name or key term.
        2. Its definition or core explanation.
        3. An example if present.
        4. The page number where it is found.
        
        Use exact text from the source for definitions and examples.
        """)
    
    # 3. Provide a clear example to guide the LLM
    examples = [
        lx.data.ExampleData(
            text="--- PAGE 1 ---\nNewton's First Law of Motion, also called the law of inertia, states that an object at rest stays at rest unless acted upon by an external force.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="key_concept",
                    extraction_text="Newton's First Law of Motion",
                    attributes={
                        "definition": "an object at rest stays at rest unless acted upon by an external force",
                        "also_known_as": "law of inertia",
                        "page": 1
                    }
                )
            ]
        )
    ]
    
    #---------------------------------------------------------------------------------
    # # 4. Run LangExtract with Ollama NOT OPTIMIZED
    # print("Running LLM extraction via Ollama...")
    # try:
    #     result = lx.extract(
    #         text_or_documents=full_text,
    #         prompt_description=prompt,
    #         examples=examples,
    #         model_id="gemma2:2b",          # Your local Ollama model name
    #         # using gemma since JSON error happens because models output free text 
    #         # instead of structured JSON. Gemma is specifically instruction-tuned to follow formatting rules, 
    #         # making it more likely to work with LangExtract.
            
    #         model_url="http://localhost:11434", # Ollama's default local URL
    #         fence_output=False,               # MUST be False for Ollama
    #         use_schema_constraints=False,     # MUST be False for Ollama
    #         max_char_buffer=1000              # Smaller chunks work better for local LLMs
    #     )
    #     print("Extraction successful!")
    #     return result
    # except Exception as e:
    #     print(f"Extraction failed: {e}")
    #     return None
    #---------------------------------------------------------------------------------
    
    
    # 4. Run LangExtract with Ollama
    print("Running LLM extraction via Ollama...")
    try:
        # MVP: Process only first 500 characters
        test_text = full_text[:500]
        
        # SIMPLIFIED EXAMPLE - matches our simpler prompt
        examples = [
            lx.data.ExampleData(
                text="Physics studies motion and forces.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="key_term",
                        extraction_text="Physics",
                        attributes={
                            "definition": "studies motion and forces"
                        }
                    )
                ]
            )
        ]
        
        result = lx.extract(
            text_or_documents=test_text,
            prompt_description="Extract 2 key educational terms with short definitions.",
            examples=examples,  # ⬅️ Now includes simple example
            model_id="gemma2:2b",
            model_url="http://localhost:11434",
            fence_output=False,
            use_schema_constraints=False,
            max_char_buffer=200,
            temperature=0.1,
            max_workers=1
        )
        print("Extraction successful!")
        return result
    except Exception as e:
        print(f"Extraction failed: {e}")
        print(f"Text attempted (first 150 chars): {full_text[:150]}...")
        return None
    
def save_raw_text(pdf_path, output_dir="."):
    """
    Extracts and saves the raw text from a PDF to a .txt file.
    Useful for debugging the text extraction step.
    """
    full_text = ""
    print(f"[DEBUG] Extracting raw text from: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n--- PAGE {i+1} ---\n{text}\n"
            else:
                full_text += f"\n--- PAGE {i+1} ---\n[NO TEXT EXTRACTED]\n"
    
    if not full_text.strip():
        print("[WARNING] No text could be extracted at all from the PDF.")
        return None
    
    # Create filename and save
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}_RAW_TEXT.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"[DEBUG] Raw text saved to: {txt_path}")
    return txt_path

def save_and_visualize(result, base_filename):
    """
    Saves the extraction results and generates an interactive visualization.
    """
    # 5. Save the structured results
    jsonl_path = f"{base_filename}_extractions.jsonl"
    lx.io.save_annotated_documents([result], output_name=jsonl_path, output_dir=".")
    print(f"Structured data saved to: {jsonl_path}")
    
    # 6. Generate an interactive HTML visualization
    html_content = lx.visualize(jsonl_path)
    html_path = f"{base_filename}_visualization.html"
    
    with open(html_path, "w", encoding='utf-8') as f:
        f.write(html_content if isinstance(html_content, str) else html_content.data)
    print(f"Interactive visualization saved to: {html_path}")
    print(f"Open this file in your browser to explore the extracted concepts.")

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python langExtractOllama.py <input.pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    base_name = os.path.splitext(pdf_path)[0]
    
    # --- NEW: Save the raw extracted text first ---
    raw_text_file = save_raw_text(pdf_path, output_dir=".")
    if not raw_text_file:
        print("Stopping. Could not extract text from PDF.")
        sys.exit(1)
    # --- END NEW ---
    
    # Proceed with LangExtract (with or without API key for Ollama)
    # If using Ollama, remember to REMOVE the api_key parameter
    extraction_result = extract_educational_concepts(pdf_path, api_key=None)
    
    if extraction_result:
        save_and_visualize(extraction_result, base_name)