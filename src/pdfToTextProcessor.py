import pdfplumber
import sys
import os

def pdf_to_text(pdf_path, txt_path):
    with pdfplumber.open(pdf_path) as pdf, open(txt_path, "w", encoding="utf-8") as out:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                out.write(f"\n--- Page {i + 1} ---\n")
                out.write(text)
                out.write("\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_to_text.py <input.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    txt_path = os.path.splitext(pdf_path)[0] + ".txt"

    pdf_to_text(pdf_path, txt_path)
    print(f"Text extracted to {txt_path}")