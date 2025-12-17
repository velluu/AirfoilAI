import pdfplumber
import pandas as pd
import re
from pathlib import Path

pdf_path = Path("1760_Cornelius_TM_120424.pdf")
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Extracting all text from PDF to find embedded data...")
with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    
    all_text = ""
    for page_num in range(len(pdf.pages)):
        page = pdf.pages[page_num]
        text = page.extract_text()
        all_text += f"\n\n=== PAGE {page_num + 1} ===\n\n" + text
    
    with open("full_pdf_text.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    print("Saved complete PDF text to full_pdf_text.txt")
    
    if "0.00,0.06" in all_text or "0.00, 0.06" in all_text:
        print("\n✓ Found numerical data in PDF!")
        lines = all_text.split('\n')
        data_lines = [line for line in lines if re.search(r'\d+\.\d+.*,.*\d+\.\d+', line)]
        print(f"Found {len(data_lines)} lines with numerical data")
        if data_lines:
            print("\nSample lines:")
            for line in data_lines[:10]:
                print(f"  {line}")
    else:
        print("\n✗ No embedded CSV data found in PDF")
        print("The data files are referenced but not included in this PDF.")
        print("They need to be obtained separately from NASA.")
