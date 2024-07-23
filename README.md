# FIREWORKS Identity Verification PoC
### DEMO ATTACHED: [Watch the Loom Video](https://www.loom.com/share/2897fde24a444ae3ab1ac41a48a26cad?sid=50748ea6-f124-4481-8765-6fe7bf8df11f)


## Objective
To create an end-to-end PoC solution for Identity Verification using Firework AI’s platform and APIs along with other tools.

## Approaches

### 1. Direct LLM Approach
- Used FireLLaVA-13b for extracting data from images.
- Observed issues with hallucinations and inconsistency.

### 2. Tesseract OCR Preprocessing
- Implemented Tesseract OCR for text extraction.
- Results were gibberish and not usable.

### 3. GCP Document AI Preprocessing
- Used GCP Document AI for extracting text from documents.
- Received raw text, which required further processing.

### 4. Combining GCP Document AI with Fireworks Mixtral
- Processed the raw text using Fireworks mixtral-8x22b-instruct model.
- Achieved better accuracy and consistency.
![Architecture of GCP to Fireworks](https://github.com/shubcodes/fw-kyc-poc/blob/fab48fa200f233a5294fb76a574d141ebd87519c/Fireworks.png)

