# Install required packages
!pip install -q transformers torch accelerate
!pip install -q python-docx PyPDF2 pdfplumber
!pip install -q pandas numpy
!pip install -q spacy
!pip install -q gradio
!python -m spacy download en_core_web_sm

import torch
import pandas as pd
import numpy as np
import re
import io
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Document processing imports
import PyPDF2
import pdfplumber
from docx import Document
import spacy

# Transformers imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Gradio import
import gradio as gr

# =============================================================================
# GRANITE MODEL LOADING (YOUR EXISTING CODE)
# =============================================================================

def load_granite_model():
    """Load IBM Granite model for text generation"""
    print("Loading IBM Granite model...")

    try:
        # Load the IBM Granite model
        model_name = "ibm-granite/granite-3.0-2b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad_token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # Create pipeline
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            device=0 if torch.cuda.is_available() else -1
        )

        print("✓ Granite model loaded successfully!")
        return text_generator, tokenizer

    except Exception as e:
        print(f"Error loading Granite model: {e}")
        print("Trying alternative model...")

        # Fallback to a smaller model if Granite fails
        try:
            fallback_model = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(fallback_model)
            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,
                temperature=0.3
            )
            print("✓ Fallback model loaded!")
            return text_generator, tokenizer
        except Exception as e2:
            print(f"Fallback model also failed: {e2}")
            return None, None

# Initialize the model
print("Initializing models...")
text_generator, tokenizer = load_granite_model()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded!")
except:
    print("Installing spaCy model...")
    !python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

# =============================================================================
# DOCUMENT PROCESSING FUNCTIONS (YOUR EXISTING CODE)
# =============================================================================

def extract_text_from_pdf(file_path, max_pages=5):
    """Extract text from PDF file (limited pages for speed)"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            pages_to_process = min(len(pdf.pages), max_pages)
            for i in range(pages_to_process):
                page_text = pdf.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs[:50]:  # Limit paragraphs for speed
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content[:10000]  # Limit to first 10k characters
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

def process_uploaded_document(file_path):
    """Process uploaded document and extract text based on file extension"""
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(file_path)
    elif file_extension == 'txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {file_extension}")
        return ""

# =============================================================================
# GRANITE-POWERED ANALYSIS FUNCTIONS (YOUR EXISTING CODE)
# =============================================================================

def classify_document_type_granite(text):
    """Classify document using Granite model"""
    if text_generator is None:
        return classify_document_type_fast(text)

    prompt = f"""Analyze this legal document and classify it into one of these categories:
- NDA
- Lease Agreement
- Employment Contract
- Service Agreement

Document text (first 1000 chars):
{text[:1000]}

Classification:"""

    try:
        response = text_generator(prompt, max_new_tokens=50, temperature=0.1)
        generated_text = response[0]['generated_text']
        classification = generated_text.split("Classification:")[-1].strip()

        # Clean up the classification
        classification = classification.split('\n')[0].split('.')[0].strip()

        # Map to standard categories
        classification_lower = classification.lower()
        if 'nda' in classification_lower or 'non-disclosure' in classification_lower:
            return "NDA"
        elif 'lease' in classification_lower:
            return "Lease Agreement"
        elif 'employment' in classification_lower:
            return "Employment Contract"
        elif 'service' in classification_lower:
            return "Service Agreement"
        else:
            return "NDA"  # Default fallback

    except Exception as e:
        print(f"Error in Granite classification: {e}")
        return classify_document_type_fast(text)

def generate_granite_summary(text, doc_type):
    """Generate summary using Granite model"""
    if text_generator is None:
        return generate_summary_fast(text, doc_type)

    prompt = f"""Summarize this {doc_type} document in 2-3 clear, concise sentences. Focus on key parties, main purpose, and critical terms.

Document:
{text[:2000]}

Summary:"""

    try:
        response = text_generator(prompt, max_new_tokens=200, temperature=0.2)
        generated_text = response[0]['generated_text']
        summary = generated_text.split("Summary:")[-1].strip()

        # Clean up the summary
        summary = summary.split('\n\n')[0].strip()
        # Remove any incomplete sentences
        sentences = summary.split('.')
        if len(sentences) > 1 and len(sentences[-1]) < 10:
            summary = '.'.join(sentences[:-1]) + '.'

        return summary

    except Exception as e:
        print(f"Error generating Granite summary: {e}")
        return generate_summary_fast(text, doc_type)

def extract_key_entities(text):
    """Extract key entities with focus on dates and parties"""
    entities = {
        'Dates': [],
        'Parties': []
    }

    try:
        # Use spaCy for NER
        doc = nlp(text[:8000])  # Process more text for better extraction

        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME'] and len(ent.text) > 4:
                # Filter out very short date mentions
                date_text = ent.text.strip()
                if any(char.isdigit() for char in date_text):
                    entities['Dates'].append(date_text)

            elif ent.label_ in ['PERSON', 'ORG'] and len(ent.text) > 2:
                # Filter out very short names
                party_text = ent.text.strip()
                if not party_text.lower() in ['agreement', 'contract', 'document', 'party', 'parties']:
                    entities['Parties'].append(party_text)

    except Exception as e:
        print(f"spaCy processing error: {e}")

    # Enhanced date pattern matching
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY or M/D/YYYY
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['Dates'].extend(matches)

    # Enhanced party extraction using common legal patterns
    party_patterns = [
        r'"([A-Z][a-z]+ [A-Z][a-z]+)"',  # Names in quotes
        r'\b([A-Z][a-z]+ [A-Z][a-z]+)\s*\("',  # Name followed by parenthesis
        r'between\s+([A-Z][a-zA-Z\s]+?)\s+and',  # "between X and"
        r'by and between\s+([A-Z][a-zA-Z\s]+?),',  # "by and between X,"
        r'\b([A-Z][A-Za-z]+\s+(?:Corporation|Corp|LLC|Inc|Company|Ltd))\b'  # Company names
    ]

    for pattern in party_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_match = match.strip()
            if len(clean_match) > 3 and len(clean_match) < 50:
                entities['Parties'].append(clean_match)

    # Remove duplicates and clean up
    entities['Dates'] = list(set([date.strip() for date in entities['Dates'] if date.strip()]))[:10]
    entities['Parties'] = list(set([party.strip() for party in entities['Parties'] if party.strip()]))[:10]

    return entities

def classify_document_type_fast(text):
    """Fast rule-based document classification (fallback method)"""
    text_lower = text.lower()

    # Define keywords for each document type
    nda_keywords = ['non-disclosure', 'confidential', 'proprietary', 'trade secret', 'nda']
    lease_keywords = ['lease', 'rent', 'tenant', 'landlord', 'premises', 'rental']
    employment_keywords = ['employment', 'employee', 'employer', 'salary', 'wages', 'job', 'position']
    service_keywords = ['service', 'services', 'contractor', 'consulting', 'agreement']

    scores = {
        'NDA': sum(1 for kw in nda_keywords if kw in text_lower),
        'Lease Agreement': sum(1 for kw in lease_keywords if kw in text_lower),
        'Employment Contract': sum(1 for kw in employment_keywords if kw in text_lower),
        'Service Agreement': sum(1 for kw in service_keywords if kw in text_lower)
    }

    if max(scores.values()) > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    else:
        return "NDA"

def generate_summary_fast(text, doc_type):
    """Fast rule-based summary generation (fallback method)"""
    # Key sentences extraction
    sentences = re.split(r'[.!?]+', text[:2000])
    key_sentences = []

    # Look for important indicators
    important_patterns = [
        r'this agreement',
        r'parties agree',
        r'in consideration',
        r'terms and conditions',
        r'effective date',
        r'whereas'
    ]

    for sentence in sentences:
        if any(pattern in sentence.lower() for pattern in important_patterns):
            key_sentences.append(sentence.strip())
            if len(key_sentences) >= 3:
                break

    if key_sentences:
        summary = '. '.join(key_sentences[:3]) + '.'
    else:
        # Fallback summary
        summary = f"This {doc_type} contains legal terms and conditions governing the relationship between parties. The document outlines specific obligations, rights, and responsibilities of all involved parties."

    return summary[:400]  # Limit length

# =============================================================================
# MAIN ANALYSIS FUNCTION (YOUR EXISTING CODE WITH MODIFIED OUTPUT)
# =============================================================================

def analyze_legal_document(text):
    """Analyze legal document and return results for Gradio display"""
    print("Analyzing document...")

    # 1. Generate Summary using Granite
    doc_type_temp = classify_document_type_granite(text)
    summary = generate_granite_summary(text, doc_type_temp)

    # 2. Extract key entities (Dates and Parties)
    entities = extract_key_entities(text)

    # 3. Final document classification
    doc_type = classify_document_type_granite(text)

    return {
        'document_type': doc_type,
        'summary': summary,
        'dates': entities['Dates'],
        'parties': entities['Parties']
    }

# =============================================================================
# GRADIO INTERFACE FUNCTIONS
# =============================================================================

def process_document_for_gradio(file):
    """Process uploaded file for Gradio interface"""
    if file is None:
        return "❌ No file uploaded. Please upload a document.", "", "", ""

    try:
        # Get the file path
        file_path = file.name

        # Extract text from the document
        document_text = process_uploaded_document(file_path)

        if not document_text or len(document_text.strip()) < 100:
            return "❌ Failed to extract meaningful text from the document. Please check if the file contains readable text.", "", "", ""

        # Analyze the document using your existing function
        results = analyze_legal_document(document_text)

        # Format the output for Gradio display
        summary_output = f"**SUMMARY:**\n{results['summary']}"

        dates_output = "**DATES:**\n"
        if results['dates']:
            for date in results['dates']:
                dates_output += f"• {date}\n"
        else:
            dates_output += "• No dates found\n"

        parties_output = "**PARTIES:**\n"
        if results['parties']:
            for party in results['parties']:
                parties_output += f"• {party}\n"
        else:
            parties_output += "• No parties identified\n"

        classification_output = f"**DOCUMENT TYPE CLASSIFICATION:** {results['document_type']}"

        return summary_output, dates_output, parties_output, classification_output

    except Exception as e:
        error_msg = f"❌ Error processing document: {str(e)}"
        return error_msg, "", "", ""

# =============================================================================
# GRADIO INTERFACE SETUP
# =============================================================================

def create_gradio_interface():
    """Create and launch Gradio interface"""

    # Custom CSS for professional styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        background-color: #f9f9f9;
    }
    .results-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Clausewise - Legal Document Analyzer") as interface:

        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: bold;">Clausewise</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">AI-Powered Legal Document Analysis</p>
        </div>
        """)

        # Description
        gr.Markdown("""
        ### 📄 Upload your legal document to get:
        - **AI-Generated Summary** using IBM Granite model
        - **Key Information Extraction** (dates, parties, terms)
        - **Document Type Classification** (NDA, Lease Agreement, Employment Contract, Service Agreement)

        **Supported formats:** PDF, DOCX, TXT
        """)

        # Upload section
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".docx", ".doc", ".txt"],
                    type="filepath"
                )

                analyze_btn = gr.Button(
                    "🔍 Analyze Document",
                    variant="primary",
                    size="lg"
                )

        # Results section
        gr.Markdown("## 📊 Analysis Results")

        with gr.Row():
            with gr.Column(scale=2):
                summary_output = gr.Markdown(
                    label="Summary",
                    value="Upload a document to see the AI-generated summary here."
                )

            with gr.Column(scale=1):
                classification_output = gr.Markdown(
                    label="Document Classification",
                    value="Document type will appear here."
                )

        with gr.Row():
            with gr.Column():
                dates_output = gr.Markdown(
                    label="Important Dates",
                    value="Key dates will be extracted and displayed here."
                )

            with gr.Column():
                parties_output = gr.Markdown(
                    label="Parties Involved",
                    value="Identified parties and entities will appear here."
                )

        # Event handlers
        analyze_btn.click(
            fn=process_document_for_gradio,
            inputs=[file_input],
            outputs=[summary_output, dates_output, parties_output, classification_output]
        )

        # Auto-analyze when file is uploaded
        file_input.change(
            fn=process_document_for_gradio,
            inputs=[file_input],
            outputs=[summary_output, dates_output, parties_output, classification_output]
        )

        # Footer
        gr.Markdown("""
        ---
        **Powered by:** IBM Granite 3.0 Model | spaCy NLP | Gradio Interface

        **Note:** This tool processes the first 5 pages of PDFs and first 50 paragraphs of DOCX files for optimal performance.
        """)

    return interface

# =============================================================================
# LAUNCH THE APPLICATION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Clausewise Legal Document Analyzer...")
    print("=" * 60)

    # Create and launch the Gradio interface
    interface = create_gradio_interface()

    # Launch with public link for Colab
    interface.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        debug=True
    )
