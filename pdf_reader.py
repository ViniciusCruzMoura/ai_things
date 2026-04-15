from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pdfplumber
from PIL import Image
import fitz  # PyMuPDF

def extract_text_pages(pdf_path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return texts

def extract_page_images(pdf_path, zoom=2):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

pdf_path = "0800029-30.2025.8.12.0002.pdf"
pages_text = extract_text_pages(pdf_path)
#page_images = extract_page_images(pdf_path)

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_from_text(prompt, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

prompt = "Resuma o seguinte informação, utilizando no maximo 10000 characteres. A sua linguagem primaria é o Portugues Brasileiro.:\n\n" + "\n\n".join(pages_text[:3])
print(generate_from_text(prompt))
