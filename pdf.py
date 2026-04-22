import os
import faiss
import numpy as np
import gradio as gr
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

class QwenChatbot:
    def __init__(
            self, 
            model_name="Qwen/Qwen3-0.6B", #Qwen/Qwen3-1.7B
            use_rag=True, 
            use_text_streamer=True,
            use_history=True,
            use_thinking=True,
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

        self.knowledge = [
#             {"role": "system", "content": "Nossas horas de serviço são das 9h às 17h, de segunda a sexta."},
        ]
        self.system = [
            {"role": "system", "content": """"""}
        ]
        for entry in self.knowledge:
            self.system.append(entry)
        for entry in self.system:
            self.history.append(entry)

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, top_p=0.3, top_k=1, temperature=0.1, use_cache=True, do_sample=False, streamer=streamer, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

def load_pdfs(pdf_files):
    text_chunks = []
    if not isinstance(pdf_files, list):
        pdf_files = [pdf_files]
    for file_path in pdf_files:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                text_chunks.extend(chunks)
    embeddings = embedding_model.encode(text_chunks[:10], convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return text_chunks, index

def query_rag(question, chunks, index, k=3):
    q_emb = embedding_model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    try:
        chatbot = QwenChatbot()
        chatbot.history.append({"role": "system", "content": "<retrieved>"+prompt+"</retrieved>"})
        return chatbot.generate_response(prompt)
    except Exception as e:
        return str(e)

def summarize_pdf(chunks):
    context = " ".join(chunks[:10])
    prompt = f"Summarize the following text:\n{context}"
    try:
        chatbot = QwenChatbot()
        chatbot.history.append({"role": "system", "content": "<retrieved>"+prompt+"</retrieved>"})
        return chatbot.generate_response(prompt)
    except Exception as e:
        return str(e)

def process(pdf_file, question, mode):
    if not pdf_file:
        return "No PDF uploaded"
    chunks, index = load_pdfs(pdf_file)
    if mode == "QnA" and question.strip():
        return query_rag(question, chunks, index)
    elif mode == "Summary":
        return summarize_pdf(chunks)
    else:
        return "Please enter a question or select Summary"

with gr.Blocks() as demo:
    with gr.Row():
        pdf_input = gr.File(type="filepath", file_types=[".pdf"], label="Upload your PDF")
    with gr.Row():
        question = gr.Textbox(label="Enter your question (leave empty for Summary)")
        mode = gr.Radio(["QnA", "Summary"], value="Summary", label="Mode")
    output = gr.Textbox(label="Output")
    submit = gr.Button("Submit")
    submit.click(fn=process, inputs=[pdf_input, question, mode], outputs=output)

demo.launch()
