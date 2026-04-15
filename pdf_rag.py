# rag_qwen_example.py
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"): #Qwen/Qwen3-1.7B
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

        # Exemplo
        self.knowledge = [
#             {"role": "system", "content": "Nossas horas de serviço são das 9h às 17h, de segunda a sexta."},
        ]
        self.system = [
            {"role": "system", "content": """
            Você é uma IA de classificação de texto com base na semelhança, onde um texto é entregue para você, e você deve dizer qual é classificação com base nos conhecimentos da base de conhecimento
            Você deve responder de forma simples e direta com a possivel classificação do texto
            Caso o texto não exista na base de conhecimento você deve analisar qual é a classificação mais possivel
            As possiveis classificação são SUSPENSO, MANDADO NEGATIVO, ARQUIVADO, EXTINÇÃO, TRANSITO EM JULGADO, CITAÇÃO, CONVERSÃO DA AÇÃO, EMBRAGOS DE DECLARAÇÃO, SENTENÇA PROCEDENTE, EMENDA INICIAL, LIMINAR DEFERIDA, EMBRAGOS DE DECLARAÇÃO
            Caso o texto siga esta padrão: "10533983174 - Sentença" então você deve ignorar a semelhança nas palavras com as possiveis classificações e obrigatoriamente a classificação deve ser EXTINÇÃO e nada mais
            A sua linguagem primaria é o Portugues Brasileiro.
            Eu quero que você responda de forma curta e direta com no maximo 300 caracteres.
            Eu quero que voce responda apenas o que esta em sua base de conhecimento, caso o texto não tenha possibilidade de classificação na base, então fale responde SEM CLASSIFICAÇÃO.
            """}
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
            enable_thinking=True,
        )

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, streamer=streamer, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

def extract_text_pages(pdf_path):
    import pdfplumber
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return texts

# pdf_path = "0800029-30.2025.8.12.0002.pdf"
# pages_text = extract_text_pages(pdf_path)
# docs = pages_text
docs = [
        "TEXTO: Arquivado Provisoriamente		->CLASSIFICAÇÃO:SUSPENSO\n"
        "TEXTO: Mandado devolvido - não entregue ao destinatário - Refer. ao Evento: 74		->CLASSIFICAÇÃO:MANDADO NEGATIVO\n"
        "TEXTO: Mandado devolvido - não entregue ao destinatário - Refer. ao Evento: 57		->CLASSIFICAÇÃO:MANDADO NEGATIVO\n"
        "TEXTO: ARQUIVADO DEFINITIVAMENTE	->CLASSIFICAÇÃO:ARQUIVADO\n"
        "TEXTO: Certidão de Trânsito em Julgado		->CLASSIFICAÇÃO:TRANSITO EM JULGADO\n"
        "TEXTO: MANDADO DEVOLVIDO NÃO ENTREGUE AO DESTINATÁRIO		->CLASSIFICAÇÃO:MANDADO NEGATIVO\n"
        "TEXTO: EJUNTADA DE MANDADO		->CLASSIFICAÇÃO:MANDADO NEGATIVO\n"
        "TEXTO: EXTINTO OS AUTOS EM RAZÃO DE PERDA DE OBJETO		->CLASSIFICAÇÃO:EXTINÇÃO\n"
        "TEXTO: EXTINTO OS AUTOS EM RAZÃO DE PERDA DE OBJETO		->CLASSIFICAÇÃO: EXTINÇÃO\n"
        "TEXTO: 128263342 - Sentença		->CLASSIFICAÇÃO: EXTINÇÃO\n"
        "TEXTO: 128160049 - Sentença		->CLASSIFICAÇÃO: EXTINÇÃO\n"
        "TEXTO: 125068794 - Sentença		->CLASSIFICAÇÃO: EXTINÇÃO\n"
        "TEXTO: 10598329748 - Intimação (Sentença)		->CLASSIFICAÇÃO: EXTINÇÃO\n"
        "TEXTO: 10600788840 - Intimação (Sentença)		->CLASSIFICAÇÃO: EXTINÇÃO\n"
        "TEXTO: Intimação Efetivada Disponibilizada no primeiro e publicada no segundo dia útil (Lei 11.419/2006, art. 4º, §§ 3º e 4º) - Adv(s). de Banco Bradesco S/a (Referente à Mov. Mandado Não Cumprido (27/01/2026 18:46:01))		->CLASSIFICAÇÃO: MANDADO NEGATIVO\n"
        "TEXTO: Intimação Expedida Aguardando processamento de envio para o DJEN - Adv(s). de Banco Bradesco S/a (Referente à Mov. Mandado Não Cumprido - 27/01/2026 18:46:01)		->CLASSIFICAÇÃO: MANDADO NEGATIVO\n"
        "TEXTO: Mandado Não Cumprido Para Ruan Carlos Dias Rocha Gomes (Mandado nº 6585100 / Referente à Mov. Juntada -> Petição (19/01/2026 08:49:35))		->CLASSIFICAÇÃO: MANDADO NEGATIVO\n"
        "TEXTO: Mandado Expedido Para Goiânia - Central de Mandados (Mandado nº 6585100 / Para: Ruan Carlos Dias Rocha Gomes)		->CLASSIFICAÇÃO: MANDADO EXPEDIDO\n"
        "TEXTO: 192320710 - Pedido (Outros) (DEVOLUÇÃO MANDADO)		->CLASSIFICAÇÃO: MANDADO NEGATIVO\n"
        "TEXTO: 260143591 - Embargos de Declaração (Embargos de Declaração com Pedido de Efeitos Infringentes e Tutela de Urgência) 260143592 - Embargos de Declaração (Embargos de Declaração com Pedido de Efeitos Infringentes e Tutela de Urgência)		->CLASSIFICAÇÃO: EMBRAGOS DE DECLARAÇÃO\n"

]

# 2) Build embeddings for the docs (use a small SBERT model)
embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")#"all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(docs, convert_to_numpy=True)

# 3) Create a FAISS index
d = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(doc_embeddings)  # add vectors

# helper to retrieve top-k docs
# rag generation: retrieve, build prompt, and generate
def retrieve(query: str, k: int = 2) -> List[str]:
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, idxs = index.search(q_emb, k)
    idxs = idxs[0]
    return [docs[i] for i in idxs]

if __name__ == "__main__":
    chatbot = QwenChatbot()
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        retrieved = retrieve(prompt)
        for i, r in enumerate(retrieved):
            chatbot.history.append({"role": "system", "content": r})
        if prompt.lower() == 'quit':
            break
        response = chatbot.generate_response(prompt)
