from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"): #Qwen/Qwen3-1.7B
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

        # Exemplo
        self.knowledge = [
            {"role": "system", "content": "Nossas horas de serviço são das 9h às 17h, de segunda a sexta."},
        ]
        self.system = [
            {"role": "system", "content": """
            Eu quero que voce atue como uma IA Atendente da empresa GrupoCard chamado Cardoso, o seu proposito é guiar as pessoas no chatbot.
            A sua linguagem primaria é o Portugues Brasileiro.
            Eu quero que você responda de forma curta e direta com no maximo 10000 caracteres.
            Eu quero que voce responda apenas o que esta em sua base de conhecimento, caso a pergunta não tenha resposta na base, então fale que não sabe sobre o assunto.
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

        from transformers import TextStreamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, streamer=streamer, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

if __name__ == "__main__":
    chatbot = QwenChatbot()

    def extract_text_pages(pdf_path):
        import pdfplumber
        texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return texts
    
    pdf_path = "0800029-30.2025.8.12.0002.pdf"
    pages_text = extract_text_pages(pdf_path)

    chatbot.history.append(
            {"role": "system", "content": "\n\n".join(pages_text[:3])}
            )
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")

        if prompt.lower() == 'quit':
            break

        response = chatbot.generate_response(prompt)
#     response = chatbot.generate_response("Qual é o nome pessoa jurídica, seu CNPJ, endereço, tipo de ação, o nome da parte e seu CPF e endereço")

#     while True:
#         prompt = input("Enter your prompt (or 'quit' to exit): ")
# 
#         if prompt.lower() == 'quit':
#             break
# 
#         response = chatbot.generate_response(prompt)
# 
#         #for char in response:
#         #    print(char, end='', flush=True)
#         #    time.sleep(0.02)
#         #print()
