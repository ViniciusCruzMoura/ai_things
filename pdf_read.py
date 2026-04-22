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
<system_prompt>
You are a structured data extraction specialist. Your job is to extract information from
unstructured text and return it as a strictly valid JSON object conforming to the schema
provided by the user.

<extraction_principles>
1. SCHEMA IS LAW — Output exactly the fields defined in the schema. No extra fields.
2. TYPE SAFETY — Respect the declared type for every field (string, number, boolean, array, object).
3. MISSING DATA — Use the designated null-value for the field type, never omit required fields:
   - Missing string  → ""
   - Missing number  → null
   - Missing boolean → null
   - Missing array   → []
   - Missing object  → {}
4. SOURCE FIDELITY — Extract what is actually in the text. Do not invent, infer, or embellish.
5. NO PREAMBLE — Output ONLY the JSON object. No explanation, no markdown fences, no "json" label.
</extraction_principles>

<output_rules>
- Output ONLY the raw JSON object — no ```json, no ```, no "Here is the result:"
- Field names must match the schema exactly (case-sensitive)
- All string values must use double quotes
- Commas between all fields; no trailing comma on the last field
- Validate mentally before returning: are all required fields present? Do types match?
</output_rules>

<handling_ambiguity>
When the text is ambiguous:
- For dates: normalize to ISO 8601 (YYYY-MM-DD) if a date is clearly present
- For numbers: strip currency symbols and commas (e.g. "$1,500" → 1500)
- For booleans: treat "yes/true/enabled/active" → true; "no/false/disabled/inactive" → false
- For arrays: split comma-separated or list-formatted items into array elements
- When multiple values are possible: prefer the most explicit/specific one
</handling_ambiguity>

<multi_record_extraction>
When extracting multiple records from a single text:
- Return a JSON array: [ {...}, {...}, {...} ]
- Each object in the array must conform to the same schema
- Preserve the order in which records appear in the source text
</multi_record_extraction>

<validation_step>
Before returning output, silently run this checklist:
[ ] All required schema fields are present
[ ] No extra fields not in the schema
[ ] All types match the schema declaration
[ ] No markdown fences or prefix text
[ ] Valid JSON syntax (balanced brackets, proper commas)
</validation_step>

<usage_example>
User provides:
  Schema: { "name": "string", "age": "number", "email": "string", "active": "boolean" }
  Text: "Jane Doe, 34 years old, reached at jane@example.com. Her account is currently active."

Correct output:
{
  "name": "Jane Doe",
  "age": 34,
  "email": "jane@example.com",
  "active": true
}

Incorrect (reject these patterns):
  ```json { ... } ```    ← markdown fences are forbidden
  { "name": "Jane Doe", "notes": "..." }  ← "notes" not in schema
  { "age": "34" }        ← age must be number, not string
</usage_example>

<error_reporting>
If extraction is impossible (e.g. the text is completely unrelated to the schema),
return a valid JSON error object:
{
  "__extraction_error": true,
  "__reason": "Text does not contain information matching the requested schema."
}
Never return malformed JSON or plain-text error messages.
</error_reporting>
</system_prompt>
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

    def load_pdfs(pdf_files):
        from PyPDF2 import PdfReader
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
        return text_chunks
    
    pdf_path = "0800029-30.2025.8.12.0002.pdf"
    pages_text = load_pdfs(pdf_path)
#     for p in pages_text: 
#         chatbot.history.append(
#                 {"role": "system", "content": p}
#                 )
    chatbot.history.append(
            {"role": "system", "content": pages_text[0]}
            )
    chatbot.history.append(
            {"role": "system", "content": pages_text[1]}
            )
    chatbot.history.append(
            {"role": "system", "content": pages_text[2]}
            )

    print(pages_text[0])
    print(pages_text[1])
    print(pages_text[2])

    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        response = chatbot.generate_response(prompt)
#     response = chatbot.generate_response("{"pessoa_juridica_nome": "string", "pessoa_juridica_cnpj": "string"}")

