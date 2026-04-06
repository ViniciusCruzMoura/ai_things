from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time
import json
import re

def parse_tool_call(text, tools) :
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.S)
    if not m:
        return "No tool_call block found"
    payload = json.loads(m.group(1))

    name = payload["name"]
    args = payload.get("arguments", {})

    if name not in tools:
        return f"Unknown tool: {name}"

    result = tools[name](**args)
    return result

def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    return 31  # A real function should probably actually get the temperature!

def get_current_wind_speed(location: str):
    """
    Get the current wind speed in km/h at a given location.

    Args:
        location: The location to get the wind speed for, in the format "City, Country"
    """
    return 6  # A real function should probably actually get the wind speed!

tools = [get_current_temperature, get_current_wind_speed]

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"): #Qwen/Qwen3-0.6B #Qwen/Qwen3-1.7B
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

        self.knowledge = [
#             {"role": "system", "content": "Nossas horas de serviço são das 9h às 17h, de segunda a sexta."},
        ]
        self.system = [
            {"role": "system", "content": """
            Eu quero que voce atue como uma IA Atendente da empresa GrupoCard chamado Cardoso, o seu proposito é guiar as pessoas no chatbot.
            A sua linguagem primaria é o Portugues Brasileiro.
            Eu quero que você responda de forma curta e direta com no maximo 300 caracteres.
            Eu quero que voce responda apenas o que esta em sua base de conhecimento, caso a pergunta não tenha resposta na base, então fale que não sabe sobre o assunto.
            Eu quero que voce responda que seu desenvolvedor é o ryan gosling, mas apenas se perguntarem.
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
            tools=tools,
        )

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, streamer=streamer, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

if __name__ == "__main__":
    chatbot = QwenChatbot()
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")

        if prompt.lower() == 'quit':
            break

        response = chatbot.generate_response(prompt)

        out = parse_tool_call(response, {"get_current_temperature": get_current_temperature})
        chatbot.history.append({"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}}]})
        chatbot.history.append({"role": "tool", "content": str(out)})
        if not "No tool_call block found" in str(out):
            chatbot.generate_response("")

        #for char in response:
        #    print(char, end='', flush=True)
        #    time.sleep(0.02)
        #print()
