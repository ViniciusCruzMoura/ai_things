import pandas as pd

INPUT_XLSX = "input.xlsx"
OUTPUT_XLSX = "output.xlsx"

df = pd.read_excel(INPUT_XLSX)

if "Movimentações" not in df.columns:
    raise KeyError("Column 'Movimentações' not found")

from pdf_rag import *
chatbot = QwenChatbot()
chatbot.rag = QwenRag(docs=docs)
def classify(text):
    prompt = text
    retrieved = chatbot.rag.retrieve(prompt)
    pclass = []
    for r in retrieved:
        m = re.search(r"<classificação>(.*?)</classificação>", r, re.IGNORECASE | re.DOTALL)
        if m:
            pclass.append(m.group(1))
    if pclass[0] != pclass[1]:
        from difflib import SequenceMatcher
        t = re.search(r"<texto>(.*?)</texto>", retrieved[0], re.IGNORECASE | re.DOTALL).group(1)
        st1 = SequenceMatcher(None, t, prompt).ratio()
        t = re.search(r"<texto>(.*?)</texto>", retrieved[1], re.IGNORECASE | re.DOTALL).group(1)
        st2 = SequenceMatcher(None, t, prompt).ratio()
        if st1 > 0.5 and st1 > st2:
            print(pclass[0])
            return pclass[0]
        elif st2 > 0.5 and st2 > st1:
            print(pclass[1])
            return pclass[1]
        elif st1 < 0.5 and st2 < 0.5:
            clean_history = chatbot.history.copy()
            retrieved = chatbot.rag.retrieve(prompt, 10)
            for i, r in enumerate(retrieved):
                chatbot.history.append({"role": "system", "content": "<retrieved>"+r+"</retrieved>"})
            response = chatbot.generate_response(prompt)
            try:
                print(re.search(r"<classificação>(.*?)</classificação>", response, re.IGNORECASE | re.DOTALL).group(1))
            except:
                return "-"
            chatbot.history = clean_history
            return re.search(r"<classificação>(.*?)</classificação>", response, re.IGNORECASE | re.DOTALL).group(1)
        else:
            return "-"
    else:
        from difflib import SequenceMatcher
        t = re.search(r"<texto>(.*?)</texto>", retrieved[0], re.IGNORECASE | re.DOTALL).group(1)
        similarity = SequenceMatcher(None, t, prompt).ratio()
        print(f"Similarity: {similarity:.2f}") # Output: 0.92
        return pclass[0]

df["Classificação"] = df["Movimentações"].apply(classify)

df.to_excel(OUTPUT_XLSX, index=False)
print(f"Wrote classifications to {OUTPUT_XLSX}")
