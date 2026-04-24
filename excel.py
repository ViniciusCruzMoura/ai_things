from pdf_rag import ClassifyAgent
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
agent = ClassifyAgent()
INPUT_XLSX = "movimentacoes_20260423_130457_classificada.xlsx" #"input.xlsx"
OUTPUT_XLSX = "2_movimentacoes_20260423_130457_classificada.xlsx" #"output.xlsx"

df = pd.read_excel(INPUT_XLSX)

if "Movimentações" not in df.columns:
    raise KeyError("Column 'Movimentações' not found")

def classify(text):
    classification = agent.run(text)
    print("Classificação >>", classification)
    return classification

# df["Classificação"] = df["Movimentações"].apply(classify)
df["Classificação"] = df["Movimentações"].progress_apply(classify)

df.to_excel(OUTPUT_XLSX, index=False)
print(f"Wrote classifications to {OUTPUT_XLSX}")
