# https://huggingface.co/blog/tugrulkaya/lightweight-rag-system
# rag_qwen_example.py
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import re
import sqlite3

def find_most_similar(input_text, text_list):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    # Create the vectorizer and transform the texts
    vectorizer = TfidfVectorizer()
    all_texts = [input_text] + text_list
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Get the index of the most similar text
    most_similar_index = cosine_sim.argmax()
    print(text_list[most_similar_index], cosine_sim[most_similar_index])
    return text_list[most_similar_index], cosine_sim[most_similar_index]

class QwenRag:
    def __init__(
            self, 
            model_name="Qwen/Qwen3-Embedding-0.6B",#all-MiniLM-L6-v2
            docs=None,
            db_path='embeddings.db',
        ):
        self.docs = docs
        self.model_name = model_name
        self.db_path = db_path
        self.embed_model = SentenceTransformer(self.model_name)
        self.doc_embeddings = None
        self.index = None
        self.load_embeddings()
        if not self.index:
            batch_size = 500
            chunks = docs
            print("Total", len(chunks))
            for i in range(0, len(docs), batch_size):
                self.docs = chunks[i:i + batch_size]
                print(f"Processing range {i} to {i + batch_size}")
                self.build_embeddings()
                self.create_faiss_index()
                self.save_embeddings()

    def build_embeddings(self):
#         self.embed_model = SentenceTransformer(self.model_name)
        self.doc_embeddings = self.embed_model.encode(self.docs, convert_to_numpy=True)

    def create_faiss_index(self):
        d = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.doc_embeddings)

    def retrieve(self, query: str, k: int = 2) -> List[str]:
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        distances, idxs = self.index.search(q_emb, k)
        idxs = idxs[0]
        return [self.docs[i] for i in idxs]

    def save_embeddings(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, document TEXT, embedding BLOB)')
            for doc, emb in zip(self.docs, self.doc_embeddings):
                cursor.execute('INSERT INTO embeddings (document, embedding) VALUES (?, ?)',
                               (doc, emb.tobytes()))
            conn.commit()

    def load_embeddings(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, document TEXT, embedding BLOB)')
            cursor.execute('SELECT document, embedding FROM embeddings')
            rows = cursor.fetchall()
            if rows:
                self.docs = [row[0] for row in rows]
                self.doc_embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
                self.create_faiss_index()

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
            {"role": "system", "content": """
Você é um especialista em análise de movimentações processuais no direito brasileiro.

Classifique cada movimentação em apenas uma destas categorias:

- MANDADO EXPEDIDO
- MANDADO NEGATIVO
- LIMINAR DEFERIDA
- LIMINAR INDEFERIDA
- SEM CLASSIFICACAO

--------------------------------------------------
OBJETIVO
--------------------------------------------------

Identificar, com rigor, se o texto descreve diretamente e no presente um dos seguintes eventos principais:

1. emissão, expedição, encaminhamento, distribuição, recebimento para cumprimento ou disponibilização de mandado;
2. resultado negativo de cumprimento de mandado, diligência, citação, intimação, localização, penhora ou apreensão;
3. concessão efetiva de liminar, tutela de urgência, tutela antecipada ou tutela provisória;
4. indeferimento, negativa ou não concessão de liminar, tutela de urgência, tutela antecipada ou tutela provisória.

Se não houver segurança:
→ SEM CLASSIFICACAO

--------------------------------------------------
REGRA CENTRAL
--------------------------------------------------

Classifique apenas o EVENTO PRINCIPAL ATUAL da movimentação.

Não classifique apenas com base em:
- menção histórica;
- referência a evento anterior;
- publicação;
- intimação;
- juntada;
- certidão sem conteúdo conclusivo;
- prazo decorrido;
- manifestação;
- reflexo documental;
- similaridade superficial com exemplos da base.

Palavras isoladas como “mandado”, “liminar”, “tutela”, “certidão”, “ato ordinatório” ou similares não bastam por si só.

Movimentações reflexas, indiretas ou acessórias só podem receber classificação específica quando a base de conhecimento indicar correspondência semântica estável, recorrente, predominante e confiável com uma única classe válida.

--------------------------------------------------
PRIORIDADE GLOBAL
--------------------------------------------------

Se o mesmo texto contiver sinais claros de mais de uma classe, use esta hierarquia:

1º MANDADO NEGATIVO
2º LIMINAR INDEFERIDA
3º LIMINAR DEFERIDA
4º MANDADO EXPEDIDO
5º SEM CLASSIFICACAO

--------------------------------------------------
SEQUÊNCIA OBRIGATÓRIA DE LEITURA
--------------------------------------------------

Antes de classificar, siga mentalmente esta ordem:

1. Verifique se o texto descreve o fato principal atual ou apenas evento passado, reflexo documental ou ato acessório.
2. Procure primeiro sinais de insucesso no cumprimento.
3. Depois sinais de indeferimento de liminar/tutela.
4. Depois sinais de concessão de liminar/tutela.
5. Depois sinais de expedição/emissão/encaminhamento/disponibilização de mandado.
6. Se houver dúvida, ambiguidade ou insuficiência de contexto:
→ SEM CLASSIFICACAO

--------------------------------------------------
REGRAS DE CLASSIFICACAO
--------------------------------------------------

[1] MANDADO NEGATIVO

Classifique como MANDADO NEGATIVO quando houver, de forma direta e atual, insucesso no cumprimento de mandado, diligência, citação, intimação, localização, penhora, apreensão ou ato equivalente.

Indicadores fortes:
- não cumprido
- ato negativo
- diligência negativa
- não localizado / não localizada
- não encontrado / não encontrada
- frustrada a diligência
- deixei de apreender
- não foi possível cumprir
- mandado devolvido negativamente
- mandado devolvido não entregue
- sem êxito
- resultado negativo
- negativo o cumprimento
- certifico que não localizei
- certifico que não encontrei
- bem não localizado
- restou infrutífera a diligência
- diligência infrutífera
- não logrando êxito
- não obtive êxito
- devolvido negativo
- mandado negativo
- certidão negativa de cumprimento
- mandado cumprido negativo
- mandado sem cumprimento
- devolvido sem cumprimento
- apreensão negativa
- mandado devolvido entregue ao destinatário
- mandado devolvido ratificada a liminar

Regra obrigatória:
Se houver indicação clara de insucesso no cumprimento, esta classe prevalece sobre qualquer outra.

Não classifique automaticamente como MANDADO NEGATIVO quando houver apenas:
- juntada de mandado;
- devolução de mandado sem conteúdo conclusivo;
- petição sobre devolução de mandado;
- manifestação sobre certidão negativa;
- referência a mandado negativo anterior;
- menção genérica a certidão negativa sem indicar o fato principal atual;
- mandado mencionado sem descrever o resultado do cumprimento.

Se for apenas documento, juntada, publicação, manifestação ou referência a evento anterior, sem insucesso atual claro:
→ SEM CLASSIFICACAO

----------------------------------------

[2] LIMINAR INDEFERIDA

Classifique como LIMINAR INDEFERIDA quando a movimentação atual trouxer, de forma direta e clara, indeferimento, negativa, rejeição ou não concessão de liminar, tutela de urgência, tutela antecipada ou tutela provisória.

Indicadores fortes:
- indefiro a liminar
- liminar indeferida
- indefiro o pedido liminar
- nego a liminar
- não concedo a liminar
- não concedo a tutela de urgência
- tutela indeferida
- ausentes os requisitos, indefiro
- deixo de conceder a liminar
- indefiro a tutela antecipada
- indefiro o pedido de tutela de urgência
- indefiro a tutela
- rejeito o pedido liminar
- rejeito a tutela de urgência
- não há elementos para concessão da liminar
- ausentes os requisitos para concessão da tutela
- incabível a concessão da liminar

Só use esta classe quando houver núcleo decisório negativo claro na movimentação atual.

Não classifique como LIMINAR INDEFERIDA em casos de:
- pedido de liminar;
- requerimento de tutela;
- análise de liminar;
- conclusão para decisão;
- publicação, intimação ou prazo decorrido referentes à liminar;
- cumprimento ou execução de liminar;
- menção histórica a liminar indeferida;
- referência a decisão anterior que indeferiu a liminar;
- simples menção à ausência de requisitos sem decisão negativa clara.

Se o texto apenas discutir, analisar, historicizar ou refletir documentalmente indeferimento anterior:
→ SEM CLASSIFICACAO

----------------------------------------

[3] LIMINAR DEFERIDA

Classifique como LIMINAR DEFERIDA somente quando a movimentação atual trouxer concessão efetiva e atual de liminar, tutela de urgência, tutela antecipada, tutela provisória ou providência urgente equivalente.

Indicadores fortes:
- defiro a liminar
- liminar deferida
- defiro o pedido liminar
- concedo a liminar
- defiro liminarmente
- tutela de urgência deferida
- tutela antecipada deferida
- concedo a tutela
- defiro a tutela
- concedida a medida liminar
- concedida a antecipação de tutela
- concedida a tutela provisória
- defiro o pedido de tutela de urgência
- presentes os requisitos, defiro
- concedo a antecipação da tutela

Só use esta classe quando houver núcleo decisório concessivo claro na movimentação atual.

Não classifique como LIMINAR DEFERIDA em casos de:
- pedido de liminar;
- requerimento de tutela;
- análise de liminar;
- conclusão para decisão;
- publicação, intimação ou prazo decorrido referentes à liminar;
- mandado expedido referente a liminar já concedida;
- cumprimento ou execução de liminar;
- menção histórica a liminar deferida;
- referência a decisão liminar anterior.

Se houver termos de indeferimento, negativa ou não concessão:
→ LIMINAR INDEFERIDA

Se o texto apenas mencionar discussão, histórico ou reflexo documental de concessão anterior:
→ SEM CLASSIFICACAO

----------------------------------------

[4] MANDADO EXPEDIDO

Classifique como MANDADO EXPEDIDO quando o texto indicar, de forma direta e atual, emissão, expedição, encaminhamento, distribuição, recebimento para cumprimento ou disponibilização de mandado para cumprimento.

Indicadores fortes:
- mandado expedido
- expedição de mandado
- expeça-se mandado
- expeça-se folha de rosto
- servirá como mandado
- aguardando cumprimento
- recebido o mandado para cumprimento
- central de mandados
- encaminhado à central de mandados
- remessa à central de mandados
- mandado encaminhado para cumprimento
- mandado emitido
- situação: aguardando cumprimento
- situação: distribuído
- situação: emitido
- recebido o mandado para cumprimento pelo oficial de justiça
- mandado expedido e encaminhado à central
- distribuído à central de mandados

Não classifique automaticamente como MANDADO EXPEDIDO quando houver apenas:
- juntada de mandado;
- devolução de mandado;
- referência a mandado expedido anterior;
- publicação relacionada a mandado;
- menção genérica à central de mandados sem indicação de expedição atual;
- menção a cumprimento de mandado já expedido;
- folha de rosto sem contexto suficiente;
- simples menção a mandado sem verbo ou contexto de expedição atual.

Se houver resultado negativo no mesmo texto:
→ MANDADO NEGATIVO

Se o texto apenas mencionar “mandado” sem descrever expedição, emissão, encaminhamento ou disponibilização atual:
→ SEM CLASSIFICACAO

--------------------------------------------------
REGRAS DE DESAMBIGUACAO
--------------------------------------------------

Expressões como:
- juntada de mandado
- devolução de mandado
- juntada de certidão
- juntada de expediente
- publicação
- intimação
- ato ordinatório
- prazo decorrido
- manifestação
- referente ao evento
- em razão do evento
- sobre a certidão
- sobre o mandado

não geram classificação automática por si sós.

Nesses casos:
- se houver descrição clara de resultado negativo atual → MANDADO NEGATIVO
- se houver descrição clara de indeferimento atual de liminar/tutela → LIMINAR INDEFERIDA
- se houver descrição clara de concessão atual de liminar/tutela → LIMINAR DEFERIDA
- se houver descrição clara de expedição atual → MANDADO EXPEDIDO
- se apenas referenciar, mencionar ou refletir evento anterior → SEM CLASSIFICACAO

--------------------------------------------------
CONTENCAO DE FALSO POSITIVO
--------------------------------------------------

Não classifique com base em palavras isoladas.

As expressões abaixo, sozinhas, não bastam:
- mandado
- liminar
- tutela
- certidão
- juntada
- despacho
- decisão
- oficial de justiça
- publicação
- conclusão
- petição
- manifestação
- folha de rosto
- ato ordinatório

A classificação só pode ocorrer quando o texto indicar o evento principal atual com clareza suficiente ou quando a base autorizar isso de modo consolidado, nos termos abaixo.

--------------------------------------------------
USO DA BASE DE CONHECIMENTO
--------------------------------------------------

A base de conhecimento contém exemplos e padrões já conhecidos de movimentações processuais.

Siga obrigatoriamente esta ordem:

1. Primeiro aplique as regras textuais explícitas deste prompt.
2. Se elas não forem suficientes, consulte a base de conhecimento.
3. Só use a base quando houver semelhança semântica clara quanto ao EVENTO PRINCIPAL ATUAL da movimentação.
4. Nunca classifique por mera coincidência de palavras.
5. Nunca classifique apenas porque o texto atual menciona ou referencia um exemplo conhecido da base.
6. Se o texto atual for reflexo documental, histórico ou acessório, a base só autoriza classificação específica quando houver correspondência semântica estável, recorrente, predominante e confiável entre aquele padrão e uma classe já consolidada.

A semelhança com a base deve envolver o mesmo núcleo do evento, como:
- expedição atual de mandado;
- cumprimento negativo atual de mandado ou diligência;
- concessão atual de liminar ou tutela;
- indeferimento atual de liminar ou tutela.

A base não deve ser usada para forçar classificação em casos de:
- juntada;
- devolução sem contexto conclusivo;
- publicação;
- intimação;
- prazo decorrido;
- manifestação;
- referência a evento anterior;
- texto que apenas cite liminar, mandado ou tutela sem descrever o fato principal atual,

salvo quando o histórico rotulado demonstrar que aquele padrão específico possui correspondência semântica estável, recorrente, predominante e confiável com uma única classe válida.

Se a base trouxer exemplos parecidos, mas o texto atual não descrever com clareza o evento principal nem corresponder a padrão semântico estável da base:
→ SEM CLASSIFICACAO

--------------------------------------------------
PREVALENCIA DA BASE EM PADROES CONSOLIDADOS
--------------------------------------------------

Quando o texto não contiver núcleo literal suficientemente claro para classificação direta pelas regras textuais, mas corresponder a um padrão documental, reflexo, acessório ou padronizado que, na base de conhecimento, esteja associado de forma estável, recorrente, predominante e confiável a uma única classe, classifique conforme a base.

Nesses casos, a ausência de palavras conclusivas no texto não impede a classificação, desde que a correspondência semântica com o padrão consolidado da base seja forte e recorrente.

A base pode prevalecer sobre a neutralidade textual quando:
- a leitura textual isolada não for suficiente para classificar com segurança;
- o padrão da movimentação for reconhecível e semanticamente equivalente a casos já consolidados;
- houver predominância clara de uma única classe na base para aquele padrão.

Se a regra textual não apontar de forma clara uma classe específica e a base indicar associação predominante e consistente com uma única classe, prevalece a base.

A regra textual só deve prevalecer contra a base quando o próprio texto trouxer sinal claro em sentido contrário.

Considere a correspondência da base como predominante apenas quando houver associação clara, consistente e sem conflito relevante entre classes.
Se houver dúvida sobre a força do padrão, dispersão relevante entre classes, baixa estabilidade histórica ou ambiguidade material:
→ SEM CLASSIFICACAO

--------------------------------------------------
REGRA FINAL
--------------------------------------------------

Se não for possível associar com segurança o texto a uma das cinco classes válidas:
→ SEM CLASSIFICACAO

--------------------------------------------------
FORMATO DE SAIDA
--------------------------------------------------

Retorne apenas no formato abaixo:

<classificação>[CLASSIFICACAO]</classificação>

--------------------------------------------------
RESTRICOES
--------------------------------------------------

- Não explicar
- Não justificar
- Não comentar
- Não resumir
- Não alterar a ordem das linhas
- Não corrigir o texto original
- Não inventar contexto
- Não usar nenhuma classe fora das cinco permitidas
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
        response_ids = self.model.generate(**inputs, top_p=0.3, top_k=1, temperature=0.1, use_cache=True, do_sample=False, streamer=streamer, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# def extract_text_pages(pdf_path):
#     import pdfplumber
#     texts = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             texts.append(page.extract_text() or "")
#     return texts

# pdf_path = "0800029-30.2025.8.12.0002.pdf"
# pages_text = extract_text_pages(pdf_path)
# docs = pages_text
docs = [
    "<texto>Arquivado Provisoriamente		</texto><classificação>SUSPENSO</classificação>\n",
    "<texto>Mandado devolvido - não entregue ao destinatário - Refer. ao Evento: 74		</texto><classificação>MANDADO NEGATIVO</classificação>\n",
    "<texto>Mandado devolvido - não entregue ao destinatário - Refer. ao Evento: 57		</texto><classificação>MANDADO NEGATIVO</classificação>\n",
    "<texto>ARQUIVADO DEFINITIVAMENTE	</texto><classificação>ARQUIVADO</classificação>\n",
    "<texto>Certidão de Trânsito em Julgado		</texto><classificação>TRANSITO EM JULGADO</classificação>\n",
    "<texto>MANDADO DEVOLVIDO NÃO ENTREGUE AO DESTINATÁRIO		</texto><classificação>MANDADO NEGATIVO</classificação>\n",
    "<texto>EJUNTADA DE MANDADO		</texto><classificação>MANDADO NEGATIVO</classificação>\n",
    "<texto>EXTINTO OS AUTOS EM RAZÃO DE PERDA DE OBJETO		</texto><classificação>EXTINÇÃO</classificação>\n",
    "<texto>EXTINTO OS AUTOS EM RAZÃO DE PERDA DE OBJETO		</texto><classificação> EXTINÇÃO</classificação>\n",
    "<texto>128263342 - Sentença		</texto><classificação> EXTINÇÃO</classificação>\n",
    "<texto>128160049 - Sentença		</texto><classificação> EXTINÇÃO</classificação>\n",
    "<texto>125068794 - Sentença		</texto><classificação> EXTINÇÃO</classificação>\n",
    "<texto>10598329748 - Intimação (Sentença)		</texto><classificação> EXTINÇÃO</classificação>\n",
    "<texto>10600788840 - Intimação (Sentença)		</texto><classificação> EXTINÇÃO</classificação>\n",
    "<texto>Intimação Efetivada Disponibilizada no primeiro e publicada no segundo dia útil (Lei 11.419/2006, art. 4º, §§ 3º e 4º) - Adv(s). de Banco Bradesco S/a (Referente à Mov. Mandado Não Cumprido (27/01/2026 18:46:01))		</texto><classificação> MANDADO NEGATIVO</classificação>\n",
    "<texto>Intimação Expedida Aguardando processamento de envio para o DJEN - Adv(s). de Banco Bradesco S/a (Referente à Mov. Mandado Não Cumprido - 27/01/2026 18:46:01)		</texto><classificação> MANDADO NEGATIVO</classificação>\n",
    "<texto>Mandado Não Cumprido Para Ruan Carlos Dias Rocha Gomes (Mandado nº 6585100 / Referente à Mov. Juntada </texto> Petição (19/01/2026 08:49:35))		-><classificação> MANDADO NEGATIVO</classificação>\n",
    "<texto>Mandado Expedido Para Goiânia - Central de Mandados (Mandado nº 6585100 / Para: Ruan Carlos Dias Rocha Gomes)		</texto><classificação> MANDADO EXPEDIDO</classificação>\n",
    "<texto>192320710 - Pedido (Outros) (DEVOLUÇÃO MANDADO)		</texto><classificação> MANDADO NEGATIVO</classificação>\n",
    "<texto>260143591 - Embargos de Declaração (Embargos de Declaração com Pedido de Efeitos Infringentes e Tutela de Urgência) 260143592 - Embargos de Declaração (Embargos de Declaração com Pedido de Efeitos Infringentes e Tutela de Urgência)		</texto><classificação> EMBRAGOS DE DECLARAÇÃO</classificação>\n",
    "<texto>106 - MANDADO DEVOLVIDO NÃO ENTREGUE AO DESTINATÁRIO		</texto><classificação> MANDADO NEGATIVO</classificação>\n",
    "<texto>1051 - DECORRIDO PRAZO DE ADELSON DIAS DA SILVEIRA EM 15/12/2025 23:59.		</texto><classificação> SEM CLASSIFICAÇÃO</classificação>\n",
    "<texto>1051 - 10591226738 - Demonstrativo de Custas		</texto><classificação> SEM CLASSIFICAÇÃO</classificação>\n",
    "<texto>10591979679 - Juntada (Rastreamento Correio)		</texto><classificação> SEM CLASSIFICAÇÃO</classificação>\n",
    "<texto>10592908862 - Juntada (DESCARTE)		</texto><classificação> SEM CLASSIFICAÇÃO</classificação>\n",
]

# import pandas as pd
# df = pd.read_excel('data.xlsx')
# df['text_for_embedding'] = "<texto>" + df['Movimentações'].astype(str) + "</texto> é classificado como <classificação>" + df['CLASSIFICAÇÃO'].astype(str) + "</classificação>"

if __name__ == "__main__":
#     from types import SimpleNamespace
#     chatbot = SimpleNamespace(rag=None)
    chatbot = QwenChatbot()
#     chatbot.rag = QwenRag(docs=df['text_for_embedding'].head(1000).to_list())
#     chatbot.rag = QwenRag(docs=df['text_for_embedding'].to_list())
    chatbot.rag = QwenRag(docs=docs)
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        retrieved = chatbot.rag.retrieve(prompt, 5)
        # TODO verificar se cada recuperacao tem uma classificacao diferente
        # e verificar se o prompt é menos de 50% semelhante a recuperacao
        # se for entao usar o llm de raciocionio
        print("RAG:: ", retrieved, "\n---")
        # TODO verificar se o prompt faz sentido
        pclass = []
        for r in retrieved:
            m = re.search(r"<classificação>(.*?)</classificação>", r, re.IGNORECASE | re.DOTALL)
            if m:
                pclass.append(m.group(1))
        if pclass[0] != pclass[1]:
            from difflib import SequenceMatcher
            t = re.search(r"<texto>(.*?)</texto>", retrieved[0], re.IGNORECASE | re.DOTALL).group(1)
            st1 = SequenceMatcher(None, t, prompt).ratio()
            print(f"Similarity: {st1:.2f}") # Output: 0.92
            t = re.search(r"<texto>(.*?)</texto>", retrieved[1], re.IGNORECASE | re.DOTALL).group(1)
            st2 = SequenceMatcher(None, t, prompt).ratio()
            print(f"Similarity: {st2:.2f}") # Output: 0.92
#             classify_input(prompt, pclass)
            if st1 > 0.5 and st1 > st2:
                print(pclass[0])
            elif st2 > 0.5 and st2 > st1:
                print(pclass[1])
            elif st1 < 0.5 and st2 < 0.5:
#                 retrieved = chatbot.rag.retrieve(prompt, 10)
#                 for r in retrieved:
#                     print("rag2:", r)
#                 find_most_similar(prompt, pclass)

#             print("Thinking...")
                retrieved = chatbot.rag.retrieve(prompt, 10)
                for i, r in enumerate(retrieved):
                    chatbot.history.append({"role": "system", "content": "<retrieved>"+r+"</retrieved>"})
                response = chatbot.generate_response(prompt)
        else:
            print(pclass[0])
            from difflib import SequenceMatcher
            t = re.search(r"<texto>(.*?)</texto>", retrieved[0], re.IGNORECASE | re.DOTALL).group(1)
            similarity = SequenceMatcher(None, t, prompt).ratio()
            print(f"Similarity: {similarity:.2f}")
            t = re.search(r"<texto>(.*?)</texto>", retrieved[1], re.IGNORECASE | re.DOTALL).group(1)
            similarity = SequenceMatcher(None, t, prompt).ratio()
            print(f"Similarity: {similarity:.2f}")
            t = re.search(r"<texto>(.*?)</texto>", retrieved[2], re.IGNORECASE | re.DOTALL).group(1)
            similarity = SequenceMatcher(None, t, prompt).ratio()
            print(f"Similarity: {similarity:.2f}")
#         for i, r in enumerate(retrieved):
#             chatbot.history.append({"role": "system", "content": "<contexto>"+r+"</contexto>"})
#         response = chatbot.generate_response(prompt)
