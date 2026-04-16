import sqlite3 
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CPJ_DB = 'uriel.db'

_001_create_doc_embeddings = """
CREATE TABLE IF NOT EXISTS DOC_EMBEDDINGS (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    DOC TEXT,
    EMBEDDING BLOB
);
CREATE INDEX IF NOT EXISTS DOC_EMBEDDINGS_001_IDX ON DOC_EMBEDDINGS (
    ID, DOC
);
"""

migrate = [
    _001_create_doc_embeddings, 
]

def sqlite_execute(sql):
    try:
        with sqlite3.connect(CPJ_DB) as conn:
            cursor = conn.cursor()
            cursor.executescript(sql)
            conn.commit()
    except sqlite3.OperationalError as e:
        print("Failed to create tables:", e)

def sqlite_migrate():
    for query in migrate:
        sqlite_execute(query)

def sqlite_insert(table, columns, values):
    sql = f"INSERT INTO {table} ({columns}) VALUES ({values});"
    sqlite_execute(sql)

def sqlite_embedding_insert(d):
    columns = ",".join([f"'{str(k).strip().upper()}'" for k in d.keys()])
    values = ",".join([f"'{str(d.get(k)).strip().upper()}'" for k in d.keys()])
    sqlite_insert("DOC_EMBEDDINGS", columns, values)

sqlite_migrate()

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

embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")#"all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(docs, convert_to_numpy=True)

for doc, embedding in zip(docs, doc_embeddings):
    embedding_blob = embedding.tobytes()  # Convert numpy array to bytes
    d = {"doc": doc, "embedding": doc_embeddings}
    sqlite_embedding_insert(d)

with sqlite3.connect(CPJ_DB) as conn:
    cursor = conn.cursor()
    cursor.executescript("SELECT embedding FROM DOC_EMBEDDINGS")
    embeddings = np.array([np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()])

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    print(index)
