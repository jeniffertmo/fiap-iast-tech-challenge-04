# Voice of Customer Intelligence com RAG

**Tech Challenge Fase 4** (IA para Devs / IAST). Solução RAG que responde perguntas em linguagem natural sobre as avaliações de clientes da Olist, citando as avaliações que sustentam cada afirmação e recusando perguntas sem evidência ou fora do escopo.

- **Integrantes:** Jeniffer Meireles, Rodrigo Alves de Souza e Saulo Freitas
- **Vídeo de apresentação:** [link]

---

## Arquitetura

```text
Pergunta
 └──> Validação e bloqueio (regex) ──── [sem LLM]
 └──> Cache por pergunta normalizada ──── [sem LLM]
 └──> Checagem de escopo (âncoras de embedding) ──── [sem LLM]
 └──> Detecção de polaridade (regex) -> Filtro por nota
 └──> Busca híbrida: vetorial (MiniLM) + BM25, fundidas por RRF
 └──> Checagem de evidência suficiente ──── [sem LLM]
 └──> Contexto compacto [R1..R6] + prompt com regras
 └──> LLM (1 chamada)
 └──> Validação de citações e lista de evidências
      └──> Resposta + review_ids citados
```

---

## Tecnologias

| Componente | Escolha | Motivo |
| :--- | :--- | :--- |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` | Multilíngue, 384 dims, rápido na T4 |
| **Busca lexical** | `rank-bm25` | Captura termos exatos ("rastreio", "nota fiscal") |
| **Fusão** | Reciprocal Rank Fusion ($k=60$) | Não exige calibrar escalas entre BM25 e cosseno |
| **Índice** | NumPy (busca exata) | ~40 mil vetores; aceita máscara de filtro direto |
| **LLM padrão** | `Qwen/Qwen2.5-3B-Instruct` local (fp16) | Custo zero, bom português, reprodutível sem chave |
| **LLM alternativo** | `gemini-2.5-flash-lite` via API compatível com OpenAI | Faixa de US$ 0,10 / 0,40 por milhão de tokens |

---

## Preparação da Base

- **Fonte:** `olist_order_reviews_dataset.csv` do Kaggle.
- **Colunas usadas:** `review_id`, `review_score`, `review_comment_title`, `review_comment_message`, `review_creation_date`.
- **Limpeza:** Nulos viram texto vazio, quebras de linha e espaços repetidos são normalizados, título e mensagem são unidos.
- **Descarte:** Avaliações sem texto são descartadas; `review_id` duplicado é removido.
- **Unidade de documento:** $1\text{ avaliação} = 1\text{ documento}$, sem chunking. As avaliações são curtas, e manter a unidade preserva `review_id` e nota para citação e filtro.
- **Sentimento:** Derivado da nota (1-2 negativo, 3 neutro, 4-5 positivo), sem custo de LLM.

---

## Decisões Técnicas

- **Uma chamada de LLM por pergunta, no máximo.** Perguntas inválidas, bloqueadas, fora do escopo, sem evidência ou repetidas são resolvidas antes do LLM.
- **Contexto compacto.** 6 avaliações, truncadas em 280 caracteres, rótulos `[R1]` no lugar do `review_id` (o ID real é recuperado depois).
- **Saída limitada e determinística.** 320 tokens no máximo, decodificação gulosa / temperatura 0.
- **Citações verificadas em código.** Citações a avaliações inexistentes são removidas; resposta sem nenhuma citação válida é rejeitada.
- **Defesa contra injeção.** Regex na pergunta e regra no prompt para tratar o texto das avaliações como dado.
- **Filtro de sentimento por metadado.** Perguntas sobre reclamações buscam só notas 1-2; sobre elogios, só 4-5.

---

## Exemplos

> *Copiar de `resultados/exemplos.json` após executar o notebook.*

| Pergunta | Rota | Trecho da resposta | Evidências |
| :--- | :--- | :--- | :--- |
| Quais são os principais problemas relatados pelos clientes? | | | |
| O que os clientes mais elogiam? | | | |
| Qual é a capital da França? | `fora_escopo` | | |

---

## Testes de Casos de Borda

A Etapa 5 do notebook roda 15 casos:
- Pergunta vazia
- Fora do escopo
- Injeção na pergunta e dentro de uma avaliação
- Dado pessoal
- Tema inexistente
- Pedido de percentual exato
- Palavras-chave soltas
- Texto sem acento
- Inglês
- Pergunta gigante
- Evidências contraditórias
- Cache

*(Colar aqui a tabela e o placar da execução).*

---

## Como Reproduzir

1. Abra `RAG_TECH_CHALLENGE_4.ipynb` no Google Colab com GPU T4.
2. Baixe o dataset no Kaggle e envie `olist_order_reviews_dataset.csv` para a pasta `data/`.
3. *(Opcional, para usar API)* Em Secrets do Colab, crie `GEMINI_API_KEY` e troque `BACKEND` para `"api"`.
4. Execute as células de setup e funções e depois a seção **"Fase 4"** em ordem.

> **Fora do Colab:** `pip install -r requirements.txt` e Python 3.10+ com GPU de pelo menos 8 GB para o modelo local.
