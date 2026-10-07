# Ask flow

How `POST /ask` turns a question into a cited answer. One pass, no loops: retrieve, prompt, generate.
Code: `app/api.py` (endpoint), `app/chain.py` (chain), `app/store.py` (hybrid store).

## Steps

```mermaid
flowchart TD
    U(["User question"]) --> API["POST /ask<br/>validate question + top_k"]
    API --> R

    subgraph R["1 · Retrieve: hybrid search in Qdrant"]
        direction LR
        D["Dense search<br/>Titan embedding<br/>matches meaning"]
        S["Sparse search<br/>BM25<br/>matches exact words"]
        F["RRF fusion<br/>merge both rankings<br/>keep top_k (default 5)"]
        D --> F
        S --> F
    end

    R --> Q{"Any chunks found?"}
    Q -- No --> REF["Fixed refusal<br/>Protokół nie zawiera informacji na ten temat."]
    Q -- Yes --> P["2 · Build prompt<br/>each chunk labelled [s. page, section]<br/>rules: use only context, cite, never invent doses"]
    P --> LLM["3 · Nova 2 Lite on Bedrock<br/>temperature 0"]
    LLM --> CL["4 · Clean up<br/>trim, m² m³ to m2 m3"]
    CL --> OUT
    REF --> OUT
    OUT(["Response<br/>answer + sources + model"])
```

## Who talks to whom

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI /ask
    participant Titan as Bedrock Titan<br/>(embeddings)
    participant Qdrant
    participant Nova as Bedrock Nova 2 Lite<br/>(LLM)

    User->>API: question
    API->>Titan: embed question
    Titan-->>API: dense vector
    Note over API: BM25 sparse vector<br/>computed locally (FastEmbed)
    API->>Qdrant: hybrid query (dense + sparse)
    Qdrant-->>API: top_k chunks, RRF-fused
    alt no chunks
        Note over API: fixed refusal, LLM not called
    else chunks found
        API->>Nova: system rules + labelled chunks + question
        Nova-->>API: answer with [s. page, section] citations
    end
    API-->>User: answer + sources (from the chunks) + model
```

## Worth knowing

- **Why two searches:** dense finds paraphrases, BM25 finds exact terms (drug names, section codes, units).
  RRF merges the two rankings by position. Hybrid won the retrieval experiments, see
  `requirements/01-rag-mvp.md`.
- **Refusals mostly come from the model.** The "no chunks" branch only fires on an empty collection; otherwise the
  prompt tells the model to answer with the exact refusal phrase when the context does not cover the question.
- **Two kinds of citations.** `[s. page, section]` inside `answer` is copied by the model from the chunk labels.
  `sources` is built from the retrieved chunks, never parsed from the model output.
- **Ingest is separate.** `app/ingest.py` parses the PDF with Docling and fills Qdrant; it is not part of this flow.
