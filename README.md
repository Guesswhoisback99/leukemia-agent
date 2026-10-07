# leukemia-agent

Agent for the PALG ALL7 treatment protocol. Ask questions in Polish, get answers with page and section citations.

Built with LangChain, AWS Bedrock (Titan v2 embeddings, Amazon Nova 2 Lite), Qdrant (hybrid dense + BM25
search) and FastAPI. How a question becomes an answer: [docs/ask-flow.md](docs/ask-flow.md).

## You need

- Docker, [uv](https://docs.astral.sh/uv/)
- An AWS profile in `~/.aws` with Bedrock access in `eu-central-1` to `amazon.titan-embed-text-v2:0`
  and `eu.amazon.nova-2-lite-v1:0`
- The protocol PDF saved as `data/protokol_palg_all7.pdf` (not included in this repo)

## Run

```sh
cp .env.example .env                       # set AWS_PROFILE if not "default"
docker compose up -d --build               # Qdrant + API
uv run python -m app.ingest                # load the PDF into Qdrant
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Jaka jest dawka metotreksatu w konsolidacji III?"}'
```

## Development notes

- **Intel Mac host:** PyTorch stops at 2.2.2 on macOS x86_64, so `pyproject.toml` pins `numpy<2` and
  `transformers<5` there only (`constraint-dependencies`). `docling-parse` has no wheel and builds from
  source on the first `uv sync` (about 15 minutes). The Linux image is not affected.
- **Docling device:** `DOCLING_DEVICE` in `app/ingest.py` is `cpu`. With `auto`, Docling picked MPS on an Intel
  Mac and silently produced chunks without headings or tables. If you change it, check the "with tables" count in
  the ingest log (on CPU: 134 chunks parsed, 49 with tables).

Scope, decisions and eval results: [docs/requirements/](docs/requirements/).
