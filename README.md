# leukemia-agent

Agent for the PALG ALL7 treatment protocol. Ask questions in Polish, get answers with page and section citations.

Built with LangChain, AWS Bedrock (Titan v2 embeddings, Amazon Nova 2 Lite), Qdrant and FastAPI.

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

## Docs

- [Requirements and decisions](docs/requirements/)
- [Implementation plans](docs/plans/)
