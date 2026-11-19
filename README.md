# taxonomail

Suggests a folder structure for your mailbox.

taxonomail reads a Maildir or an IMAP folder, clusters the mails with
[BERTopic](https://maartengr.github.io/BERTopic/), lets an LLM name the clusters and merge
them into folders, and tells you which folder each mail belongs in. It only reads; it
never moves or changes mail.

## Usage

```sh
uv sync --frozen
uv run taxonomail run --maildir ~/Maildir --out result
uv run taxonomail run --imap-host imap.example.org --imap-user me --out result  # password in $IMAP_PASSWORD
```

The result directory holds `taxonomy.yaml` (the folders), `assignments.jsonl` (the folder
of each mail) and `clusters.json` (the clusters behind the folders).

## Configuration

Embeddings come from any OpenAI-compatible embeddings API; the LLM is Claude or any
OpenAI-compatible chat model. Set models and endpoints in a TOML file passed with
`--config` (see [taxonomail.example.toml](taxonomail.example.toml)) and API keys in `.env`
(see [.env.example](.env.example)).

Mail text (sender domain, subject, start of the body) is sent to the embedding service and,
for a few mails per cluster, to the LLM. The output contains senders and subjects: keep it
private.

## Development

```sh
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest
```
