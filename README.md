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

## How it works

1. **Embed.** The embedding API receives each mail's sender domain, subject and first 1500
   body characters, capped together at 8000 UTF-8 bytes. The returned vectors are
   L2-normalized, so cosine similarity compares mails.
2. **Cluster.** [BERTopic](https://maartengr.github.io/BERTopic/) groups the vectors in
   two steps, then labels the groups (see [Clustering](#clustering) below).
3. **Name.** For each cluster, the naming model sees its keywords and up to `representatives`
   mails closest to the cluster's mean (sender domain, subject, first 500 body characters)
   and returns a name and description.
4. **Merge.** The consolidation model sees every cluster's name, description, size and
   keywords and groups the clusters into between `min_folders` and `max_folders` folders;
   the minimum is capped at the number of clusters. A plan with mapping errors (a cluster
   missing, unknown or in two folders), duplicate folder ids or too many or too few folders
   is sent back once with the problems listed. If the folder count is still out of range,
   the run stops. A cluster the plan still leaves out goes, as a whole, to the folder whose
   mails are most similar on average.
5. **Assign.** Each clustered mail goes to its cluster's folder. Each noise mail goes to the
   folder whose mean vector is closest to it (`method: nearest` in `assignments.jsonl`).

### Clustering

Density is hard to estimate in high-dimensional embedding space, so BERTopic first reduces
the vectors, then clusters the reduced ones:

- **[UMAP](https://umap-learn.readthedocs.io/)** builds a graph of cosine neighbors, with
  `n_neighbors` setting the neighborhood size, and lays it out in `n_components` dimensions
  so that neighbors stay close. Small `n_neighbors` keeps fine, local structure; large values
  favor the broad picture. `min_dist` is fixed at 0 so similar mails pack tightly, which
  helps the next step. `seed` makes the layout reproducible for the same mails in the same
  order, with the same embedding model and software versions.
- **[HDBSCAN](https://hdbscan.readthedocs.io/)** looks for regions where mails lie densely
  together, separated by sparser space. It builds a hierarchy of such regions and keeps the
  most stable ones that hold at least `min_cluster_size` mails. Unlike k-means, it does not
  need the number of clusters in advance, and it does not force every mail into a cluster:
  mails in sparse regions become noise (cluster `-1`). `min_cluster_size` sets the smallest
  allowed cluster and, since `min_samples` defaults to it, also how dense a region must be;
  lowering it allows smaller clusters, but its effect on the number of clusters and on noise
  depends on the mailbox.
- **Keywords**: BERTopic joins the mails of each cluster into one document and scores words
  with class-based TF-IDF (c-TF-IDF), which favors words frequent within the cluster and
  downweights words frequent across all mails. The top `keywords` words per cluster go to
  the naming model.

With fewer than three mails, or fewer than `min_cluster_size`, there is nothing to cluster;
everything is noise and the run stops. For small mailboxes the code caps `n_neighbors` at
n − 1 and `n_components` at n − 2 (n = number of mails) and then starts UMAP from a random
instead of a spectral layout. All settings are in the `[clustering]` section of the config.

## Configuration

Embeddings come from any OpenAI-compatible embeddings API; the LLM is Claude or any
OpenAI-compatible chat model. Set models and endpoints in a TOML file passed with `--config`
(see [taxonomail.example.toml](taxonomail.example.toml)) and API keys in `.env` (see
[.env.example](.env.example)). By default `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` go only
to the official APIs; a custom `base_url` gets only the key named in `api_key_env`
(whichever variable you name there). `.env` is read from the working directory or a parent.
IMAP always uses TLS.

Mail text (sender domain, subject, start of the body) is sent to the embedding service and,
for a few mails per cluster, to the LLM. The LLM also gets each cluster's keywords and size,
and the generated cluster names and descriptions. The output contains senders and subjects:
keep it private.

## Development

```sh
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest
```
