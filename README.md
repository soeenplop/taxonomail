# taxonomail

Suggests a folder structure for your mailbox.

taxonomail reads a Maildir or an IMAP folder, clusters the mails with
[BERTopic](https://maartengr.github.io/BERTopic/), lets an LLM name the clusters and merge
them into folders, and tells you which folder each mail belongs in. `run` only reads;
`classify --move` sorts mail into the folders (see [Sorting mail](#sorting-mail)).

## Usage

```sh
uv sync --frozen
uv run taxonomail run --maildir ~/Maildir --out result
uv run taxonomail run --imap-host imap.example.org --imap-user me --out result  # password in $IMAP_PASSWORD
```

The result directory holds `taxonomy.yaml` (the folders), `assignments.jsonl` (the folder
of each mail) and `clusters.json` (the clusters behind the folders).

## Sorting mail

Two commands sort mail into the folders of a finished run, with
[Jev](https://typesafe.ai/) (TypeSafe System One). Jev answers a choice question: it gets
the mail (sender domain, subject, first 1500 body characters) and every folder's name,
description and cluster descriptions from the run, and returns one of the folder ids with a
probability for each. It cannot return anything else, so a mail can't talk it into a new
folder. A mail whose best folder is less likely than `min_probability` stays where it is.

```sh
# Existing mail: list the folder of each mail in result/classified.jsonl, then move it.
uv run taxonomail classify --taxonomy result --imap-host imap.example.org --imap-user me
uv run taxonomail classify --taxonomy result --imap-host imap.example.org --imap-user me --move

# One new mail on stdin: prints the folder name (no newline) or nothing.
uv run taxonomail classify-mail --taxonomy result < mail.eml
```

- `classify` reads a Maildir (list only) or an IMAP folder. With `--move` it creates the
  folders it needs, subscribes to them and moves each mail with IMAP `MOVE`. A move counts
  as `"moved": true` only when the server confirms it (UIDPLUS `COPYUID`), otherwise
  `"unconfirmed"`. A mail Jev could not classify (after three retries) is listed with an
  error and left in place. A rejected key (HTTP 401/402/403), a wrong URL (404) or a batch
  of 100 mails that all fail stops the command.
- Folders get the readable names from `taxonomy.yaml`; `/`, `.`, `\`, `%` and `*` become
  `-` so a name never creates a subfolder. `--prefix "INBOX."` puts them under the inbox
  (write the server's hierarchy separator yourself).
- `classify-mail` never fails a delivery: on any error it prints nothing, writes the error
  type to stderr and exits 0, so the mail lands in the inbox. It does not retry, always
  reads the mail to the end and leaves mails over 50 MiB unclassified. Pass
  `--env-file` with the key, since mail filters usually run without your environment.

### Mail that is no longer relevant

`classify --stale-folder "Archive"` asks Jev a second, yes/no question in the same request:
is this mail no longer worth keeping? A mail goes to that folder instead of its topic folder
only when all of these hold, so anything in doubt stays:

- Jev's probability for "no longer worth keeping" is at least `min_probability` (0.8),
- the mail's Date header is at least `min_age_days` (30) days old, and
- the request succeeded.

Add `--only-stale` to move nothing but that: every other mail stays where it is, and the
topic folders are only listed in `classified.jsonl`, never created.

The topic does not matter here: a mail that is surely stale is moved even when Jev is
unsure about its topic folder. A mail whose Date header is missing, repeated or has no
known time zone is never moved there.

By default, no longer relevant means one-time and two-factor codes, login notifications,
newsletters and promotions, payment notifications, beta/TestFlight notifications,
registration confirmations and updated terms; invoices, contracts, tax and official mail,
bookings, orders, credentials and personal mail are worth keeping. The question and both
descriptions can be changed in the `[relevance]` section. `classify-mail` does not ask
this question, since a newly delivered mail is never old.

With Dovecot and Pigeonhole's
[extprograms](https://doc.dovecot.org/main/core/plugins/sieve_extprograms.html) plugin
(`sieve_execute_bin_dir` holds the wrapper script, `vnd.dovecot.execute` enabled):

```sh
#!/bin/sh
# taxonomail-classify, in sieve_execute_bin_dir
exec /path/to/taxonomail classify-mail --taxonomy /path/to/result --env-file /path/to/.env
```

```sieve
require ["vnd.dovecot.execute", "fileinto", "mailbox", "variables"];
if execute :pipe :output "folder" "taxonomail-classify" {
  if not string :is "${folder}" "" {
    fileinto :create "${folder}";
  }
}
```

Keep `[classification] timeout` below Dovecot's `sieve_execute_exec_timeout` (10 s by
default). Jev runs in the United States: every classified mail, including every new one,
is sent there.

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
   is sent back once with the problems listed. Folder ids still shared after that get a
   numeric suffix (`cafe`, `cafe-2`). If the folder count is still out of range, the run
   stops. A cluster the plan still leaves out goes, as a whole, to the folder whose
   mean vector is most similar (cosine) to the cluster's mean vector.
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

Every step (embedding, naming, consolidation) has its own `base_url` and `api_key_env`, so
any OpenAI-compatible endpoint works: OpenAI, Ollama, Scaleway, and so on. `api_key_env`
names the environment variable (upper case) that holds the step's key and defaults to
`OPENAI_API_KEY`. A step sends its key to its own `base_url`, so steps that share a
variable share that key: when you point a step at another provider, give it its own
variable too:

```toml
[embedding]
model = "qwen3-embedding-8b"
base_url = "https://api.scaleway.ai/v1"
api_key_env = "SCW_SECRET_KEY"
```

Set models, endpoints and key variables in a TOML file passed with `--config` (see
[taxonomail.example.toml](taxonomail.example.toml)) and the keys in `.env` (see
[.env.example](.env.example)). `.env` is read from the working directory or a
parent. IMAP always uses TLS.

Folder and cluster names and descriptions are in English unless you set another language,
either per run with `--language German` or as `language` under `[taxonomy]` in the config
file. Folder ids stay lowercase ASCII slugs.

Mail text (sender domain, subject, start of the body) is sent to the embedding service and,
for a few mails per cluster, to the LLM; `classify` and `classify-mail` send it to Jev. The LLM also gets each cluster's keywords and size,
and the generated cluster names and descriptions. The output, including `classified.jsonl`, contains
senders and subjects: keep it private.

## Development

```sh
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest
```
