# Entity resolution in the existing ingestion pipeline

The resolver lives in `pipeline/common/entity_resolver.py`. It reads
`pipeline/common/entity_aliases.json`, whose entries must provide an HTTPS
authority URL and evidence URLs for the Korean and English names. The opaque
`entity_id` is derived from the authority URL and entity type, never from a
mention string. The initial catalogue contains BTS and BLACKPINK, backed by
their agencies' profiles. Expand the catalogue only after checking the source;
a translated title alone must stay `unresolved` or `translated_only`.

Supported types are Artist, Group, Person, Work, Drama, Movie, Album, Song,
Fandom, Platform, Company and Topic. A matching alias without an explicit type
is left unresolved when it belongs to more than one catalogue entry. Upstream
entity extraction can pass `entities` with `surface_form` and `entity_type`.
The resolver also recognizes catalogue aliases in source text and preserves
quoted Korean title candidates as unresolved when no verified alias exists.

The news path is `collect -> cleaner -> entity_resolver -> translator ->
PostgreSQL loader`. The translator rechecks the catalogue, protects confirmed
names with markers, rejects a translation that loses a marker, then restores
the confirmed English name. It leaves `title_original` and `content_original`
unchanged. The paper path is `collect -> download -> extractor ->
entity_resolver -> PostgreSQL loader`. The existing PostgreSQL loader ignores
the additional `entities` JSON member, since the deployed `papers` and `news`
tables have no entity JSON column. No PostgreSQL schema change is required.

Graph sync reads the loaded PostgreSQL rows. It keeps the existing Paper,
Author, Keyword and Source graph logic, adds a News node keyed by PostgreSQL
`news_id`, and adds `MENTIONS` edges for resolved entities only. Every resolved
entity receives an `Entity` label plus its type label and is merged by
`entity_id`. An Entity uniqueness constraint and a News ID uniqueness
constraint are created if absent. The `Paper.paper_id` convention remains the
PostgreSQL `paper_id`. Re-run with `--reconcile-papers` after adding verified
aliases, because normal paper sync reads only `graph_synced = FALSE` rows.

Local commands from the repository root:

```bash
python -m pipeline.common.entity_resolver --kind news
python -m pipeline.news.translator.translate --input data/processed/news/news_resolved.json
python -m pipeline.news.loader.postgres --input data/processed/news/news_ready.json --apply
python -m pipeline.common.entity_resolver --kind papers
python -m pipeline.papers.loader.postgres --input data/processed/papers/papers_resolved.json --apply
python -m rag.graph.sync.synchronizer --source both
python -m rag.graph.sync.synchronizer --source both --reconcile-papers
```

`graph-sync.yml` runs after successful ingest workflows or by manual dispatch.
Set the repository variable `DB_RUNNER_LABELS_JSON` to
`["self-hosted","linux","x64","ec2-db"]` once a runner with those labels
is registered and can reach both databases. Until configured, it uses the
existing hosted-runner behavior. Database and Neo4j credentials remain in
GitHub secrets or the ignored local `.env` file.
