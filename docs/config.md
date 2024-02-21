# Configuration

To configure the Nendo Library Plugin, use environment variables:

```bash
$ export POSTGRES_HOST=my-postgres-host:5432
$ export POSTGRES_PASSWORD=mysecretpassword
# more configuration variables ...
```

## Configuration Reference

| **Name** | **Env var** | **Type** | **Default** | **Description** |
|---|---|---|---|---|
| storage_location | STORAGE_LOCATION | `str` | `"local"` | The location of the storage. Can be either of `"local"` or `"gcs"`. |
| google_storage_credentials | GOOGLE_STORAGE_CREDENTIALS | `str` | `"{}"` | The google storage credentials to use, if `storage_location` is set to `"gcs"`, as a json string. |
| postgres_host | POSTGRES_HOST | `str` | `"localhost:5432"` | The PostgresDB hostname and port to connect to. |
| postgres_user | POSTGRES_USER | `str` | `"nendo"` | The name of the user with which to connect to the PostgresDB |
| postgres_password | POSTGRES_PASSWORD | `str` | `"nendo"` | The PostgresDB user password. |
| postgres_db | POSTGRES_DB | `str` | `"nendo"` | The name of the Postgres Database in which to store the Nendo Library. |
| embedding_plugin | EMBEDDING_PLUGIN | `str` | `"nendo_plugin_embed_clap"` | The name of the embedding plugin to use for computing embeddings. |
