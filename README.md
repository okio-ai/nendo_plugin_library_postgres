# Nendo PostgresDB Library Plugin

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="Nendo Core">
</p>
<br>

<p align="left">
<a href="https://okio.ai" target="_blank">
    <img src="https://img.shields.io/website/https/okio.ai" alt="Website">
</a>
<a href="https://twitter.com/okio_ai" target="_blank">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai" alt="Twitter">
</a>
<a href="https://discord.gg/gaZMZKzScj" target="_blank">
    <img src="https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat" alt="Discord">
</a>
</p>

---

PostgresDB implementation of the Nendo Library Plugin.

## Features


- Supports use of a local or remote PostgresDB instance for storing Nendo's library data.
- Comes with the Nendo Library vector extension, providing support for the storing, querying and retrieval of Embeddings related to Nendo tracks.
- Provides functions for performing a neighborhood search of similar NendoTracks based on various distance metrics.
- Provides a set of special filtering functions for more fine-grained search and retrieval of items from the Nendo Library.

## Requirements

### PostgreSQL

Using this plugin requires you to install the postgres client libraries on the local system. Depending on your OS, you can either install just the client libraries or the entire Postgres package, depending on whether you intend to connect to a remote or a local instance of PostgresDB:

- **Ubuntu**:
    - `sudo apt-get install libpq-dev` (client libraries)
    - `sudo apt-get install postgresql` (complete postgresDB)
- **Mac OS**:
    - `brew install libpq` (client libraries)
    - `brew install postgresql@14` (complete postgresDB)

### pgvector extension

Furthermore, the plugin requires the `pgvector` extension to be installed. Please follow it's official [installation instructions](https://github.com/pgvector/pgvector#installation) to install it before proceeding further.

After you've installed the `pgvector` package, you must also enable the extension for the `nendo` database. Connect to your PostgreSQL instance **with an account that has superuser priviledges** and execute:

```sql
USE nendo;
CREATE EXTENSION IF NOT EXISTS vector;
```

### Embedding plugin

To use the vector features of the Nendo Library, you also need to have at least one `NendoEmbeddingPlugin` installed in your environment and configured to be used by Nendo. For example, to use the `nendo_plugin_embed_clap`, [install it](https://github.com/okio-ai/nendo/nendo_plugin_embed_clap#requirements) and then configure Nendo to also load it on startup. You can use an environment variable for this:

```bash
export PLUGINS='["nendo_plugin_embed_clap"]'
```

The PostgresDB Nendo Library plugin will then pick up the embedding plugin automatically.

## Installation

1. Make sure to meet the [requirements](#requirements).
1. `pip install nendo-plugin-library-postgres`

## How to use

To enable the postgres library plugin, Nendo must be started with the `library_plugin` configuration variable set to `nendo_plugin_library_postgres`:

```python
from nendo import Nendo,NendoConfig
nd = Nendo(config=NendoConfig(library_plugin="nendo_plugin_library_postgres",plugins=["nendo_plugin_embed_clap"]))
```

Alternatively, you can use environment variables to configure Nendo and enable the postgres library plugin. Refer to the [configuration](#plugin-configuration) section for more information.

### Plugin configuration

Please use environment variables to configure the plugin:

```bash
export LIBRARY_PLUGIN="nendo_plugin_library_postgres"
export PLUGINS='["nendo_plugin_embed_clap"]'
export POSTGRES_HOST="localhost:5432"
export POSTGRES_USER="nendo"
export POSTGRES_PASSWORD="nendo"
export POSTGRES_DB="nendo"
```

Then, the configuration will be applied once you start Nendo:

```python
from nendo import Nendo

nd = Nendo()

# print library info
print(nd.library)
```
