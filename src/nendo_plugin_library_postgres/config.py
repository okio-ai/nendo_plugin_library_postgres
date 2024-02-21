"""Default settings for the Nendo Postgres Library."""
from nendo import NendoConfig, ResourceLocation
from pydantic import Field


class PostgresConfig(NendoConfig):
    storage_location: ResourceLocation = Field(default=ResourceLocation.local)
    google_storage_credentials: str = Field(default=r"{}")
    postgres_host: str = Field(default="localhost:5432")
    postgres_user: str = Field(default="nendo")
    postgres_password: str = Field(default="nendo")
    postgres_db: str = Field(default="nendo")
    embedding_plugin: str = Field(default="nendo_plugin_embed_clap")
