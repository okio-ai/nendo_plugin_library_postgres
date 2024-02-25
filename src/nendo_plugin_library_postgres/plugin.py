# -*- encoding: utf-8 -*-
"""Nendo Postgresql library plugin."""

import logging
import uuid
from importlib import metadata
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from sqlalchemy import Engine, and_, asc, create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session
from sqlalchemy.orm.exc import NoResultFound

from nendo import (
    DistanceMetric,
    NendoConfig,
    NendoEmbedding,
    NendoEmbeddingBase,
    NendoEmbeddingCreate,
    NendoEmbeddingPlugin,
    NendoLibraryVectorExtension,
    NendoStorage,
    NendoStorageLocalFS,
    NendoTrack,
    NendoUser,
    ResourceLocation,
    SqlAlchemyNendoLibrary,
)
from nendo.library import model
from nendo.utils import ensure_uuid

from .config import PostgresConfig
from .model import Base, NendoEmbeddingDB
from .storage import NendoStorageGCS  # NendoStorageGCSTranscode

plugin_package = metadata.metadata(__package__ or __name__)
plugin_config = PostgresConfig()
# Base = declarative_base(metadata=MetaData())
logger = logging.getLogger("nendo")


class PostgresDBLibrary(SqlAlchemyNendoLibrary, NendoLibraryVectorExtension):
    config: NendoConfig = None
    plugin_config: PostgresConfig = None
    user: NendoUser = None
    db: Engine = None
    embedding_plugin: Optional[NendoEmbeddingPlugin] = None
    storage_driver: NendoStorage = None

    def __init__(
            self,
            db: Optional[Engine] = None,
            config: Optional[NendoConfig] = None,
            user_id: Optional[uuid.UUID] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs, embedding_plugin=plugin_config.embedding_plugin)
        self.config = config
        self.plugin_config = plugin_config
        if self.plugin_config.storage_location == ResourceLocation.gcs:
            self.logger.info("Using GCS storage backend.")
            self.storage_driver = NendoStorageGCS(  # NendoStorageGCSTranscode(
                environment=self.config.environment,
                credentials_json=(self.plugin_config.google_storage_credentials),
            )
        else:
            if self.plugin_config.storage_location != ResourceLocation.local:
                self.logger.warning(
                    "Configured storage location"
                    f"{self.config.storage_location} not supported"
                    "by this plugin. Falling back to local storage.",
                )
            self.storage_driver = NendoStorageLocalFS(
                library_path=self.config.library_path,
                user_id=user_id or self.config.user_id,
            )
        self._connect(db)
        self.storage_driver.init_storage_for_user(user_id=str(self.user.id))

    def _connect(
            self,
            db: Optional[Engine] = None,
            # user_id: Optional[uuid.UUID] = None,
    ):
        """Open Postgres session."""
        engine_string = (
            "postgresql://"
            f"{self.plugin_config.postgres_user}:"
            f"{self.plugin_config.postgres_password}@"
            f"{self.plugin_config.postgres_host}/"
            f"{self.plugin_config.postgres_db}"
        )
        self.db = db or create_engine(engine_string)
        try:
            Base.metadata.create_all(bind=self.db)
        except IntegrityError as e:
            logger.error(f"Failed to initialize database: {e}")            
        # if user_id is not None:
        #     with self.session_scope() as session:
        #         db_user = (
        #             session.query(model.NendoUserDB).filter_by(id=user_id).one_or_none()
        #         )
        #         if db_user is not None:a
        #             self.user = db_user
        #         else:
        #             raise NendoUserNotFoundError(target_id=user_id)
        # else:
        #     self.user = self.default_user
        self.user = self.default_user

    def _pg_distance(self, distance_metric: DistanceMetric) -> Any:
        if distance_metric == DistanceMetric.euclidean:
            return NendoEmbeddingDB.embedding.l2_distance
        if distance_metric == DistanceMetric.cosine:
            return NendoEmbeddingDB.embedding.cosine_distance
        if distance_metric == DistanceMetric.max_inner_product:
            return NendoEmbeddingDB.embedding.max_inner_product
        raise ValueError(
            f"Got unexpected value for distance: {distance_metric}. "
            f"Should be one of {', '.join([ds.value for ds in DistanceMetric])}.",
        )

    def _get_meta_filter_query(
        self,
        query: Query,
        search_meta: Optional[Dict[str, List[str]]] = None,
    ) -> Query:
        # apply meta data filter
        query_local = query
        if search_meta and len(search_meta) > 0:
            conditions = []
            for search_key, search_values in search_meta.items():
                # if specific key is given
                if len(search_key) > 0:
                    for value in search_values:
                        conditions.append(
                            model.NendoTrackDB.meta[search_key].astext.ilike(f"%{value}%"),
                        )
                # if key is empty string, search over all values
                else:
                    for value in search_values:
                        json_value_filter = text("""
                            EXISTS (
                                SELECT 1
                                FROM json_each_text(tracks.meta) AS meta
                                WHERE meta.value ILIKE :value_like
                            )
                        """).bindparams(value_like=f"%{value}%")
                        conditions.append(json_value_filter)
            combined_condition = and_(*conditions)
            query_local = query_local.filter(combined_condition)
        return query_local

    @property
    def distance_metric(self) -> Any:  # noqa: D102
        return self._pg_distance(self._default_distance)

    def reset(
            self,
            force: bool = False,
            user_id: Optional[Union[str, uuid.UUID]] = None,
    ) -> None:
        user_id = self._ensure_user_uuid(user_id)
        should_proceed = (
                force
                or input(
            "Are you sure you want to reset the library? "
            "This will purge ALL tracks, collections and relationships!"
            "Enter 'y' to confirm: ",
        ).lower()
                == "y"
        )

        if not should_proceed:
            logger.info("Reset operation cancelled.")
            return

        logger.info("Resetting nendo library.")
        with self.session_scope() as session:
            # delete all relationships
            session.query(model.TrackTrackRelationshipDB).delete()
            session.query(model.TrackCollectionRelationshipDB).delete()
            session.query(model.CollectionCollectionRelationshipDB).delete()
            # delete all plugin data
            session.query(model.NendoPluginDataDB).delete()
            # delete all embeddings
            session.query(NendoEmbeddingDB).delete()
            session.commit()
            # delete all collections
            session.query(model.NendoCollectionDB).delete()
            # delete all tracks
            session.query(model.NendoTrackDB).delete()
        # remove files
        for library_file in self.storage_driver.list_files(user_id=str(user_id)):
            self.storage_driver.remove_file(file_name=library_file, user_id=str(user_id))

    def remove_track(
            self,
            track_id: Union[str, uuid.UUID],
            remove_relationships: bool = False,
            remove_plugin_data: bool = True,
            remove_resources: bool = True,
            remove_embeddings: bool = True,
            user_id: Optional[Union[str, uuid.UUID]] = None,
    ) -> bool:
        embeddings = self.get_embeddings(track_id=track_id)
        if len(embeddings) > 0:
            if remove_embeddings:
                with self.session_scope() as session:
                    logger.info("Removing %d embeddings", len(embeddings))
                    session.query(NendoEmbeddingDB).filter(
                        NendoEmbeddingDB.track_id == track_id,
                    ).delete()
                    session.commit()
            else:
                logger.warning(
                    "Cannot remove due to %d existing "
                    "embedding entries. Set `remove_embeddings=True` "
                    "to remove them.",
                    len(embeddings),
                )
                return False
        return super().remove_track(
            track_id=track_id,
            remove_relationships=remove_relationships,
            remove_plugin_data=remove_plugin_data,
            remove_resources=remove_resources,
            user_id=user_id,
        )

    def filter_tracks_by_meta(
        self,
        filters: Optional[Dict[str, Any]] = None,
        search_meta: Optional[Dict[str, List[str]]] = None,
        track_type: Optional[Union[str, List[str]]] = None,
        user_id: Optional[Union[str, uuid.UUID]] = None,
        collection_id: Optional[Union[str, uuid.UUID]] = None,
        plugin_names: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order: str = "asc",
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        session: Optional[Session] = None,
    ) -> Union[List, Iterator]:
        """Obtain tracks from the db by filtering over plugin data and meta data.

        Args:
            filters (Optional[dict]): Dictionary containing the filters to apply.
                Defaults to None.
            search_meta (dict, optional): Dictionary containing separate track.meta filters
                which will be applied in conjunction. The keys of the dictionary should
                correspond to potential field names in track.meta and the values should
                contain the string values which should be contained in the respective
                `track.meta` field's value.
            track_type (Union[str, List[str]], optional): Track type to filter for.
                Can be a singular type or a list of types. Defaults to None.
            user_id (Union[str, UUID], optional): The user ID to filter for.
            collection_id (Union[str, uuid.UUID], optional): Collection id to
                which the filtered tracks must have a relationship. Defaults to None.
            plugin_names (list, optional): List used for applying the filter only to
                data of certain plugins. If None, all plugin data related to the track
                is used for filtering.
            order_by (str, optional): Key used for ordering the results.
            order (str, optional): Ordering ("asc" vs "desc"). Defaults to "asc".
            limit (int, optional): Limit the number of returned results.
            offset (int, optional): Offset into the paginated results (requires limit).

        Returns:
            Union[List, Iterator]: List or generator of tracks, depending on the
                configuration variable stream_mode
        """
        user_id = self._ensure_user_uuid(user_id)
        s = session or self.session_scope()
        with s as session_local:
            """Obtain tracks from the db by filtering w.r.t. various fields."""
            query = self._get_filtered_tracks_query(
                session=session_local,
                filters=filters,
                search_meta=[],
                track_type=track_type,
                user_id=user_id,
                collection_id=collection_id,
                plugin_names=plugin_names,
            )
            query = self._get_meta_filter_query(
                query=query,
                search_meta=search_meta,
            )
            return self.get_tracks(
                query=query,
                order_by=order_by,
                order=order,
                limit=limit,
                offset=offset,
                load_related_tracks=False,
                session=session_local,
            )

    def filter_related_tracks_by_meta(
        self,
        track_id: Union[str, uuid.UUID],
        direction: str = "to",
        filters: Optional[Dict[str, Any]] = None,
        search_meta: Optional[Dict[str, List[str]]] = None,
        track_type: Optional[Union[str, List[str]]] = None,
        user_id: Optional[Union[str, uuid.UUID]] = None,
        collection_id: Optional[Union[str, uuid.UUID]] = None,
        plugin_names: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = "asc",
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Union[List, Iterator]:
        """Get tracks with a relationship to a track and filter the results.

        Args:
            track_id (Union[str, UUID]): ID of the track to be searched for.
            direction (str, optional): The relationship direction ("to", "from", "both").
            filters (Optional[dict]): Dictionary containing the filters to apply.
                Defaults to None.
            search_meta (dict): Dictionary containing the keywords to search for
                over the track.resource.meta field. The dictionary's values
                should contain singular search tokens and the keys currently have no
                effect but might in the future. Defaults to {}.
            track_type (Union[str, List[str]], optional): Track type to filter for.
                Can be a singular type or a list of types. Defaults to None.
            user_id (Union[str, UUID], optional): The user ID to filter for.
            collection_id (Union[str, uuid.UUID], optional): Collection id to
                which the filtered tracks must have a relationship. Defaults to None.
            plugin_names (list, optional): List used for applying the filter only to
                data of certain plugins. If None, all plugin data related to the track
                is used for filtering.
            order_by (str, optional): Key used for ordering the results.
            order (str, optional): Order in which to retrieve results ("asc" or "desc").
            limit (int, optional): Limit the number of returned results.
            offset (int, optional): Offset into the paginated results (requires limit).

        Returns:
            Union[List, Iterator]: List or generator of tracks, depending on the
                configuration variable stream_mode
        """
        user_id = self._ensure_user_uuid(user_id)
        with self.session_scope() as session:
            query = self._get_related_tracks_query(
                track_id=ensure_uuid(track_id),
                session=session,
                user_id=user_id,
                direction=direction,
            )
            query = self._get_filtered_tracks_query(
                session=session,
                query=query,
                filters=filters,
                search_meta=[],
                track_type=track_type,
                user_id=user_id,
                collection_id=collection_id,
                plugin_names=plugin_names,
            )
            query = self._get_meta_filter_query(
                query=query,
                search_meta=search_meta,
            )
            return self.get_tracks(
                query=query,
                order_by=order_by,
                order=order,
                limit=limit,
                offset=offset,
                load_related_tracks=True,
                session=session,
            )

    def add_embedding(
            self,
            embedding: NendoEmbeddingBase,
    ) -> NendoEmbedding:
        embedding_create = NendoEmbeddingCreate(**embedding.dict())
        # cast to float32 for compatibility with pgvector
        embedding_create.embedding = embedding_create.embedding.astype(np.float32)
        with self.session_scope() as session:
            embedding_dict = embedding_create.model_dump()
            embedding_db = NendoEmbeddingDB(**embedding_dict)
            session.add(embedding_db)
            session.commit()
            return NendoEmbedding.model_validate(embedding_db)

    def get_embedding(self, embedding_id: uuid.UUID) -> Optional[NendoEmbedding]:
        with self.session_scope() as session:
            embedding_db = (
                session.query(NendoEmbeddingDB)
                .filter(NendoEmbeddingDB.id == embedding_id)
                .first()
            )
            return (
                NendoEmbedding.model_validate(embedding_db)
                if embedding_db is not None
                else None
            )

    def get_embeddings(
            self,
            track_id: Optional[uuid.UUID] = None,
            plugin_name: Optional[str] = None,
            plugin_version: Optional[str] = None,
    ) -> List[NendoEmbedding]:
        with self.session_scope() as session:
            query = session.query(NendoEmbeddingDB)
            if track_id is not None:
                query = query.filter(
                    NendoEmbeddingDB.track_id == track_id,
                )
            if plugin_name is not None:
                query = query.filter(
                    NendoEmbeddingDB.plugin_name == plugin_name,
                )
            if plugin_version is not None:
                query = query.filter(
                    NendoEmbeddingDB.plugin_version == plugin_version,
                )
            return [
                NendoEmbedding.model_validate(embedding_db) for embedding_db in query
            ]

    def update_embedding(
            self,
            embedding: NendoEmbedding,
    ) -> NendoEmbedding:
        with self.session_scope() as session:
            embedding_db = (
                session.query(
                    NendoEmbeddingDB,
                )
                .filter_by(
                    id=embedding.id,
                )
                .one_or_none()
            )
            if embedding_db is None:
                raise NoResultFound(f"No embedding found with id {embedding.id}")
            embedding_db.user_id = embedding.user_id
            embedding_db.plugin_name = embedding.plugin_name
            embedding_db.plugin_version = embedding.plugin_version
            embedding_db.text = embedding.text
            embedding_db.embedding = embedding.embedding.astype(np.float32)
            session.commit()
            return NendoEmbedding.model_validate(embedding_db)

    def remove_embedding(self, embedding_id: uuid.UUID) -> bool:
        with self.session_scope() as session:
            embedding_db = (
                session.query(NendoEmbeddingDB).filter_by(id=embedding_id).one_or_none()
            )
            if embedding_db is None:
                self.logger.warning("Embedding with id %s not found", embedding_id)
                return False
            session.delete(embedding_db)
            return True

    def _get_nearest_query(
        self,
        session: Session,
        vec: npt.ArrayLike,
        user_id: Optional[uuid.UUID] = None,
        embedding_name: Optional[str] = None,
        embedding_version: Optional[str] = None,
        distance_metric: Optional[DistanceMetric] = None,
        ) -> Query:
        user_id = user_id or self.user.id
        plugin_name = (
            embedding_name if embedding_name is not None else
            self.embedding_plugin.plugin_name
        )
        plugin_version = (
            embedding_version if embedding_version is not None else
            self.embedding_plugin.plugin_version
        )
        distance = (
            self._pg_distance(distance_metric)
            if distance_metric is not None
            else self.distance_metric
        )
        return (
            session.query(
                NendoEmbeddingDB,
                distance(vec).label("distance"),
            )
            .filter(
                NendoEmbeddingDB.user_id == user_id,
                NendoEmbeddingDB.plugin_name == plugin_name,
                NendoEmbeddingDB.plugin_version == plugin_version,
            )
            .join(
                model.NendoTrackDB,
                NendoEmbeddingDB.track_id == model.NendoTrackDB.id,
            )
        )

    def nearest_by_vector_with_score(
        self,
        vec: npt.ArrayLike,
        limit: int = 10,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        search_meta: Optional[Dict[str, List[str]]] = None,
        track_type: Optional[Union[str, List[str]]] = None,
        user_id: Optional[Union[str, uuid.UUID]] = None,
        collection_id: Optional[Union[str, uuid.UUID]] = None,
        plugin_names: Optional[List[str]] = None,
        embedding_name: Optional[str] = None,
        embedding_version: Optional[str] = None,
        distance_metric: Optional[DistanceMetric] = None,
    ) -> List[Tuple[NendoTrack, float]]:
        user_id = self._ensure_user_uuid(user_id)
        # cast to float32 for compatibility with pgvector
        vec = vec.astype(np.float32)
        with self.session_scope() as session:
            query = self._get_nearest_query(
                session=session,
                vec=vec,
                user_id=user_id,
                embedding_name=embedding_name,
                embedding_version=embedding_version,
                distance_metric=distance_metric,
            )
            query = self._get_filtered_tracks_query(
                session=session,
                query=query,
                filters=filters,
                search_meta=[],
                track_type=track_type,
                user_id=user_id,
                collection_id=collection_id,
                plugin_names=plugin_names,
            )
            query = self._get_meta_filter_query(
                query=query,
                search_meta=search_meta,
            )
            query = query.order_by(asc("distance")).limit(limit)
            if offset:
                query = query.offset(offset)
            query = query.all()
            # Construct list of tuples (track, score)
            return [
                (NendoTrack.model_validate(embedding.track), distance)
                for embedding, distance in query
            ]
