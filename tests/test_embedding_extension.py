# -*- encoding: utf-8 -*-
"""Tests for the Nendo framework."""
from nendo import (
    Nendo,
    NendoConfig,
    NendoEmbeddingCreate,
)

import numpy as np
import unittest
import uuid

nd = Nendo(
    config=NendoConfig(
        log_level="WARNING",
        library_plugin="nendo_plugin_library_postgres",
        library_path="tests/library",
        copy_to_library=False,
        max_threads=1,
        plugins=["nendo_plugin_embed_clap"],
        stream_mode=False,
        stream_chunk_size=3,
    ),
)

class EmbeddingExtensionTests(unittest.TestCase):

    # def test_init_with_no_embedding_plugin_auto_detects_clap(self):

    #     self.assertEqual(
    #         nd.library.embedding_plugin.plugin_name,
    #         "nendo_plugin_embed_clap",
    #     )


    def test_run_embedding_plugin_adds_embedding(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        test_embedding = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test",
            embedding=np.zeros(10),
        )
        saved_embedding = nd.library.add_embedding(embedding=test_embedding)
        retrieved_embedding = nd.library.get_embedding(
            embedding_id=saved_embedding.id,
        )
        self.assertTrue((test_embedding.embedding == saved_embedding.embedding).all())
        self.assertTrue((test_embedding.embedding == retrieved_embedding.embedding).all())
        self.assertTrue((saved_embedding.embedding == retrieved_embedding.embedding).all())

    def test_run_add_nan_embedding(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        test_embedding_1 = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test",
            embedding=np.zeros(2),
        )
        saved_embedding_1 = nd.library.add_embedding(embedding=test_embedding_1)
        test_embedding_2 = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test2",
            embedding=np.ones(2),
        )
        nd.library.add_embedding(embedding=test_embedding_2)
        nd.library.nearest_by_vector_with_score(saved_embedding_1.embedding)

    def test_update_embedding_updates_embedding(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        test_embedding = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test",
            embedding=np.zeros(10),
        )
        saved_embedding = nd.library.add_embedding(embedding=test_embedding)
        new_user_id = uuid.uuid4()
        saved_embedding.plugin_name="updated_nendo_plugin_embed_clap"
        saved_embedding.user_id = new_user_id
        saved_embedding.plugin_version="1.1"
        saved_embedding.text="Updated Test"
        saved_embedding.embedding=np.arange(20)
        updated_embedding = nd.library.update_embedding(embedding=saved_embedding)
        retrieved_embedding = nd.library.get_embedding(
            embedding_id=updated_embedding.id,
        )
        self.assertEqual(
            retrieved_embedding.plugin_name,
            "updated_nendo_plugin_embed_clap",
        )
        self.assertEqual(
            retrieved_embedding.plugin_version,
            "1.1",
        )
        self.assertEqual(
            retrieved_embedding.user_id,
            new_user_id,
        )
        self.assertEqual(
            retrieved_embedding.text,
            "Updated Test",
        )
        self.assertEqual(
            len(retrieved_embedding.embedding),
            20,
        )
        self.assertTrue(
            (retrieved_embedding.embedding == np.arange(20)).all(),
        )

    def test_remove_embedding_removes_embedding(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        test_embedding = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test",
            embedding=np.zeros(10),
        )
        saved_embedding = nd.library.add_embedding(embedding=test_embedding)
        result = nd.library.remove_embedding(embedding_id=saved_embedding.id)
        self.assertTrue(result)
        retrieved_embedding = nd.library.get_embedding(embedding_id=saved_embedding.id)
        self.assertEqual(retrieved_embedding, None)

    def test_remove_nonexisten_embedding_returns_false(self):
        nd.library.reset(force=True)
        result = nd.library.remove_embedding(embedding_id=uuid.uuid4())
        self.assertFalse(result)

    def test_nearest_by_vector_with_score(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(
            file_path="tests/assets/test.mp3",
            meta={"test_key": "test_value"},
        )
        test_embedding_1 = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test",
            embedding=np.array([1,1,1]),
        )
        saved_embedding_1 = nd.library.add_embedding(embedding=test_embedding_1)
        test_embedding_2 = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test2",
            embedding=np.array([1,1,0]),
        )
        saved_embedding_2 = nd.library.add_embedding(embedding=test_embedding_2)
        test_embedding_3 = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test3",
            embedding=np.array([1,0,0]),
        )
        saved_embedding_3 = nd.library.add_embedding(embedding=test_embedding_3)
        retrieved_tracks_with_scores = nd.library.nearest_by_vector_with_score(
            vec=test_embedding_1.embedding,
            limit=3,
        )
        self.assertEqual(len(retrieved_tracks_with_scores), 3)
        self.assertEqual(type(retrieved_tracks_with_scores[0]), tuple)
        self.assertEqual(retrieved_tracks_with_scores[0][0].id, track.id)
        self.assertEqual(retrieved_tracks_with_scores[1][0].id, track.id)
        self.assertEqual(retrieved_tracks_with_scores[2][0].id, track.id)
        self.assertEqual(retrieved_tracks_with_scores[0][1], 0.0)
        self.assertEqual(retrieved_tracks_with_scores[1][1], 0.18350341907227385)
        self.assertEqual(retrieved_tracks_with_scores[2][1], 0.42264973081037416)
        retrieved_tracks_with_scores = nd.library.nearest_by_vector_with_score(
            vec=test_embedding_1.embedding,
            limit=3,
            search_meta={"": ["test_value"]},
        )
        self.assertEqual(len(retrieved_tracks_with_scores), 3)
        retrieved_tracks_with_scores = nd.library.nearest_by_vector_with_score(
            vec=test_embedding_1.embedding,
            limit=3,
            search_meta={"": ["wrong_value"]},
        )
        self.assertEqual(len(retrieved_tracks_with_scores), 0)
        retrieved_tracks_with_scores = nd.library.nearest_by_vector_with_score(
            vec=test_embedding_1.embedding,
            limit=3,
            track_type=None,
            filters={},
        )
        self.assertEqual(len(retrieved_tracks_with_scores), 3)

        nearest_by_track = nd.library.nearest_by_track(
            track=track,
            limit=10,
            offset=0,
            filters={},
            search_meta={},
            track_type=None,
            collection_id=None,
        )
        self.assertEqual(len(nearest_by_track), 2)

        nearest_by_track = nd.library.nearest_by_track(
            track=track,
            limit=10,
            offset=0,
            filters={},
            search_meta={"": ["wrong_value"]},
            track_type=None,
            collection_id=None,
        )
        self.assertEqual(len(nearest_by_track), 0)

        nearest_by_track = nd.library.nearest_by_track(
            track=track,
            limit=10,
            offset=0,
            filters={},
            search_meta={},
            track_type="all",
            collection_id=None,
        )
        self.assertEqual(len(nearest_by_track), 0)
    
    def test_count_nearest_by_track(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(
            file_path="tests/assets/test.mp3",
            meta={"test_key": "test_value"},
        )
        test_embedding_1 = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test",
            embedding=np.array([1,1,1]),
        )
        nd.library.add_embedding(embedding=test_embedding_1)
        test_embedding_2 = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test2",
            embedding=np.array([1,1,0]),
        )
        nd.library.add_embedding(embedding=test_embedding_2)
        test_embedding_3 = NendoEmbeddingCreate(
            track_id = track.id,
            user_id=nd.library.user.id,
            plugin_name="nendo_plugin_embed_clap",
            plugin_version="0.1.0",
            text="Test3",
            embedding=np.array([1,0,0]),
        )
        nd.library.add_embedding(embedding=test_embedding_3)
        num_nearest_by_track = nd.library.count_nearest_by_track(
            track=track,
            filters={},
            search_meta={},
            track_type=None,
            collection_id=None,
        )
        self.assertEqual(num_nearest_by_track, 2)


if __name__ == "__main__":
    unittest.main()
