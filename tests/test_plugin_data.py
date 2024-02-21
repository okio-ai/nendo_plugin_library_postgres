# -*- encoding: utf-8 -*-
"""Tests for the Nendo MIR core plugin."""
from nendo import Nendo, NendoConfig

import unittest

nd = Nendo(
    config=NendoConfig(
        log_level="DEBUG",
        library_plugin="PostgresDBLibrary",
        max_threads=1,
        plugins=[],
        stream_mode=False,
        stream_chunk_size=3,
        remote_storage=True,
    )
)


class ExamplePluginTest(unittest.TestCase):
    def test_add_plugin_data(self):
        global nd
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="./assets/test.mp3")
        pd = nd.library.add_plugin_data(
            track_id=track.id,
            plugin_name="test_plugin",
            plugin_version="1.0",
            key="test",
            value="value",
        )
        track = nd.library.get_track(track_id=track.id)
        self.assertEqual(len(track.plugin_data), 1)
        self.assertEqual(track.plugin_data[0], pd)

    def test_filter_random_track(self):
        global nd
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="./assets/test.mp3")
        pd = nd.library.add_plugin_data(
            track_id=track.id,
            plugin_name="test_plugin",
            plugin_version="1.0",
            key="test",
            value="value",
        )
        example_data = nd.library.filter_tracks(
            filters={"test": "value"},
            order_by="random",
            plugin_names=["test_plugin"],
        )[0].get_plugin_data(plugin_name="test_plugin")
        self.assertEqual(len(example_data), 1)
        self.assertEqual(example_data[0], pd)

    def test_filter_by_plugin_data_and_filename(self):
        global nd
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="./assets/test.mp3")
        pd = nd.library.add_plugin_data(
            track_id=track.id,
            plugin_name="test_plugin",
            plugin_version="1.0",
            key="test",
            value="value",
        )
        example_data = nd.library.filter_tracks(
            filters={"test": "value"},
            resource_filters={},
            order_by="random",
            plugin_names=["test_plugin"],
        )[0].get_plugin_data(plugin_name="test_plugin")
        self.assertEqual(len(example_data), 1)
        self.assertEqual(example_data[0], pd)

    def test_filter_tracks_by_plugin_data(self):
        global nd
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="./assets/test.mp3")
        pd = nd.library.add_plugin_data(
            track_id=track.id,
            plugin_name="test_plugin",
            plugin_version="1.0",
            key="foo",
            value="bar",
        )
        track = nd.library.get_track(track_id=track.id)
        example_data = nd.library.filter_tracks(
            filters={"foo": "bar"},
            plugin_names=["test_plugin"],
        )[0].get_plugin_data(plugin_name="test_plugin")
        self.assertEqual(len(example_data), 1)
        self.assertEqual(example_data[0], track.plugin_data[0])
        example_data = nd.library.filter_tracks(
            filters={"foo": ["bar", "baz"]},
            plugin_names=["test_plugin"],
        )[0].get_plugin_data(plugin_name="test_plugin")
        self.assertEqual(len(example_data), 1)
        self.assertEqual(example_data[0], pd)
        example_data = nd.library.filter_tracks(
            filters={"foo": ["bat", "baz"]},
            plugin_names=["test_plugin"],
        )
        self.assertEqual(len(example_data), 0)
        pd2 = nd.library.add_plugin_data(
            track_id=track.id,
            plugin_name="test_plugin",
            plugin_version="1.0",
            key="number",
            value="15.10289371",
        )
        track = nd.library.get_track(track_id=track.id)
        example_data = nd.library.filter_tracks(
            filters={"number": (10.0, 20.0)},
            plugin_names=["test_plugin"],
        )[0].get_plugin_data_by_key(plugin_name="test_plugin", key="number")
        self.assertEqual(len(example_data), 1)
        self.assertEqual(example_data[0], pd2)
        example_data = nd.library.filter_tracks(
            filters={"number": (10, 20)},
            plugin_names=["test_plugin"],
        )[0].get_plugin_data_by_key(plugin_name="test_plugin", key="number")
        self.assertEqual(len(example_data), 1)
        self.assertEqual(example_data[0], pd2)
        example_data = nd.library.filter_tracks(
            filters={"number": (20.0, 30.0)},
            plugin_names=["test_plugin"],
        )
        self.assertEqual(len(example_data), 0)

    def test_filter_tracks_by_multiple_plugin_keys(self):
        global nd
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="./assets/test.mp3")
        pd = nd.library.add_plugin_data(
            track_id=track.id,
            plugin_name="test_plugin",
            plugin_version="1.0",
            key="foo1",
            value="bar1",
        )
        pd2 = nd.library.add_plugin_data(
            track_id=track.id,
            plugin_name="test_plugin",
            plugin_version="1.0",
            key="foo2",
            value="bar2",
        )
        pd3 = nd.library.add_plugin_data(
            track_id=track.id,
            plugin_name="test_plugin",
            plugin_version="1.0",
            key="number",
            value="15.10289371",
        )
        example_data = nd.library.filter_tracks(
            filters={"foo1": ["bar1", "baz1"], "foo2": "bar2", "number": (12.0, 18.0)},
            plugin_names=["test_plugin"],
        )
        self.assertEqual(len(example_data), 1)
        example_data = nd.library.filter_tracks(
            filters={"foo1": ["bar1", "baz1"], "foo2": "bar3", "number": (12.0, 18.0)},
            plugin_names=["test_plugin"],
        )
        self.assertEqual(len(example_data), 0)
        example_data = nd.library.filter_tracks(
            filters={"foo1": ["bar1", "baz1"], "foo2": "bar2", "number": (20.0, 21.0)},
            plugin_names=["test_plugin"],
        )
        self.assertEqual(len(example_data), 0)
        example_data = nd.library.filter_tracks(
            filters={"foo1": ["bat1", "baz1"], "foo2": "bar2", "number": (12.0, 18.0)},
            plugin_names=["test_plugin"],
        )
        self.assertEqual(len(example_data), 0)


if __name__ == "__main__":
    unittest.main()
