"""Test BlockAccessed event serialization compatibility between SGLang (Python) and Dynamo (Rust)."""
import importlib
import importlib.util
import os
import sys
import types
import unittest

# Set up minimal sglang module path without triggering sglang/__init__.py
_sglang_root = "/home/ubuntu/sglang/python"


def _import_kv_events():
    """Import kv_events directly without triggering sglang's heavy __init__.py."""
    # Ensure sglang.srt.disaggregation package structure exists in sys.modules
    for pkg in ["sglang", "sglang.srt", "sglang.srt.disaggregation"]:
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [os.path.join(_sglang_root, *pkg.split("."))]
            mod.__package__ = pkg
            sys.modules[pkg] = mod

    spec = importlib.util.spec_from_file_location(
        "sglang.srt.disaggregation.kv_events",
        os.path.join(_sglang_root, "sglang/srt/disaggregation/kv_events.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_kv_events = _import_kv_events()
BlockAccessed = _kv_events.BlockAccessed
BlockStored = _kv_events.BlockStored
BlockRemoved = _kv_events.BlockRemoved
KVEventBatch = _kv_events.KVEventBatch


class TestBlockAccessedSerialization(unittest.TestCase):
    def test_event_creation(self):
        """Test BlockAccessed can be created with correct fields."""

        event = BlockAccessed(
            block_hashes=[123456789, -987654321, 0],
            request_id="req-abc-123",
            num_cached=2,
            num_prefilled=1,
            cached_mask=[True, True, False],
            medium_per_block=["GPU", "CPU_PINNED", None],
        )

        self.assertEqual(event.block_hashes, [123456789, -987654321, 0])
        self.assertEqual(event.request_id, "req-abc-123")
        self.assertEqual(event.num_cached, 2)
        self.assertEqual(event.num_prefilled, 1)
        self.assertEqual(event.cached_mask, [True, True, False])

    def test_roundtrip_msgpack(self):
        """Test BlockAccessed survives msgpack encode/decode roundtrip."""
        import msgspec
        event = BlockAccessed(
            block_hashes=[100, 200, 300],
            request_id="req-roundtrip",
            num_cached=2,
            num_prefilled=1,
            cached_mask=[True, True, False],
            medium_per_block=["GPU", "GPU", None],
        )

        batch = KVEventBatch(ts=1234567890.0, events=[event])

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

        payload = encoder.encode(batch)
        decoded = decoder.decode(payload)

        self.assertEqual(len(decoded.events), 1)
        decoded_event = decoded.events[0]
        self.assertIsInstance(decoded_event, BlockAccessed)
        self.assertEqual(decoded_event.block_hashes, [100, 200, 300])
        self.assertEqual(decoded_event.request_id, "req-roundtrip")
        self.assertEqual(decoded_event.num_cached, 2)
        self.assertEqual(decoded_event.num_prefilled, 1)
        self.assertEqual(decoded_event.cached_mask, [True, True, False])
        self.assertEqual(decoded_event.medium_per_block, ["GPU", "GPU", None])

    def test_mixed_event_batch(self):
        """Test BlockAccessed works alongside BlockStored and BlockRemoved in a batch."""
        import msgspec
        events = [
            BlockStored(
                block_hashes=[111],
                parent_block_hash=None,
                token_ids=[1, 2, 3],
                block_size=3,
                lora_id=None,
                medium="GPU",
            ),
            BlockAccessed(
                block_hashes=[111, 222],
                request_id="req-mixed",
                num_cached=1,
                num_prefilled=1,
                cached_mask=[True, False],
                medium_per_block=["GPU", None],
            ),
            BlockRemoved(
                block_hashes=[111],
                medium="GPU",
            ),
        ]

        batch = KVEventBatch(ts=99.0, events=events)

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

        payload = encoder.encode(batch)
        decoded = decoder.decode(payload)

        self.assertEqual(len(decoded.events), 3)
        self.assertIsInstance(decoded.events[0], BlockStored)
        self.assertIsInstance(decoded.events[1], BlockAccessed)
        self.assertIsInstance(decoded.events[2], BlockRemoved)

    def test_all_cached(self):
        """Test BlockAccessed with all blocks cached."""
        event = BlockAccessed(
            block_hashes=[1, 2, 3, 4],
            request_id="req-all-cached",
            num_cached=4,
            num_prefilled=0,
            cached_mask=[True, True, True, True],
            medium_per_block=["GPU", "GPU", "GPU", "GPU"],
        )
        self.assertEqual(event.num_cached, 4)
        self.assertEqual(event.num_prefilled, 0)
        self.assertTrue(all(event.cached_mask))

    def test_none_cached(self):
        """Test BlockAccessed with no blocks cached (cold cache)."""
        event = BlockAccessed(
            block_hashes=[1, 2, 3],
            request_id="req-cold",
            num_cached=0,
            num_prefilled=3,
            cached_mask=[False, False, False],
            medium_per_block=[None, None, None],
        )
        self.assertEqual(event.num_cached, 0)
        self.assertEqual(event.num_prefilled, 3)
        self.assertFalse(any(event.cached_mask))


    def test_large_block_hashes(self):
        """Test BlockAccessed with large hash values (i64 range)."""
        import msgspec
        from sglang.srt.disaggregation.kv_events import BlockAccessed, KVEventBatch

        max_i64 = 2**63 - 1
        min_i64 = -(2**63)

        event = BlockAccessed(
            block_hashes=[max_i64, min_i64, 0, -1],
            request_id="req-large-hashes",
            num_cached=2,
            num_prefilled=2,
            cached_mask=[True, True, False, False],
            medium_per_block=["GPU", "CPU_PINNED", None, None],
        )

        batch = KVEventBatch(ts=0.0, events=[event])
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

        payload = encoder.encode(batch)
        decoded = decoder.decode(payload)

        decoded_event = decoded.events[0]
        self.assertEqual(decoded_event.block_hashes, [max_i64, min_i64, 0, -1])

    def test_msgpack_array_format_tag(self):
        """Verify the msgpack format includes the class tag for Rust deserialization."""
        import msgspec
        event = BlockAccessed(
            block_hashes=[42],
            request_id="req-tag-test",
            num_cached=1,
            num_prefilled=0,
            cached_mask=[True],
            medium_per_block=["GPU"],
        )

        batch = KVEventBatch(ts=1.0, events=[event])
        encoder = msgspec.msgpack.Encoder()
        payload = encoder.encode(batch)

        # Decode as raw to inspect structure
        raw_decoder = msgspec.msgpack.Decoder()
        raw = raw_decoder.decode(payload)

        # raw should be [ts, [event1, ...]]
        self.assertIsInstance(raw, (list, tuple))
        # The event should be tagged: ["BlockAccessed", ...fields...]
        event_raw = raw[1][0]
        self.assertIsInstance(event_raw, (list, tuple))
        self.assertEqual(event_raw[0], "BlockAccessed")


if __name__ == "__main__":
    unittest.main()
