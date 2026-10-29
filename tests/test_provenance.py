import tempfile
import unittest
from pathlib import Path

from lob_xai.provenance import file_identity, verify_file_identity


class ProvenanceTests(unittest.TestCase):
    def test_identity_detects_file_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.bin"
            p.write_bytes(b"first")
            identity = file_identity(p)
            verify_file_identity(p, identity)
            p.write_bytes(b"second")
            with self.assertRaises(ValueError):
                verify_file_identity(p, identity)


if __name__ == "__main__":
    unittest.main()
