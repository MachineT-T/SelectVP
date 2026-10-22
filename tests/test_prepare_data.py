"""Regression checks for the annotation validation used by reproduction scripts."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from selectvp.cli import main
from selectvp.utils.io import write_json


ROOT = Path(__file__).resolve().parents[1]


class PrepareDataCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="selectvp-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.annotations = self.directory / "queries.json"
        self.truth = self.directory / "ground_truth.json"
        write_json(self.annotations, [
            {"id": 7, "video": "demo.mp4", "query": "Open the door", "duration": 6},
            {"id": "7", "video": "demo.mp4", "query": "Close the door", "duration": 6},
        ])
        write_json(self.truth, [
            {"id": 7, "relevant_windows": [[1, 3]]},
            {"id": "7", "relevant_windows": [[3, 5]]},
        ])

    def check_annotations(self, dataset="charades_sta", sample_ids=None):
        args = ["prepare-data", "--config", str(ROOT / "configs" / f"{dataset}.yaml"),
                "--check", "--annotations-only",
                "--set", f"data.annotations={self.annotations}",
                "--set", f"data.ground_truth={self.truth}",
                "--set", f"data.video_dir={self.directory / 'missing-videos'}"]
        args += ["--sample-ids", str(sample_ids)] if sample_ids else ["--full"]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_valid_annotations_do_not_require_videos(self):
        for dataset in ("charades_sta", "activitynet", "qvhighlights"):
            with self.subTest(dataset=dataset):
                result, stdout, stderr = self.check_annotations(dataset)
                self.assertEqual(result, 0, stderr)
                self.assertIn("Validated 2 selected queries", stdout)

    def test_missing_ground_truth_file_is_rejected(self):
        self.truth.unlink()
        result, stdout, stderr = self.check_annotations()
        self.assertEqual(result, 1)
        self.assertNotIn("Validated", stdout)
        self.assertIn("FileNotFoundError", stderr)

    def test_missing_ground_truth_id_is_rejected(self):
        write_json(self.truth, [{"id": 7, "relevant_windows": [[1, 3]]}])
        result, _, stderr = self.check_annotations()
        self.assertEqual(result, 1)
        self.assertIn("Missing ground truth ID", stderr)

    def test_duplicate_ground_truth_ids_are_rejected(self):
        write_json(self.truth, [{"id": 7, "relevant_windows": [[1, 3]]}] * 2)
        result, _, stderr = self.check_annotations()
        self.assertEqual(result, 1)
        self.assertIn("Duplicate ID", stderr)

    def test_invalid_ground_truth_windows_are_rejected(self):
        for windows in ([], [[3, 1]], [[1, 1]], [[-1, 2]], [[1, 2, 3]]):
            with self.subTest(windows=windows):
                write_json(self.truth, [{"id": 7, "relevant_windows": windows}])
                result, _, stderr = self.check_annotations()
                self.assertEqual(result, 1)
                self.assertIn("GT window", stderr)

    def test_subset_only_requires_ground_truth_for_selected_ids(self):
        selected = self.directory / "sample_ids.json"
        write_json(selected, ["7"])
        write_json(self.truth, [{"id": "7", "relevant_windows": [[3, 5]]}])
        result, stdout, stderr = self.check_annotations(sample_ids=selected)
        self.assertEqual(result, 0, stderr)
        self.assertIn("Validated 1 selected queries", stdout)


if __name__ == "__main__":
    unittest.main()
