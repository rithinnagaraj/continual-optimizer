"""Regression cases for configuration commits and terminal-output recovery.

Fixtures exercise configuration and journal state without training.
"""
import json
from pathlib import Path
import tempfile
import unittest

from paper_run_integrity import lock_e2_configuration, require_complete_cell


class PaperRunIntegrityTests(unittest.TestCase):
    def config(self):
        return {"scientific_settings": {"spring_step": "proximal", "lr": .001},
                "seeds": [42], "grade": {}}

    def test_e2_configuration_exists_before_first_result_and_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"run_config.json"
            expected=self.config()
            lock_e2_configuration(path, expected)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), expected)
            self.assertEqual(lock_e2_configuration(path, expected, require_existing=True), expected)

    def test_e2_changed_science_and_unidentified_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"run_config.json"
            with self.assertRaises(RuntimeError):
                lock_e2_configuration(path, self.config(), require_existing=True)
            lock_e2_configuration(path, self.config())
            changed=self.config();changed["scientific_settings"]["spring_step"]="explicit"
            with self.assertRaises(RuntimeError):lock_e2_configuration(path, changed)

    def rows(self):
        scalars=[{"arm":"cap","seed":42,"step":s} for s in range(1,5)]
        tasks=[{"arm":"cap","seed":42,"task":t} for t in (1,2)]
        pool=[{"arm":"cap","seed":42,"task":t,"pool_index":d,"is_current":d==t-1}
              for t in (1,2) for d in (0,1)]
        return scalars,tasks,pool

    def validate(self, rows, counts=None):
        require_complete_cell(*rows, arm="cap", seed=42, task_count=2,
            total_steps=4, scalar_every=1, pool_size=2,
            committed_counts=counts or {"scalars":4,"tasks":2,"pool":4})

    def test_complete_committed_cell_is_accepted(self):
        self.validate(self.rows())

    def test_final_task_row_does_not_admit_missing_pool_or_scalars(self):
        rows=self.rows();rows[2].pop()
        with self.assertRaises(RuntimeError):self.validate(rows)
        rows=self.rows();rows[0].pop()
        with self.assertRaises(RuntimeError):self.validate(rows)

    def test_duplicate_rows_and_wrong_committed_counts_are_rejected(self):
        rows=self.rows();rows[2][-1]=dict(rows[2][0])
        with self.assertRaises(RuntimeError):self.validate(rows)
        with self.assertRaises(RuntimeError):
            self.validate(self.rows(), {"scalars":4,"tasks":1,"pool":4})

    def test_each_task_requires_one_current_domain(self):
        rows=self.rows()
        for row in rows[2]:row["is_current"]=False
        with self.assertRaises(RuntimeError):self.validate(rows)
