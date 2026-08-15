from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROVENANCE = load_module(
    "racer_provenance",
    P2 / "scripts" / "racer_c" / "audit_data_provenance.py",
)
ROLE = load_module(
    "racer_role_feasibility",
    P2 / "scripts" / "racer_c" / "role_feasibility.py",
)


def synthetic_rows(n0: int, n1: int, blank_clusters: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, count in ((0, n0), (1, n1)):
        for index in range(count):
            global_index = len(rows)
            rows.append(
                {
                    "endpoint": "Synthetic",
                    "structure_id": f"s{global_index:05d}",
                    "target": str(label),
                    "murcko_scaffold_id": f"scaffold_{global_index % 140:03d}",
                    "similarity_cluster_id": "" if blank_clusters else f"cluster_{global_index % 70:03d}",
                }
            )
    return rows


class Phase1ProvenanceTests(unittest.TestCase):
    def test_manifest_is_complete_and_fail_closed(self) -> None:
        provenance_path = P2 / "protocols" / "data_provenance_license_manifest.csv"
        candidate_path = P2 / "protocols" / "endpoint_candidate_manifest.csv"
        rows = PROVENANCE.audit_rows(
            PROVENANCE.read_csv(provenance_path),
            PROVENANCE.read_csv(candidate_path),
        )
        self.assertEqual(len(rows), 29)
        by_endpoint = {row["endpoint"]: row for row in rows}
        self.assertIn("original_terms_unresolved", by_endpoint["HIA_Hou"]["blockers"])
        self.assertEqual(by_endpoint["CYP2C9_Veith"]["raw_hash_ready"], "true")
        self.assertEqual(by_endpoint["CYP2C9_Veith"]["freeze1_ready"], "true")
        self.assertEqual(by_endpoint["CYP2C9_Veith"]["license_ready"], "true")
        self.assertEqual(by_endpoint["AMES"]["license_ready"], "false")
        self.assertEqual(by_endpoint["Tox21_NR_ER"]["license_ready"], "true")
        self.assertEqual(by_endpoint["Tox21_NR_ER"]["raw_hash_ready"], "true")
        self.assertEqual(by_endpoint["Tox21_NR_ER"]["freeze1_ready"], "true")

    def test_canonical_hash_ignores_row_order_and_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.csv"
            second = Path(tmp) / "second.csv"
            first.write_bytes(b"a,b\r\n1,x\r\n2,y\r\n")
            second.write_bytes(b"a,b\n2,y\n1,x\n")
            self.assertEqual(
                PROVENANCE.canonical_csv_sha256(first),
                PROVENANCE.canonical_csv_sha256(second),
            )


class Phase1RoleFeasibilityTests(unittest.TestCase):
    fractions = {"dev": 0.50, "policy": 0.10, "conformal": 0.20, "test": 0.20}

    def test_group_assignment_is_deterministic_under_row_permutation(self) -> None:
        rows = synthetic_rows(120, 120)
        forward = ROLE.allocate_groups(rows, "murcko_scaffold_id", self.fractions, 101)
        reverse = ROLE.allocate_groups(list(reversed(rows)), "murcko_scaffold_id", self.fractions, 101)
        self.assertEqual(forward.assignment, reverse.assignment)
        self.assertEqual(forward.role_class_counts, reverse.role_class_counts)

    def test_scaffold_assignment_is_label_blind(self) -> None:
        rows = synthetic_rows(120, 120)
        # Use a non-complement permutation.  A global 0/1 flip preserves the
        # per-group majority count and cannot detect label-dependent ordering.
        labels = [row["target"] for row in rows]
        shifted = labels[37:] + labels[:37]
        permuted = [dict(row, target=label) for row, label in zip(rows, shifted)]
        original = ROLE.allocate_groups(
            rows,
            "murcko_scaffold_id",
            self.fractions,
            101,
            use_labels_for_assignment=False,
        )
        changed_labels = ROLE.allocate_groups(
            permuted,
            "murcko_scaffold_id",
            self.fractions,
            101,
            use_labels_for_assignment=False,
        )
        self.assertEqual(original.assignment, changed_labels.assignment)

    def test_structure_groups_are_indivisible(self) -> None:
        rows = synthetic_rows(40, 40)
        result = ROLE.allocate_groups(rows, "murcko_scaffold_id", self.fractions, 103)
        self.assertEqual(len(result.assignment), 80)
        self.assertEqual(sum(result.role_totals.values()), len(rows))
        self.assertEqual(set(result.assignment.values()).issubset(set(ROLE.ROLES)), True)

    def test_clintox_like_rare_class_fails_primary_count_gate(self) -> None:
        rows = synthetic_rows(1349, 93)
        summary, resolution = ROLE.audit_one(
            rows,
            "random_grouped",
            self.fractions,
            101,
            minimum_retention=0.50,
            alpha=0.10,
        )
        self.assertEqual(summary["primary_count_gate"], "fail")
        self.assertIn("class_1_total_lt_350", summary["failure_reasons"])
        selected_positive = [
            row
            for row in resolution
            if row["population"] == "selected_floor" and row["true_class"] == 1
        ][0]
        self.assertEqual(selected_positive["primary_precision_minimum_35"], "false")

    def test_missing_cluster_ids_fail_closed(self) -> None:
        rows = synthetic_rows(60, 60, blank_clusters=True)
        with self.assertRaisesRegex(ValueError, "blank similarity_cluster_id"):
            ROLE.allocate_groups(rows, "similarity_cluster_id", self.fractions, 101)

    def test_csv_reader_preserves_required_schema(self) -> None:
        rows = synthetic_rows(5, 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_role_input.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(ROLE.validate_role_input(ROLE.read_csv(path))), 10)

    def test_empty_failure_table_overwrites_stale_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.csv"
            path.write_text("stale\nrow\n", encoding="utf-8")
            fields = ["endpoint", "track", "allocation", "seed", "reason"]
            ROLE.write_csv(path, [], fieldnames=fields)
            self.assertEqual(path.read_text(encoding="utf-8").strip(), ",".join(fields))


if __name__ == "__main__":
    unittest.main()
