#!/usr/bin/env python3
"""
MediaVault - Secure Data Management for Cell Culture Media
"""

import json
import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np


@dataclass
class MediaRecord:
    """Single media formulation record"""
    id: str
    composition: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "composition": self.composition,
            "metadata": self.metadata
        }


class MediaVault:
    """
    Secure data management for media formulations.

    Features:
    - Multi-format import (CSV, Excel, JSON)
    - Quality assurance
    - Similarity search
    - Differential privacy anonymization
    """

    def __init__(self):
        self.records: Dict[str, MediaRecord] = {}
        self._import_history: List[Dict] = []

    def add_record(self, record: MediaRecord) -> None:
        """Add a single record to vault"""
        self.records[record.id] = record

    def import_csv(self, filepath: str) -> int:
        """Import from CSV file"""
        count = 0
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                comp = {k: float(v) for k, v in row.items()
                       if k not in ['id', 'name', 'notes'] and v.replace('.','',1).isdigit()}
                meta = {k: v for k, v in row.items()
                       if k in ['id', 'name', 'notes']}
                rec = MediaRecord(id=row.get('id', f"rec_{count}"),
                              composition=comp, metadata=meta)
                self.add_record(rec)
                count += 1

        self._import_history.append({"type": "csv", "file": filepath, "count": count})
        return count

    def import_json(self, filepath: str) -> int:
        """Import from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        count = 0
        for item in data:
            rec = MediaRecord(
                id=item.get('id', f"rec_{count}"),
                composition=item.get('composition', {}),
                metadata=item.get('metadata', {})
            )
            self.add_record(rec)
            count += 1

        self._import_history.append({"type": "json", "file": filepath, "count": count})
        return count

    def export_dataframe(self) -> 'np.ndarray':
        """Export to pandas DataFrame format"""
        if not self.records:
            return np.array([])

        # Collect all components
        all_components = set()
        for rec in self.records.values():
            all_components.update(rec.composition.keys())

        # Build matrix
        ids = list(self.records.keys())
        headers = sorted(all_components)
        data = []

        for rid in ids:
            row = [self.records[rid].composition.get(c, 0.0) for c in headers]
            data.append(row)

        return np.array(data)

    def similarity_search(self, query: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar formulations using cosine similarity.

        Args:
            query: Composition to search for
            top_k: Number of results to return

        Returns:
            List of (id, similarity_score) tuples
        """
        if not self.records:
            return []

        # Build vectors
        all_components = set(query.keys())
        for rec in self.records.values():
            all_components.update(rec.composition.keys())

        q_vec = np.array([query.get(c, 0.0) for c in sorted(all_components)])
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scores = []
        for rid, rec in self.records.items():
            r_vec = np.array([rec.composition.get(c, 0.0) for c in sorted(all_components)])
            r_norm = np.linalg.norm(r_vec)
            if r_norm == 0:
                continue

            # Cosine similarity
            sim = np.dot(q_vec, r_vec) / (q_norm * r_norm)
            scores.append((rid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def quality_report(self) -> Dict[str, Any]:
        """Generate quality assurance report"""
        n_records = len(self.records)
        if n_records == 0:
            return {"status": "empty"}

        # Analyze compositions
        all_comps = set()
        for rec in self.records.values():
            all_comps.update(rec.composition.keys())

        return {
            "status": "ok",
            "n_records": n_records,
            "n_components": len(all_comps),
            "components": sorted(all_comps),
            "import_history": self._import_history
        }

    def anonymize(self, epsilon: float = 1.0) -> 'MediaVault':
        """
        Apply differential privacy anonymization.

        Uses Laplace mechanism for numeric perturbations.

        Args:
            epsilon: Privacy budget (smaller = more private)

        Returns:
            New anonymized MediaVault
        """
        from numpy.random import laplace

        scale = 1.0 / epsilon
        new_vault = MediaVault()

        for rid, rec in self.records.items():
            noisy_comp = {}
            for comp, val in rec.composition.items():
                noise = laplace(0, scale)
                noisy_comp[comp] = max(0, val + noise)

            new_rec = MediaRecord(
                id=f"anon_{rid}",
                composition=noisy_comp,
                metadata={"original_id": rid}
            )
            new_vault.add_record(new_rec)

        return new_vault


def load_media_vault(filepath: str) -> MediaVault:
    """Load media vault from file"""
    vault = MediaVault()
    if filepath.endswith('.json'):
        vault.import_json(filepath)
    elif filepath.endswith('.csv'):
        vault.import_csv(filepath)
    return vault


# Demo usage
if __name__ == "__main__":
    # Create sample vault
    vault = MediaVault()

    # Add sample records
    samples = [
        {"id": "M001", "composition": {"glucose": 25.0, "氨基酸": 10.0, "盐": 5.0}},
        {"id": "M002", "composition": {"glucose": 30.0, "氨基酸": 12.0, "盐": 6.0}},
        {"id": "M003", "composition": {"glucose": 20.0, "氨基酸": 8.0, "盐": 4.0}},
    ]

    for s in samples:
        vault.add_record(MediaRecord(**s))

    # Test similarity search
    results = vault.similarity_search({"glucose": 25.0, "氨基酸": 10.0, "盐": 5.0})
    print("Similarity search:", results)

    # Quality report
    print("\nQuality report:", vault.quality_report())