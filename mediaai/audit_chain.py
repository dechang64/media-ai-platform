#!/usr/bin/env python3
"""
AuditChain - Research Documentation and Provenance Tracking
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ExperimentRecord:
    """Single experiment record"""
    timestamp: str
    experiment_id: str
    data_hash: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    previous_hash: str


@dataclass
class AuditEntry:
    """Blockchain-style audit entry"""
    index: int
    timestamp: str
    experiment_id: str
    data_hash: str
    previous_hash: str
    signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditChain:
    """
    Research documentation with blockchain-based provenance.

    Features:
    - SHA-256 hash chaining
    - Tamper detection
    - Full experiment traceability
    - Export to JSON/CSV
    """

    def __init__(self, chain_id: str = "mediaai"):
        self.chain_id = chain_id
        self.chain: List[AuditEntry] = []
        self.experiments: Dict[str, ExperimentRecord] = {}

    def _compute_hash(self, data: str) -> str:
        """Compute SHA-256 hash"""
        return hashlib.sha256(data.encode()).hexdigest()

    def record_experiment(self, experiment_id: str,
                       parameters: Dict[str, Any],
                       results: Dict[str, Any]) -> str:
        """
        Record experiment result.

        Returns:
            Transaction hash
        """
        now = datetime.now().isoformat()
        data_str = json.dumps({
            "experiment_id": experiment_id,
            "parameters": parameters,
            "results": results,
            "timestamp": now
        }, sort_keys=True)

        data_hash = self._compute_hash(data_str)

        # Previous hash
        prev_hash = "genesis"
        if self.chain:
            prev_hash = self.chain[-1].data_hash

        # Create entry
        entry = AuditEntry(
            index=len(self.chain),
            timestamp=now,
            experiment_id=experiment_id,
            data_hash=data_hash,
            previous_hash=prev_hash,
            signature=self._sign(data_hash)
        )

        self.chain.append(entry)

        # Store experiment record
        self.experiments[experiment_id] = ExperimentRecord(
            timestamp=now,
            experiment_id=experiment_id,
            data_hash=data_hash,
            parameters=parameters,
            results=results,
            previous_hash=prev_hash
        )

        return data_hash

    def _sign(self, data_hash: str) -> str:
        """Create digital signature (simplified)"""
        return hashlib.sha256(data_hash.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """
        Verify chain integrity.

        Returns:
            True if no tampering detected
        """
        if not self.chain:
            return True

        # Check genesis
        if self.chain[0].previous_hash != "genesis":
            return False

        # Check each link
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i-1]

            # Verify previous hash link
            if curr.previous_hash != prev.data_hash:
                return False

            # Verify chain index
            if curr.index != i:
                return False

        return True

    def get_provenance(self, experiment_id: str) -> List[AuditEntry]:
        """
        Trace experiment provenance.

        Args:
            experiment_id: Target experiment

        Returns:
            Ordered list of audit entries
        """
        if experiment_id not in self.experiments:
            return []

        provenance = []
        exp = self.experiments[experiment_id]

        # Find forward from start
        current_hash = exp.data_hash
        for entry in self.chain:
            if entry.data_hash == current_hash:
                provenance.append(entry)
                current_hash = entry.previous_hash

        return list(reversed(provenance))

    def export_json(self, filepath: str) -> None:
        """Export chain to JSON"""
        data = {
            "chain_id": self.chain_id,
            "verified": self.verify_integrity(),
            "entries": [
                {
                    "index": e.index,
                    "timestamp": e.timestamp,
                    "experiment_id": e.experiment_id,
                    "data_hash": e.data_hash,
                    "previous_hash": e.previous_hash,
                    "signature": e.signature,
                    "metadata": e.metadata
                }
                for e in self.chain
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: str) -> None:
        """Export experiments to CSV"""
        lines = ["index,timestamp,experiment_id,data_hash,previous_hash"]

        for e in self.chain:
            lines.append(
                f"{e.index},{e.timestamp},{e.experiment_id},"
                f"{e.data_hash},{e.previous_hash}"
            )

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))

    def generate_report(self) -> Dict[str, Any]:
        """Generate audit report"""
        return {
            "chain_id": self.chain_id,
            "length": len(self.chain),
            "verified": self.verify_integrity(),
            "experiments": list(self.experiments.keys()),
            "created_at": self.chain[0].timestamp if self.chain else None
        }


class ExperimentTracker:
    """High-level experiment tracking interface"""

    def __init__(self):
        self.audit_chain = AuditChain()
        self._current_context: Dict[str, Any] = {}

    def start_experiment(self, name: str,
                      parameters: Dict[str, Any]) -> str:
        """Start tracking new experiment"""
        import uuid
        exp_id = f"exp_{uuid.uuid4().hex[:8]}"

        self._current_context = {
            "id": exp_id,
            "name": name,
            "parameters": parameters,
            "start_time": time.time()
        }

        return exp_id

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log experiment metrics"""
        if not self._current_context:
            return

        self._current_context["metrics"] = metrics

    def end_experiment(self, results: Dict[str, Any]) -> str:
        """End and record experiment"""
        if not self._current_context:
            raise ValueError("No active experiment")

        exp_id = self._current_context["id"]
        parameters = self._current_context.get("parameters", {})
        parameters.update({
            "duration": time.time() - self._current_context.get("start_time", 0)
        })

        tx_hash = self.audit_chain.record_experiment(
            exp_id, parameters, results
        )

        self._current_context = {}
        return tx_hash

    def get_history(self) -> List[Dict]:
        """Get experiment history"""
        return [
            {
                "experiment_id": exp.experiment_id,
                "timestamp": exp.timestamp,
                "parameters": exp.parameters,
                "results": exp.results
            }
            for exp in self.experiments.values()
        ]


# Demo usage
if __name__ == "__main__":
    # Create tracker
    tracker = ExperimentTracker()

    # Run experiments
    for i in range(3):
        exp_id = tracker.start_experiment(
            f"media_opt_{i}",
            {"iteration": i, "lr": 0.01}
        )

        tracker.log_metrics({
            "viability": 0.7 + i * 0.1,
            "cost": 100 - i * 10
        })

        tx_hash = tracker.end_experiment({
            "best_viability": 0.7 + i * 0.1,
            "optimal": True
        })
        print(f"Experiment {i}: {tx_hash[:16]}...")

    # Verify integrity
    print(f"\nVerified: {tracker.audit_chain.verify_integrity()}")

    # Generate report
    report = tracker.audit_chain.generate_report()
    print(f"\nAudit report: {report}")