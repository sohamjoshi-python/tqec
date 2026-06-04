"""Verify optimized ``tqecd`` produces the same detectors as the reference Pauli implementation.

The reference implementation is the pre-optimization dict-based ``PauliString``
(``tqecd.pauli_reference``). Comparison runs in isolated subprocesses so imports
do not leak between modes.

Examples:
    python benchmarks/verify_detector_annotations.py
    python benchmarks/verify_detector_annotations.py --circuit path/to/file.stim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import stim

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CIRCUIT = BENCHMARK_DIR / "fixtures" / "rsc15_z_memory.stim"
WORK_DIR = BENCHMARK_DIR / "data" / "annotation_parity"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuit",
        type=Path,
        action="append",
        default=None,
        help="Stim file to check (default: RSC-15 fixture). May be passed multiple times.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=WORK_DIR / "parity_report.json",
        help="Where to write the JSON comparison report.",
    )
    args = parser.parse_args()

    circuit_paths = args.circuit if args.circuit else [DEFAULT_CIRCUIT]
    for path in circuit_paths:
        if not path.is_file():
            print(f"Circuit not found: {path}", file=sys.stderr)
            raise SystemExit(1)

    from tqecd.annotation_parity import (
        compare_annotation_on_circuit,
        write_comparison_report,
    )

    results = {}
    for circuit_path in circuit_paths:
        circuit = stim.Circuit.from_file(circuit_path)
        label = circuit_path.stem
        print(f"Comparing detectors for {circuit_path} ...")
        result = compare_annotation_on_circuit(
            circuit,
            circuit_label=label,
            work_dir=WORK_DIR / label,
        )
        results[label] = result
        if result.matched:
            print(
                f"  OK: {result.current_detector_count} detectors "
                "(reference and current agree)."
            )
        else:
            print(
                f"  MISMATCH: reference={result.reference_detector_count} "
                f"current={result.current_detector_count}"
            )
            print(f"  only in reference: {len(result.only_in_reference)}")
            print(f"  only in current: {len(result.only_in_current)}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_report(results, args.report)
    print(f"Report written to {args.report}")

    if any(not r.matched for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
