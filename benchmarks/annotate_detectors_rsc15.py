"""Benchmark automatic detector annotation on an RSC-15 Z-memory circuit.

Default circuit: rotated surface code, distance 15, 14 measurement rounds, with
``MR`` replaced by separate ``M`` / ``R`` moments (as required by ``tqecd``).

To use the exact stim text from the issue instead, save it to
``benchmarks/fixtures/rsc15_z_memory.stim`` and pass ``--circuit`` (see
``benchmarks/fixtures/README.md``).

Examples:
    # Wall-clock timing (3 runs, reports median):
    python benchmarks/annotate_detectors_rsc15.py

    # PauliString / tqecd.pauli time breakdown (uses pyinstrument, ~3-4x slower):
    python benchmarks/annotate_detectors_rsc15.py --profile

    # pyinstrument HTML flame graph (open the .html in a browser):
    python -m pyinstrument -o annotate_detectors_rsc_15.html -r html \\
        benchmarks/annotate_detectors_rsc15.py --repeats 1

    # With uv (bench dependency group):
    uv run --group bench python -m pyinstrument -o annotate_detectors_rsc15.html \\
        -r html benchmarks/annotate_detectors_rsc15.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import stim
from tqecd.construction import annotate_detectors_automatically

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CIRCUIT_PATH = BENCHMARK_DIR / "fixtures" / "rsc15_z_memory.stim"


def _transform_generated_circuit_for_tqecd(circuit: stim.Circuit) -> stim.Circuit:
    """Strip annotations and replace ``MR`` with ``M`` (``tqecd``-compatible)."""
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(
                    inst.repeat_count,
                    _transform_generated_circuit_for_tqecd(inst.body_copy()),
                )
            )
        elif isinstance(inst, stim.CircuitInstruction):
            if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"):
                continue
            if inst.name == "MR":
                out.append("M", inst.targets_copy())
            else:
                out.append(inst)
        else:
            out.append(inst)
    return out


def circuit_for_tqecd_from_generated(
    distance: int = 15,
    rounds: int = 14,
) -> stim.Circuit:
    """Build the benchmark circuit from ``stim.Circuit.generated``."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
    )
    return _transform_generated_circuit_for_tqecd(circuit)


def load_circuit(path: Path) -> stim.Circuit:
    if not path.is_file():
        raise FileNotFoundError(
            f"Circuit file not found: {path}\n"
            "Save the issue stim to that path, or run with --regenerate-fixture."
        )
    return stim.Circuit.from_file(path)


def _frame_self_time(frame: object) -> float:
    children = getattr(frame, "children", ())
    return float(getattr(frame, "time")) - sum(
        float(getattr(child, "time")) for child in children
    )


def _pauli_file_self_time(root: object) -> float:
    """Time spent executing lines in ``tqecd/pauli.py`` (excludes child frames)."""
    total = 0.0

    def visit(frame: object) -> None:
        nonlocal total
        path = (getattr(frame, "file_path_short", None) or "").replace("\\", "/")
        if "tqecd/pauli.py" in path:
            total += _frame_self_time(frame)
        for child in getattr(frame, "children", ()):
            visit(child)

    visit(root)
    return total


def _sum_inclusive_time_for_function(root: object, name: str) -> tuple[float, int]:
    """Sum inclusive time over every call site with ``frame.function == name``.

    Many short calls (e.g. thousands of ``anticommutes``) each look small in the
    HTML tree; summing call sites is the right way to see total cost.
    """
    total = 0.0
    count = 0

    def visit(frame: object) -> None:
        nonlocal total, count
        if getattr(frame, "function", None) == name:
            total += float(getattr(frame, "time"))
            count += 1
        for child in getattr(frame, "children", ()):
            visit(child)

    visit(root)
    return total, count


def print_pauli_profile_summary(profiler: object) -> None:
    """Print time spent in ``tqecd.pauli`` (PauliString hot paths)."""
    session = profiler.last_session  # type: ignore[attr-defined]
    root = session.root_frame()
    wall_s = float(getattr(root, "time"))
    pauli_self = _pauli_file_self_time(root)

    print()
    print("Profile breakdown (pyinstrument adds ~3-4x overhead vs plain python):")
    print(f"  wall time (profiled): {wall_s:.3f} s")
    print(
        f"  time in pauli.py itself (self): {pauli_self:.3f} s "
        f"({100 * pauli_self / wall_s:.1f}% of wall)"
    )
    print(
        "  (Most Pauli work is dict/[self] loops in pauli.py; method names below "
        "are per-call-site inclusive sums.)"
    )
    print()
    print("  PauliString methods (sum of inclusive time at every call site):")
    for fn in (
        "to_int",
        "collapse_by",
        "anticommutes",
        "commutes",
        "__getitem__",
        "after",
        "__mul__",
    ):
        total, count = _sum_inclusive_time_for_function(root, fn)
        if total >= 0.05:
            print(f"    {total:7.3f} s  {fn}  ({count:,} call sites)")
    print()
    print("  Note: adding the rows above double-counts (anticommutes runs inside collapse_by).")
    print("  Dominant path in HTML: match -> find_exact_cover -> PauliString.to_int.")
    print("  Full call tree: annotate_detectors_rsc_15.html (from pyinstrument -m ...).")


def run_benchmark(circuit: stim.Circuit, repeats: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        annotated = annotate_detectors_automatically(circuit)
        timings.append(time.perf_counter() - start)
        if annotated.num_detectors == 0:
            raise RuntimeError("annotate_detectors_automatically returned no detectors")
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuit",
        type=Path,
        default=DEFAULT_CIRCUIT_PATH,
        help=f"Path to a .stim file (default: {DEFAULT_CIRCUIT_PATH})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs (default: 3)",
    )
    parser.add_argument(
        "--regenerate-fixture",
        action="store_true",
        help="Overwrite the default fixture from stim.Circuit.generated",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run once under pyinstrument and print PauliString time breakdown",
    )
    args = parser.parse_args()

    if args.profile:
        args.repeats = 1

    if args.regenerate_fixture:
        circuit = circuit_for_tqecd_from_generated()
        args.circuit.parent.mkdir(parents=True, exist_ok=True)
        circuit.to_file(args.circuit)
        print(f"Wrote {args.circuit} ({args.circuit.stat().st_size} bytes)")

    circuit = load_circuit(args.circuit)
    print(f"Circuit: {args.circuit}")
    print(f"  num_qubits={circuit.num_qubits}")
    print(f"  num_measurements={circuit.num_measurements}")
    print(f"  num_detectors (input)={circuit.num_detectors}")

    if args.profile:
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        timings = run_benchmark(circuit, 1)
        profiler.stop()
        print(f"annotate_detectors_automatically (profiled run): {timings[0]:.3f} s")
        print_pauli_profile_summary(profiler)
        return

    timings = run_benchmark(circuit, args.repeats)
    print(f"annotate_detectors_automatically ({args.repeats} runs):")
    for i, t in enumerate(timings):
        print(f"  run {i + 1}: {t:.3f} s")
    print(f"  median: {statistics.median(timings):.3f} s")
    print()
    print("For PauliString breakdown: python benchmarks/annotate_detectors_rsc15.py --profile")
    print("For flame graph: python -m pyinstrument -o annotate_detectors_rsc_15.html -r html \\")
    print("    benchmarks/annotate_detectors_rsc15.py --repeats 1")


if __name__ == "__main__":
    main()
