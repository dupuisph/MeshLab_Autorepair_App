#!/usr/bin/env python3
# ~/Projects/Meshlab_Autorepair_App/meshlab_autorepair_app.py
# Batch STL mesh repair via PyMeshLab: clean junk geometry, repair non-manifold topology, close holes, and verify watertightness.
# Current Version: 1.0.3
# Current Revision: 2026-11-15-r6
# Previous Revision: none

"""Batch mesh repair via PyMeshLab.

Pipeline per mesh: remove duplicate faces/vertices and null/zero-area faces,
drop isolated debris pieces, repair non-manifold edges and vertices, reorient
faces coherently, optionally force normals outward, close holes up to a
maximum boundary size, then verify watertightness from topological measures
(boundary_edges == 0 and number_holes == 0 => PASS).

Safety guards: the debris-removal filters delete ANY component below their
thresholds, including the mesh's only component (observed: a 12-face cube
reduced to 0 faces). repair_mesh therefore skips every debris step when the
whole mesh sits at or below --minfaces, and verify_watertight returns ERROR
for an empty mesh, so an emptied mesh can never earn PASS and ships as
_fixed_REVIEW.stl.

Outputs are named <stem>_fixed.stl on PASS and <stem>_fixed_REVIEW.stl when
the verdict is FAIL, UNVERIFIED, or ERROR, so unconfirmed meshes never
masquerade as clean. Self-intersecting faces are deliberately NOT
auto-removed; that requires human judgment.

Filter-name adaptivity: primary names and parameter spellings below are
confirmed against installed pymeshlab 2025.7.post1 via print_filter_list,
print_filter_parameter_list, and isolated probes (2026-11-15). Legacy
spellings follow as fallbacks. A step whose every candidate fails is SKIPPED
with a WARNING plus per-candidate first-line errors; the version's own
introspection (pymeshlab.print_filter_list(),
pymeshlab.print_filter_parameter_list('<filter>')) resolves the drift.

Requires: pymeshlab (installed via this project's Poetry environment).
Run inside the project:
  poetry run python meshlab_autorepair_app.py model.stl
  poetry run python meshlab_autorepair_app.py ./meshes -o ./repaired --maxholesize 40
  poetry run python meshlab_autorepair_app.py model.stl --dry-run

Exit codes:
  0    all files processed without processing errors (individual verdicts
       may be FAIL, which is not a script error)
  1    no input found, or one or more files raised processing errors
  2    bad arguments (argparse)
  3    another instance holds the lock
  130  interrupted by user (Ctrl-C) during processing

Logs: ~/.local/logs/meshlab_autorepair_app/meshlab_autorepair_app-YYYY-MM-DD_HHMMSS.log
(latest.log symlink points at the current run; overridable via
MESHLAB_AUTOREPAIR_APP_LOG_DIR).
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import pymeshlab
except ImportError:
    sys.exit(
        "PyMeshLab not importable. Run inside the project's Poetry venv:\n"
        "  poetry run python meshlab_autorepair_app.py <input>"
    )

SCRIPT_STEM = "meshlab_autorepair_app"
LOG_STEM = "meshlab_autorepair_app"
LOG_DIR = Path(
    os.environ.get("MESHLAB_AUTOREPAIR_APP_LOG_DIR", Path.home() / ".local" / "logs" / LOG_STEM)
)
LOCK_DIR = Path(os.environ.get("MESHLAB_AUTOREPAIR_APP_LOCK_DIR", LOG_DIR / ".lock"))
NOTIFY_WATCHDOG_SECS = float(os.environ.get("MESHLAB_AUTOREPAIR_APP_NOTIFY_WATCHDOG_SECS", "30"))


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"Invalid {name}='{raw}': must be an integer.")


RETENTION_DAYS = env_int("MESHLAB_AUTOREPAIR_APP_RETENTION_DAYS", 90)
KEEP_MIN_RUNS = env_int("MESHLAB_AUTOREPAIR_APP_KEEP_MIN_RUNS", 10)

NOTIFIER = Path("/opt/homebrew/bin/terminal-notifier")
OSA_BIN = Path("/usr/bin/osascript")

DRY_RUN = False
LOG_FILE = None
OWN_LOCK = False
START_MONO = time.monotonic()
INTERRUPTED = False


def elapsed_secs() -> int:
    return int(time.monotonic() - START_MONO)


def log(line: str) -> None:
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}"
    print(stamped, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(stamped + "\n")


def notify_result(status_kind: str, detail: str = "") -> None:
    """Completion notification per the notification contract: title is the
    script stem, status_kind maps to the fixed-status vocabulary, detail is
    the short status update. Suppressed entirely in dry-run mode."""
    if DRY_RUN:
        return
    vocab = {
        "SUCCESS": "Script Ran Successfully",
        "ISSUES": "Script Ran With Issues",
        "FAILED": "Script Failed",
    }
    status_line = vocab.get(status_kind, "Script Failed")
    message = f"{status_line} — {detail}" if detail else status_line
    title = SCRIPT_STEM
    try:
        if NOTIFIER.exists() and os.access(NOTIFIER, os.X_OK):
            cmd = [
                str(NOTIFIER), "-title", title, "-message", message,
                "-sound", "Glass", "-group", LOG_STEM,
                "-execute", f"/usr/bin/open '{LOG_DIR}/latest.log'",
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            watchdog = threading.Timer(NOTIFY_WATCHDOG_SECS, proc.kill)
            watchdog.daemon = True
            watchdog.start()
        elif OSA_BIN.exists() and os.access(OSA_BIN, os.X_OK):
            cmd = [
                str(OSA_BIN), "-e",
                (
                    f'display notification "{message}" with title "{title}" '
                    f'sound name "Glass"'
                ),
            ]
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, check=False,
            )
    except Exception as exc:  # notification must never crash the run
        log(f"WARNING: notification could not be sent ({exc}).")


def acquire_lock() -> None:
    """Single-instance lock: atomic mkdir with PID stale-holder detection."""
    global OWN_LOCK
    try:
        LOCK_DIR.mkdir()
    except FileExistsError:
        pid_text = ""
        try:
            pid_text = (LOCK_DIR / "pid").read_text().strip()
        except OSError:
            pass
        holder_alive = False
        if pid_text.isdigit():
            try:
                os.kill(int(pid_text), 0)
                holder_alive = True
            except OSError:
                holder_alive = False
        if holder_alive:
            log(f"ERROR: another instance is running (PID {pid_text}); exiting.")
            notify_result("FAILED", f"Another instance is running (PID {pid_text}).")
            sys.exit(3)
        log(f"WARNING: stale lock detected (recorded PID: {pid_text or 'none'}); recovering.")
        try:
            shutil.rmtree(LOCK_DIR)
            LOCK_DIR.mkdir()
        except OSError as exc:
            log(f"ERROR: could not recover stale lock ({exc}); aborting.")
            notify_result("FAILED", f"Could not recover stale lock ({LOCK_DIR}).")
            sys.exit(3)
    (LOCK_DIR / "pid").write_text(str(os.getpid()))
    OWN_LOCK = True
    log(f"Lock acquired ({LOCK_DIR}, PID {os.getpid()}).")


def release_lock() -> None:
    global OWN_LOCK
    if OWN_LOCK:
        shutil.rmtree(LOCK_DIR, ignore_errors=True)
        OWN_LOCK = False


def prune_old_logs() -> None:
    """Retention: delete logs older than RETENTION_DAYS except the
    KEEP_MIN_RUNS newest; every decision is logged. The glob matches the
    emitted filename pattern meshlab_autorepair_app-*.log."""
    try:
        logs = sorted(
            LOG_DIR.glob(f"{LOG_STEM}-*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError as exc:
        log(f"Retention: could not list log directory ({exc}); skipping.")
        return
    if not logs:
        log("Retention: no log files to evaluate.")
        return
    if len(logs) <= KEEP_MIN_RUNS:
        log(
            f"Retention: {len(logs)} log file(s) present — all protected "
            f"(<= KEEP_MIN_RUNS={KEEP_MIN_RUNS})."
        )
        return
    cutoff = time.time() - RETENTION_DAYS * 86400
    for idx, candidate in enumerate(logs):
        if idx < KEEP_MIN_RUNS:
            log(f"Retention: protecting {candidate.name} (newest {KEEP_MIN_RUNS}).")
        elif candidate.stat().st_mtime < cutoff:
            log(f"Retention: deleting {candidate.name} (older than {RETENTION_DAYS} days).")
            try:
                candidate.unlink()
            except OSError as exc:
                log(f"Retention: could not delete {candidate.name} ({exc}).")
        else:
            log(f"Retention: keeping {candidate.name} (within RETENTION_DAYS={RETENTION_DAYS}).")


def as_percentage(value):
    """Build the pymeshlab.PercentageValue typed object required by the
    installed version's connected-component-by-diameter filter (a plain
    int or float is rejected by apply_filter). Returns the object, or
    None if construction fails — callers fall back or degrade loudly."""
    try:
        return pymeshlab.PercentageValue(value)
    except Exception as exc:
        log(f"  WARNING: pymeshlab.PercentageValue({value}) construction failed ({exc}).")
        return None


def apply_step(ms, label: str, candidates: list, degraded: list) -> str:
    """Try each candidate (name, params) spelling; return the name applied.
    Every candidate failure is logged (first line only — pymeshlab embeds
    the entire filter list in its exception text, which is noise here).
    On total failure, degrade with a WARNING instead of aborting the batch."""
    for name, params in candidates:
        try:
            ms.apply_filter(name, **params)
            applied = f"{name} {params}" if params else name
            log(f"  {label}: applied {applied}")
            return name
        except Exception as exc:
            err_text = str(exc).splitlines()
            first = err_text[0] if err_text else type(exc).__name__
            log(f"    {label}: {name} failed: {first}")
    degraded.append(label)
    log(f"  WARNING: {label} SKIPPED — no working candidate "
        f"({[c[0] for c in candidates]})")
    log("    Inspect this version's names/signatures via "
        "pymeshlab.print_filter_list() and "
        "pymeshlab.print_filter_parameter_list('<filter>')")
    return ""


def verify_watertight(ms) -> tuple:
    """Returns (verdict, detail_lines). PASS requires boundary_edges == 0 and
    number_holes == 0 from topological measures (get_topological_measures,
    confirmed in the installed version). An empty mesh returns ERROR — an
    empty mesh has trivially zero boundary edges and holes and must never
    be rewarded for that. Falls back to geometric measures with an explicit
    UNVERIFIED label (volume is NOT proof of watertightness)."""
    cur = ms.current_mesh()
    if cur.face_number() == 0 or cur.vertex_number() == 0:
        return "ERROR", [
            "mesh is EMPTY — the pipeline deleted all geometry, or the "
            "input contained none; refusing to award any verdict"
        ]
    topo = None
    try:
        topo = ms.get_topological_measures()
    except AttributeError:
        log("  get_topological_measures method missing; trying apply_filter.")
    except Exception as exc:
        log(f"  Topological measures via method call failed ({exc}); trying apply_filter.")
    if topo is None:
        try:
            topo = ms.apply_filter("get_topological_measures")
        except Exception as exc:
            topo = None
            log(f"  Topological measures unavailable via apply_filter too ({exc}).")
    if isinstance(topo, dict) and "boundary_edges" in topo and "number_holes" in topo:
        b_edges = topo.get("boundary_edges")
        n_holes = topo.get("number_holes")
        lines = [
            f"boundary_edges={b_edges}",
            f"number_holes={n_holes}",
            f"is_mesh_two_manifold={topo.get('is_mesh_two_manifold')}",
            f"genus={topo.get('genus')}",
            f"connected_components={topo.get('connected_components_number')}",
        ]
        if b_edges == 0 and n_holes == 0:
            return "PASS", lines
        return "FAIL", lines
    log("  WARNING: topological measures unavailable — verdict cannot be proven.")
    geo = None
    try:
        geo = ms.get_geometric_measures()
    except Exception:
        try:
            geo = ms.apply_filter("get_geometric_measures")
        except Exception as exc:
            return "UNVERIFIED", [f"no verification measures available ({exc})"]
    vol = geo.get("mesh_volume") if isinstance(geo, dict) else None
    return "UNVERIFIED", [
        "topological measures unavailable; geometric fallback below",
        f"mesh_volume={vol} (volume != proof of watertightness)",
    ]


def build_steps(maxholesize: int, minratio: float, minfaces: int) -> list:
    """Ordered pipeline: (label, [(filter_name, params), ...]). Names and
    parameter spellings confirmed against installed pymeshlab 2025.7.post1
    via print_filter_list, print_filter_parameter_list, and isolated probes
    (2026-11-15). The debris steps are guarded in repair_mesh: if the whole
    mesh is at or below --minfaces, they are skipped so small legitimate
    models cannot be deleted wholesale (observed: a 12-face cube reduced to
    0 faces by the face-count filter)."""
    dia_pv = as_percentage(minratio)
    dia_candidates = []
    if dia_pv is not None:
        dia_candidates.append(
            ("meshing_remove_connected_component_by_diameter",
             {"mincomponentdiag": dia_pv})
        )
    dia_candidates.append(
        ("remove_isolated_pieces_wrt_diameter", {"minratio": minratio})
    )
    return [
        ("remove duplicate faces", [
            ("meshing_remove_duplicate_faces", {}),
            ("remove_duplicate_faces", {}),
        ]),
        ("remove duplicate vertices", [
            ("meshing_remove_duplicate_vertices", {}),
            ("remove_duplicate_vertices", {}),
        ]),
        ("remove null / zero-area faces", [
            ("meshing_remove_null_faces", {}),
            ("remove_zero_area_faces", {}),
        ]),
        ("remove isolated pieces (diameter)", dia_candidates),
        ("remove isolated pieces (face count)", [
            ("meshing_remove_connected_component_by_face_number",
             {"mincomponentsize": minfaces}),
            ("remove_isolated_pieces_wrt_number_face_tri_edges", {"minsize": minfaces}),
        ]),
        ("repair non-manifold edges", [
            ("meshing_repair_non_manifold_edges", {"method": 1}),
            ("meshing_repair_non_manifold_edges", {}),
        ]),
        ("repair non-manifold vertices", [
            ("meshing_repair_non_manifold_vertices", {"method": 1}),
            ("meshing_repair_non_manifold_vertices", {}),
        ]),
        ("reorient faces coherently", [
            ("meshing_re_orient_faces_coherently", {}),
            ("reorient_face_coherently", {}),
        ]),
        ("force normals outward", [
            ("meshing_re_orient_faces_by_geometry", {}),
            ("normals_force_flip", {}),
        ]),
        ("close holes", [
            ("meshing_close_holes", {"maxholesize": maxholesize}),
            ("close_holes", {"maxholesize": maxholesize}),
        ]),
    ]


def repair_mesh(in_path: Path, outdir, maxholesize: int,
                minratio: float, minfaces: int) -> dict:
    """Runs the pipeline on one mesh; returns a result dict. Raises on
    load/save errors only — individual filter steps degrade with warnings.
    Debris steps are skipped when the whole mesh is at or below --minfaces
    (whole-mesh guard). Output naming depends on the verdict, so
    verification happens BEFORE the output path is chosen and BEFORE saving."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(in_path))
    degraded: list = []
    for label, candidates in build_steps(maxholesize, minratio, minfaces):
        if label.startswith("remove isolated pieces"):
            faces_now = ms.current_mesh().face_number()
            if faces_now <= minfaces:
                log(f"  {label}: SKIPPED (whole-mesh guard) — mesh currently "
                    f"has {faces_now} faces, at or below --minfaces={minfaces}; "
                    "debris removal would delete the entire model")
                continue
        apply_step(ms, label, candidates, degraded)
    verdict, detail_lines = verify_watertight(ms)
    base = Path(outdir) if outdir else in_path.parent
    suffix = "_fixed" if verdict == "PASS" else "_fixed_REVIEW.stl"
    out_path = base / f"{in_path.stem}{suffix}.stl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(out_path))
    for line in detail_lines:
        log(f"  {line}")
    if degraded:
        log(f"  Degraded steps (SKIPPED): {', '.join(degraded)}")
    return {
        "input": str(in_path),
        "output": str(out_path),
        "verdict": verdict,
        "detail": detail_lines,
        "degraded": degraded,
    }


def collect_files(src: Path) -> list:
    if src.is_dir():
        return sorted(
            p for p in src.iterdir()
            if p.is_file() and p.suffix.lower() in {".stl", ".obj", ".ply"}
        )
    return [src]


def _handle_sigint(signum, frame):
    # Convert SIGINT/SIGTERM into SystemExit(130) so the finally-block in
    # main() can still run cleanup (lock release, notification).
    raise SystemExit(130)


def summarize(results: list) -> None:
    log("")
    log("=== SUMMARY ===")
    counts = {"PASS": 0, "FAIL": 0, "UNVERIFIED": 0, "ERROR": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        log(f"{Path(r['input']).name}: {r['verdict']}"
            + (f" -> {r['output']}" if r.get("output") else ""))
    log(f"Totals: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
        f"{counts['UNVERIFIED']} UNVERIFIED, {counts['ERROR']} ERROR "
        f"(of {len(results)} processed).")
    if counts["FAIL"] or counts["UNVERIFIED"] or counts["ERROR"]:
        log("Models needing attention: raise --maxholesize, adjust --minratio, "
            "or fix self-intersections / residual defects by hand.")


def main(argv=None) -> int:
    global DRY_RUN, LOG_FILE, INTERRUPTED
    ap = argparse.ArgumentParser(
        description="Batch mesh repair (PyMeshLab)",
        epilog=f"Logs are written to {LOG_DIR} with a latest.log symlink.",
    )
    ap.add_argument("input", help="STL/OBJ/PLY file, or a directory of meshes")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Output directory (default: alongside inputs)")
    ap.add_argument("--maxholesize", type=int, default=30,
                    help="Max hole boundary size (edges) for hole closing (default 30)")
    ap.add_argument("--minratio", type=float, default=10.0,
                    help="Debris removal diameter ratio for isolated pieces "
                         "(percentage-style value passed to pymeshlab as a "
                         "PercentageValue, default 10).")
    ap.add_argument("--minfaces", type=int, default=25,
                    help="Min face count for the isolated-pieces pass; also "
                         "the whole-mesh guard threshold — a mesh at or below "
                         "this face count skips debris removal entirely so "
                         "small models cannot be deleted (default 25)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be processed; no side effects, no "
                         "log file, no lock, no notification")
    args = ap.parse_args(argv)
    DRY_RUN = args.dry_run

    src = Path(args.input)
    if not src.exists():
        print(f"Input path not found: {src}", file=sys.stderr)
        return 1
    files = collect_files(src)
    if not files:
        print(f"No STL/OBJ/PLY files found at {src}", file=sys.stderr)
        return 1

    if DRY_RUN:
        print("DRY-RUN: would process the following files (no side effects):")
        for f in files:
            print(f"  {f}")
        print("Outputs would be written as <stem>_fixed.stl (PASS) or "
              "<stem>_fixed_REVIEW.stl (FAIL/UNVERIFIED/ERROR)")
        print(f"A live run would log to {LOG_DIR}/ and notify on completion.")
        return 0

    # --- log setup (side effects; dry-run already returned) ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / f"{LOG_STEM}-{time.strftime('%Y-%m-%d_%H%M%S')}.log"
    LOG_FILE.touch()
    tmp_link = LOG_DIR / f"latest.log.{os.getpid()}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(LOG_FILE, tmp_link)
    os.replace(tmp_link, LOG_DIR / "latest.log")

    log("────────── meshlab_autorepair_app ──────────")
    log(f"pymeshlab version: {version_string()}")
    log(f"Input: {src}")
    log(f"Output dir: {Path(args.outdir) if args.outdir else '(alongside inputs)'}")
    log(f"Tunables: maxholesize={args.maxholesize} minratio={args.minratio} "
        f"minfaces={args.minfaces}")
    log(f"Log: {LOG_FILE}")

    acquire_lock()
    prune_old_logs()

    results = []
    exit_code = 0
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
        signal.signal(signal.SIGTERM, _handle_sigint)
        for f in files:
            log(f"Processing {f.name} ...")
            try:
                r = repair_mesh(f, args.outdir, args.maxholesize,
                                args.minratio, args.minfaces)
                results.append(r)
                log(f"  -> {r['verdict']}")
            except SystemExit as exc:
                if exc.code == 130:
                    INTERRUPTED = True
                raise
            except Exception as exc:
                results.append({
                    "input": str(f), "output": "", "verdict": "ERROR",
                    "detail": [str(exc)], "degraded": [],
                })
                log(f"  -> ERROR: {exc}")
        summarize(results)
    finally:
        try:
            n_bad = sum(
                1 for r in results if r["verdict"] in {"FAIL", "UNVERIFIED", "ERROR"}
            )
            if INTERRUPTED:
                log(f"Interrupted by user after {elapsed_secs()} s.")
                notify_result("ISSUES",
                              f"Interrupted by user after {elapsed_secs()} s.")
            elif not results:
                pass  # early sys.exit paths already notified
            elif any(r["verdict"] == "ERROR" for r in results):
                notify_result("FAILED",
                              f"{n_bad} of {len(results)} file(s) failed processing "
                              f"after {elapsed_secs()} s.")
            elif n_bad:
                notify_result("ISSUES",
                              f"{n_bad} of {len(results)} mesh(es) need review "
                              f"after {elapsed_secs()} s.")
            else:
                notify_result("SUCCESS",
                              f"{len(results)} mesh(es) repaired and verified "
                              f"watertight in {elapsed_secs()} s.")
        finally:
            release_lock()
            log(f"Done (elapsed {elapsed_secs()} s). Log: {LOG_FILE}")

    if INTERRUPTED:
        return 130
    if any(r["verdict"] == "ERROR" for r in results):
        return 1
    return exit_code


def version_string() -> str:
    """Installed pymeshlab version via package metadata; modern pymeshlab
    does not expose a module-level __version__ attribute."""
    try:
        from importlib.metadata import version
        return f"pymeshlab {version('pymeshlab')}"
    except Exception:
        return "pymeshlab (version unknown)"


if __name__ == "__main__":
    sys.exit(main())
