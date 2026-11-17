#!/usr/bin/env python3
# ~/Projects/Meshlab_Autorepair_App/meshlab_autorepair_app.py
# Batch STL mesh repair via PyMeshLab: clean junk geometry, repair non-manifold topology, close holes, disk-verify the saved artifact, and manage an input/output/processed folder workflow.
# Current Version: 1.1.4
# Current Revision: 2026-11-16-r6
# Previous Revision: none

"""Batch mesh repair via PyMeshLab with a folder-based intake workflow.

Pipeline per mesh: remove duplicate faces/vertices and null/zero-area
faces, drop isolated debris pieces, repair non-manifold edges and
vertices, reorient faces coherently, force normals outward, close holes
up to a maximum boundary size, then verify watertightness TWICE — once
on the in-memory mesh (preliminary) and once on a fresh re-load of the
saved file from disk (authoritative). The saved artifact is what
downstream tools (slicers, DCC apps) load, so the DISK verdict is the
one that names the output: every mesh is saved to a temporary __tmp_
name, verified from disk, then atomically renamed to its final
verdict-tagged name. Binary STL stores 32-bit floats and re-import
re-welds vertices, so the disk mesh can differ from the RAM mesh; when
the two verdicts disagree, the disk verdict wins.

Folder workflow: with no arguments, the app processes all STL/OBJ/PLY
files in the input/ folder, writes outputs to output/ as
<runstamp>_<stem>_<verdict-tag>.stl (fixed/fail/unverified/error), and
moves each consumed original to processed/ immediately after that file
finishes — exact original name preserved; a <runstamp>_ prefix is added
only on a name collision, and nothing is ever overwritten. Non-mesh
entries in input/ are left untouched and flagged for manual removal.
An explicit file or directory argument processes just that path with the
same output and archival rules, without prompting. When launched
interactively with an empty input folder, a native file-picker dialog
offers to identify meshes anywhere on disk; picked files are copied into
input/ (source originals stay where they are) and the normal flow
resumes. Cancelling the picker falls through to the standard no-input
error. A y/N confirmation gate (file list, tunables, destinations)
runs in folder mode on a TTY unless --yes is passed or an explicit path
was given; non-TTY contexts proceed without prompting.

A per-run manifest is written to output/<runstamp>_manifest.txt
(additive to — never a replacement for — the run log in
~/.local/logs/). Aborted or interrupted runs write no manifest.

Safety guards: debris-removal filters delete ANY component below their
thresholds, including the mesh's only component (observed: a 12-face
cube reduced to 0 faces). repair_mesh therefore skips every debris step
when the whole mesh sits at or below --minfaces, and verify_watertight
returns ERROR for a mesh with no geometry, so an emptied mesh can never
earn PASS. Self-intersecting faces are deliberately NOT auto-removed;
that requires human judgment.

Filter-name adaptivity: primary names and parameter spellings below are
confirmed against installed pymeshlab 2025.7.post1 via
print_filter_list, print_filter_parameter_list, and isolated probes
(2026-11-15). Legacy spellings follow as fallbacks. A step whose every
candidate fails is SKIPPED with a WARNING plus per-candidate first-line
errors; the version's own introspection
(pymeshlab.print_filter_list(),
pymeshlab.print_filter_parameter_list('<filter>')) resolves the drift.

Requires: pymeshlab (installed via this project's Poetry environment).
Run inside the project, or use the ~/scripts/Meshlab_Autorepair_Run.zsh
wrapper:
  poetry run python meshlab_autorepair_app.py
  poetry run python meshlab_autorepair_app.py model.stl
  poetry run python meshlab_autorepair_app.py ./meshes -o ./repaired
  poetry run python meshlab_autorepair_app.py --dry-run

Folders (script-relative, overridable via MESHLAB_AUTOREPAIR_APP_*):
  input/     meshes waiting to be processed
  output/    repaired outputs + per-run manifest
  processed/ archive of consumed originals

Exit codes:
  0    all files processed without processing errors (individual verdicts
       may be FAIL, which is not a script error); also a y/N decline
  1    no input found (empty input/ and no picks), or one or more files
       raised processing errors
  2    bad arguments (argparse), including a non-mesh explicit input
  3    another instance holds the lock
  130 interrupted by user (Ctrl-C)

Logs: ~/.local/logs/meshlab_autorepair_app/meshlab_autorepair_app-YYYY-MM-DD_HHMMSS.log
(latest.log symlink points at the current run; overridable via
MESHLAB_AUTOREPAIR_APP_LOG_DIR).
"""

import argparse
import os
import re
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

SCRIPT_PATH = Path(__file__).resolve().parent
SCRIPT_STEM = "meshlab_autorepair_app"
LOG_STEM = "meshlab_autorepair_app"

DEFAULT_LOG_DIR = Path.home() / ".local" / "logs" / LOG_STEM
INPUT_DIR = Path(
    os.environ.get("MESHLAB_AUTOREPAIR_APP_INPUT_DIR", SCRIPT_PATH / "input")
)
OUTPUT_DIR = Path(
    os.environ.get("MESHLAB_AUTOREPAIR_APP_OUTPUT_DIR", SCRIPT_PATH / "output")
)
PROCESSED_DIR = Path(
    os.environ.get("MESHLAB_AUTOREPAIR_APP_PROCESSED_DIR", SCRIPT_PATH / "processed")
)
PICK_FOLDER = Path(
    os.environ.get("MESHLAB_AUTOREPAIR_APP_PICK_FOLDER", Path.home() / "Downloads")
)
LOG_DIR = Path(os.environ.get("MESHLAB_AUTOREPAIR_APP_LOG_DIR", DEFAULT_LOG_DIR))
LOCK_DIR = Path(os.environ.get("MESHLAB_AUTOREPAIR_APP_LOCK_DIR", LOG_DIR / ".lock"))
NOTIFY_WATCHDOG_SECS = float(os.environ.get("MESHLAB_AUTOREPAIR_APP_NOTIFY_WATCHDOG_SECS", "30"))

MESH_SUFFIXES = {".stl", ".obj", ".ply"}
VERDICT_TAGS = {"PASS": "fixed", "FAIL": "fail", "UNVERIFIED": "unverified", "ERROR": "error"}
TMP_PREFIX = "__tmp_"


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
OPEN_BIN = Path("/usr/bin/open")

DRY_RUN = False
LOG_FILE = None
OWN_LOCK = False
RUN_STAMP = None
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
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            watchdog = threading.Timer(NOTIFY_WATCHDOG_SECS, proc.kill)
            watchdog.daemon = True
            watchdog.start()
        elif OSA_BIN.exists() and os.access(OSA_BIN, os.X_OK):
            cmd = [
                str(OSA_BIN), "-e",
                f'display notification "{message}" with title "{title}" sound name "Glass"',
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
        log(f"Retention: {len(logs)} log file(s) present — all protected "
            f"(<= KEEP_MIN_RUNS={KEEP_MIN_RUNS}).")
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
            ("mesh is EMPTY — the pipeline deleted all geometry, or the "
             "input contained none; refusing to award any verdict")
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
    0 faces by the face-count filter). The always-failing {'method': 1}
    candidate for meshing_repair_non_manifold_vertices was dropped after it
    failed on every file in every observed run of this pipeline."""
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


def sanitize_stem(name: str) -> str:
    """Filesystem-safe stem for use inside output filenames: spaces and
    characters a shell treats specially become underscores, so run stamps,
    verdict tags, and paths remain parseable and glob-friendly."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unnamed"


def unique_output_path(directory: Path, filename: str) -> Path:
    """Return a non-clobbering path: if filename already exists in the
    directory (possible only with same-stem inputs in different formats
    in one run — the shared run stamp cannot collide across runs), append
    _2, _3, ... before the extension, with a loud WARNING naming both."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, ext = os.path.splitext(filename)
    n = 2
    while True:
        candidate = directory / f"{stem}_{n}{ext}"
        if not candidate.exists():
            break
        n += 1
    log(f"  WARNING: output name collision — writing {candidate.name} instead of "
        f"{filename} (same-stem inputs in one run; nothing was overwritten).")
    return candidate


def verify_saved_file(out_path: Path, faces_expected: int) -> tuple:
    """Round-trip disk verification: re-load the saved STL into a fresh
    MeshSet and re-run verify_watertight on the re-loaded mesh. The saved
    artifact is what downstream tools load, so this verdict — not the
    in-memory one — is authoritative for naming the output. Basic
    integrity first: file exists and is non-empty, re-load succeeds, and
    the re-loaded face count is within a heuristic +/-10% band of the
    count that was saved (binary STL stores 32-bit floats and re-import
    re-welds vertices; wild face-count divergence indicates an anomaly
    and is surfaced even when the topology check passes)."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return "ERROR", ["saved file missing or empty after write"]
    ms_disk = pymeshlab.MeshSet()
    try:
        ms_disk.load_new_mesh(str(out_path))
    except Exception as exc:
        return "ERROR", [f"saved file could not be re-loaded: {exc}"]
    disk_verdict, disk_lines = verify_watertight(ms_disk)
    if faces_expected > 0:
        faces_disk = ms_disk.current_mesh().face_number()
        if faces_disk < int(faces_expected * 0.9) or faces_disk > int(faces_expected * 1.1):
            disk_lines = list(disk_lines) + [
                (f"WARNING: face-count anomaly on re-load — saved {faces_expected} "
                 f"faces, re-loaded {faces_disk} (outside the heuristic +/-10% band).")
            ]
    return disk_verdict, disk_lines


def repair_mesh(in_path: Path, outdir, maxholesize: int,
                minratio: float, minfaces: int) -> dict:
    """Runs the pipeline on one mesh; returns a result dict. Raises on
    load/save errors only — individual filter steps degrade with warnings.
    Debris steps are skipped when the whole mesh is at or below --minfaces
    (whole-mesh guard). Two-phase save: the repaired mesh is first written
    to a temporary __tmp_ name, verified by re-loading it FROM DISK, and
    only then atomically renamed to its final verdict-tagged name — so a
    RAM-PASS can never mislabel an artifact whose on-disk round-trip
    fails. An interrupt during save/verify removes the temp file and
    re-raises."""
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
    ram_verdict, _ram_lines = verify_watertight(ms)
    log(f"  Preliminary (in-memory) verdict: {ram_verdict}")
    faces_pre_save = ms.current_mesh().face_number()
    base = Path(outdir) if outdir else OUTPUT_DIR
    stem = sanitize_stem(in_path.stem)
    tmp_path = base / f"{TMP_PREFIX}{RUN_STAMP}_{stem}.stl"
    try:
        base.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(tmp_path))
        disk_verdict, disk_lines = verify_saved_file(tmp_path, faces_pre_save)
        tag = VERDICT_TAGS.get(disk_verdict, "error")
        out_path = unique_output_path(base, f"{RUN_STAMP}_{stem}_{tag}.stl")
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    if disk_verdict == "PASS" and ram_verdict == "PASS":
        verdict = "PASS"
        detail_lines = ["RAM and disk verdicts agree (PASS)."] + disk_lines
    elif disk_verdict == "ERROR":
        verdict = "ERROR"
        detail_lines = ["Disk verification failed (file integrity or re-load)."] + disk_lines
    else:
        verdict = disk_verdict
        detail_lines = [
            (f"WARNING: in-memory verdict was {ram_verdict} but the on-disk verdict "
             f"is {disk_verdict} — the STL round-trip altered the mesh (32-bit float "
             "quantization and/or vertex re-welding on import); the disk verdict is "
             "authoritative."),
        ] + disk_lines
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


def collect_files(src: Path) -> tuple:
    """Returns (mesh_files, ignored_entries). Mesh files are regular files
    with a supported suffix, sorted lexicographically; everything else
    found directly in the folder (unsupported suffix or subdirectory) is
    reported as ignored so the operator can remove it manually."""
    meshes = []
    ignored = []
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.is_dir():
                ignored.append((p, "subdirectory (pass it explicitly to process its contents)"))
            elif p.is_file() and p.suffix.lower() in MESH_SUFFIXES:
                meshes.append(p)
            else:
                ignored.append((p, "unsupported extension — not processed, remove manually"))
    elif src.is_file():
        if src.suffix.lower() in MESH_SUFFIXES:
            meshes.append(src)
        else:
            print(f"Refusing non-mesh input: {src} "
                  f"(supported: {', '.join(sorted(MESH_SUFFIXES))})", file=sys.stderr)
            sys.exit(2)
    return meshes, ignored


def archive_original(in_path: Path) -> tuple:
    """Move the consumed original to the archive folder. The ORIGINAL NAME
    is always preserved: a plain move when no same-named file exists; a
    <runstamp>_ prefix ONLY on collision (never overwrites). Returns
    (archived_path_or_None, status) where status is ARCHIVED or
    ARCHIVE_PENDING (move failed — original stays in input/, surfaced in
    summary and manifest)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / in_path.name
    if dest.exists():
        dest = unique_output_path(PROCESSED_DIR, f"{RUN_STAMP}_{in_path.name}")
        log(f"  WARNING: archive collision — storing as {dest.name} "
            "(an earlier original with this name is already archived).")
    try:
        shutil.move(str(in_path), str(dest))
    except OSError as exc:
        log(f"  WARNING: archive move failed ({exc}) — original left in input/ "
            "(ARCHIVE_PENDING).")
        return None, "ARCHIVE_PENDING"
    return dest, "ARCHIVED"


def _copy_picks_into_input(picked: list) -> list:
    """Copy picked files into input/ (source originals stay where they
    are). Name collisions inside input/ are skipped with a notice — the
    resident file takes priority and will be processed with this run.
    Returns the list of newly copied files."""
    copied = []
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in picked:
        dest = INPUT_DIR / p.name
        if dest.exists():
            print(f"Skipping pick {p.name}: a file with this name is already "
                  "in input/ (it will be processed with this run).")
            continue
        shutil.copy2(str(p), str(dest))
        copied.append(dest)
    return copied


def pick_files_interactive() -> list:
    """Empty-input interactive pick (TTY only): present a native choose
    file dialog (multiple selections allowed) via osascript so the operator
    can identify meshes anywhere on disk. Picked files are validated
    (readable, supported suffix) and copied into input/. Clicking Cancel
    returns an empty list, which the caller treats
    as the standard no-input error; a picker FAILURE (any non-cancel
    osascript error) prints the osascript error text and falls back to
    terminal path entry. Fallback when osascript is unavailable:
    open the pick folder in Finder and read whitespace-separated paths
    from stdin (iTerm2 drag-and-drop inserts the path); an empty line
    cancels. Ctrl-C aborts via the caller's KeyboardInterrupt handler.
    IMPORTANT: this function must never be reached in dry-run mode — the
    caller gates it on DRY_RUN being false."""
    if OSA_BIN.exists() and os.access(OSA_BIN, os.X_OK):
        script = (
            "set picks to choose file with prompt "
            "'Select mesh file(s) to process (STL/OBJ/PLY)' "
            "with multiple selections allowed\n"
            "set outText to \"\"\n"
            "repeat with f in picks\n"
            "set outText to outText & POSIX path of f & linefeed\n"
            "end repeat\n"
            "return outText"
        )
        try:
            proc = subprocess.run(
                [str(OSA_BIN), "-e", script],
                capture_output=True, text=True, check=False,
            )
        except Exception as exc:
            print(f"File picker failed to launch ({exc}); falling back to path entry.")
        else:
            if proc.returncode != 0:
                if "-128" in proc.stderr:
                    print("File selection cancelled — no files identified.")
                    return []
                print(f"FILE PICKER FAILED (osascript exit {proc.returncode}): "
                      f"{proc.stderr.strip() or '(no stderr text)'}")
                print("Falling back to terminal path entry.")
            raw = proc.stdout.strip()
            if not raw:
                print("File selection cancelled — no files identified.")
                return []
            picked = [Path(x) for x in raw.splitlines() if x.strip()]
            bad = [p for p in picked if not p.is_file() or p.suffix.lower() not in MESH_SUFFIXES]
            if bad:
                print("Rejected (not a readable STL/OBJ/PLY file): "
                      + ", ".join(str(b) for b in bad))
                return pick_files_interactive()
            return _copy_picks_into_input(picked)
    # stdin fallback (osascript missing or failed to launch)
    PICK_FOLDER.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([str(OPEN_BIN), str(PICK_FOLDER)], check=False)
    except Exception as exc:
        print(f"WARNING: could not open the pick folder {PICK_FOLDER} in Finder ({exc}).")
    print(f"Type or drag-and-drop the path(s) of mesh file(s) to process into "
          f"the terminal, separated by spaces, then press Return (empty line "
          f"cancels; pick folder {PICK_FOLDER} was opened):")
    line = sys.stdin.readline().strip()
    if not line:
        return []
    picked = [Path(tok) for tok in line.split()]
    bad = [p for p in picked if not p.is_file() or p.suffix.lower() not in MESH_SUFFIXES]
    if bad:
        print("Rejected (not a readable STL/OBJ/PLY file): "
              + ", ".join(str(b) for b in bad))
        return pick_files_interactive()
    return _copy_picks_into_input(picked)


def confirm_interactive(files: list, ignored: list, args) -> bool:
    """Folder-mode confirmation gate (TTY, no explicit path): show the file
    list with sizes, ignored entries, tunables, and destinations; y/Y/Enter
    proceeds, anything else declines (caller exits 0 with nothing done).
    EOF on stdin is treated as a decline so scripted feeds cannot hang."""
    print()
    print(f"Run stamp:          {RUN_STAMP}")
    print(f"Files to process:  {len(files)}")
    for f in files:
        try:
            size = f.stat().st_size
        except OSError:
            size = "?"
        print(f"  {f.name}  ({size} bytes)")
    if ignored:
        print(f"Ignored entries in input/ ({len(ignored)}) — NOT processed, "
              "remove manually:")
        for p, why in ignored:
            print(f"  {p.name}: {why}")
    print(f"Outputs to:         {OUTPUT_DIR} as {RUN_STAMP}_<name>_<verdict>.stl")
    print(f"Originals moved to: {PROCESSED_DIR}")
    print(f"Tunables:           maxholesize={args.maxholesize} "
          f"minratio={args.minratio} minfaces={args.minfaces}")
    print()
    try:
        answer = input("Process these files? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "")


def finalize_manifest(results: list, args) -> None:
    """Per-run manifest written to output/<runstamp>_manifest.txt: header,
    one record per file (input, output, verdict, archive destination and
    status, per-file elapsed), then totals. ADDITIVE to the run log — the
    log continues to live in ~/.local/logs/ under its own naming and
    retention. Aborted and interrupted runs write no manifest (a manifest
    implies completeness)."""
    path = OUTPUT_DIR / f"{RUN_STAMP}_manifest.txt"
    counts = {"PASS": 0, "FAIL": 0, "UNVERIFIED": 0, "ERROR": 0}
    lines = [
        f"Manifest for run {RUN_STAMP}",
        f"pymeshlab: {version_string()}",
        (f"Tunables: maxholesize={args.maxholesize} "
        f"minratio={args.minratio} minfaces={args.minfaces}"),
    ]
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        lines.append("")
        lines.append(f"input:    {r['input']}")
        lines.append(f"output:   {r.get('output') or '(none — processing error)'}")
        lines.append(f"verdict:  {r['verdict']}")
        if r.get("archive"):
            lines.append(f"archive:  {r['archive'][0]} ({r['archive'][1]})")
        else:
            lines.append(f"archive:  {r.get('archive_status', '-')}")
        lines.append(f"elapsed:  {r.get('elapsed', '?')} s")
    lines.append("")
    lines.append(f"Totals: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
                 f"{counts['UNVERIFIED']} UNVERIFIED, {counts['ERROR']} ERROR "
                 f"of {len(results)} processed.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Manifest written: {path}")


def cleanup_tmp_artifacts(base: Path) -> None:
    """Remove stale __tmp_* artifacts left in the output directory by an
    interrupted previous run (the interrupt path deletes its own temp
    file, but a hard kill -9 or power loss bypasses it). Every removal is
    logged; failures to remove are warnings, never fatal."""
    base.mkdir(parents=True, exist_ok=True)
    for leftover in sorted(base.glob(f"{TMP_PREFIX}*")):
        try:
            leftover.unlink()
            log(f"Removed stale temp artifact from an interrupted run: {leftover.name}")
        except OSError as exc:
            log(f"WARNING: could not remove stale temp artifact {leftover.name} ({exc}).")

def _handle_sigint(signum, frame):
    # Convert SIGINT/SIGTERM into SystemExit(130) so the finally-block in
    # main() can still run cleanup (lock release, notification). The flag
    # is set here so an interrupt landing outside the per-file try block
    # (between files or during summary) still notifies as interrupted.
    global INTERRUPTED
    INTERRUPTED = True
    raise SystemExit(130)

def summarize(results: list, ignored_count: int) -> None:
    """Terminal + log summary: per-file verdict with output path, totals,
    attention guidance, ARCHIVE_PENDING surfacing, and the manual-cleanup
    reminders (archive folder size, non-mesh leftovers in input/)."""
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
    if any(r.get("archive_status") == "ARCHIVE_PENDING" for r in results):
        log("ARCHIVE_PENDING: one or more originals could not be moved to "
            f"{PROCESSED_DIR} and remain in input/ — resolve manually.")
    if ignored_count:
        log(f"NOTE: {ignored_count} non-mesh entr(ies) remain in input/ — "
            "they were NOT processed and must be removed manually.")
    try:
        proc_entries = list(PROCESSED_DIR.iterdir())
        proc_size = sum(p.stat().st_size for p in proc_entries if p.is_file())
        proc_note = f"{len(proc_entries)} entr(ies), {proc_size} bytes)"
    except OSError:
        proc_note = "size unavailable"
    log(f"REMINDER: review and clean up the archive folder — "
        f"{PROCESSED_DIR} ({proc_note}).")

def main(argv=None) -> int:
    global DRY_RUN, LOG_FILE, RUN_STAMP, INTERRUPTED
    ap = argparse.ArgumentParser(
        description="Batch mesh repair (PyMeshLab) with folder-based intake",
        epilog=f"Logs are written to {LOG_DIR} with a latest.log symlink.",
    )
    ap.add_argument("input", nargs="?", default=None,
                    help="STL/OBJ/PLY file, or a directory of meshes "
                         "(default: the input/ folder next to this script)")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Output directory (default: output/ folder next to "
                         "this script)")
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
    ap.add_argument("--yes", action="store_true",
                    help="Skip the interactive confirmation prompt (folder mode)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be processed; no side effects, no "
                         "log file, no lock, no notification, no dialogs")
    args = ap.parse_args(argv)
    DRY_RUN = args.dry_run

    explicit_mode = args.input is not None
    interactive = sys.stdin.isatty()
    if explicit_mode:
        src = Path(args.input)
        if not src.exists():
            print(f"Input path not found: {src}", file=sys.stderr)
            return 1
        meshes, ignored = collect_files(src)
    else:
        src = INPUT_DIR
        meshes, ignored = collect_files(src)
        if not meshes and interactive and not DRY_RUN:
            try:
                print("input/ contains no mesh files.")
                picked = pick_files_interactive()
            except KeyboardInterrupt:
                print("Interrupted by user.")
                return 130
            if picked:
                meshes, ignored = collect_files(INPUT_DIR)
        if not meshes:
            msg = (f"No STL/OBJ/PLY files found at {src} — add meshes to "
                   f"{INPUT_DIR} or pass an explicit file/directory path.")
            print(msg, file=sys.stderr)
            return 1

    if DRY_RUN:
        print("DRY-RUN: would process the following files (no side effects):")
        for f in meshes:
            print(f"  {f}")
        out_base = args.outdir if args.outdir else OUTPUT_DIR
        print(f"Outputs would be written to {out_base} "
              "as <runstamp>_<stem>_<verdict>.stl")
        print(f"Originals would be moved to {PROCESSED_DIR}")
        if ignored:
            print(f"Ignored entries (would be left untouched): {len(ignored)}")
        print(f"A live run would log to {LOG_DIR}/ and notify on completion.")
        return 0

    # --- side effects begin (dry-run returned above) ---
    RUN_STAMP = time.strftime("%Y-%m-%d_%H%M%S")

    if not explicit_mode and interactive and not args.yes:
        try:
            confirmed = confirm_interactive(meshes, ignored, args)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            return 130
        if not confirmed:
            print("Declined by user — nothing processed.")
            return 0

    for folder in (INPUT_DIR, OUTPUT_DIR, PROCESSED_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / f"{LOG_STEM}-{RUN_STAMP}.log"
    LOG_FILE.touch()
    tmp_link = LOG_DIR / f"latest.log.{os.getpid()}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(LOG_FILE, tmp_link)
    os.replace(tmp_link, LOG_DIR / "latest.log")

    log("────────── meshlab_autorepair_app ──────────")
    log(f"pymeshlab version: {version_string()}")
    log(f"Input: {src}")
    log(f"Output dir: {args.outdir if args.outdir else OUTPUT_DIR}")
    log(f"Tunables: maxholesize={args.maxholesize} minratio={args.minratio} "
        f"minfaces={args.minfaces}")
    log(f"Log: {LOG_FILE}")
    if not explicit_mode and not interactive:
        log("Non-interactive context: proceeding without confirmation.")
    for p, why in ignored:
        log(f"WARNING: ignoring {p.name}: {why}")

    acquire_lock()
    prune_old_logs()
    out_base = Path(args.outdir) if args.outdir else OUTPUT_DIR
    cleanup_tmp_artifacts(out_base)

    results = []
    exit_code = 0
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
        signal.signal(signal.SIGTERM, _handle_sigint)
        total = len(meshes)
        for idx, f in enumerate(meshes, 1):
            log(f"Processing {f.name} ({idx}/{total}) ...")
            t0 = time.monotonic()
            try:
                r = repair_mesh(f, args.outdir, args.maxholesize,
                                args.minratio, args.minfaces)
                log(f"  -> {r['verdict']}")
            except SystemExit as exc:
                if exc.code == 130:
                    INTERRUPTED = True
                raise
            except Exception as exc:
                r = {
                    "input": str(f), "output": "", "verdict": "ERROR",
                    "detail": [str(exc)], "degraded": [],
                }
                log(f"  -> ERROR: {exc}")
            r["elapsed"] = int(time.monotonic() - t0)
            if not INTERRUPTED:
                archive, status = archive_original(f)
                r["archive"] = (archive, status)
                r["archive_status"] = status
                if archive is not None:
                    log(f"  Archived original -> {archive}")
            results.append(r)
        summarize(results, len(ignored))
        finalize_manifest(results, args)
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
                pass  # early sys.exit paths already handled their own messaging
            elif any(r["verdict"] == "ERROR" for r in results):
                detail = (f"{n_bad} of {len(results)} file(s) failed processing "
                          f"after {elapsed_secs()} s.")
                if ignored:
                    detail += (f" {len(ignored)} non-mesh file(s) left in input/, "
                               "remove manually.")
                notify_result("FAILED", detail)
            elif n_bad or ignored:
                detail = (f"{n_bad} of {len(results)} mesh(es) need review "
                          f"after {elapsed_secs()} s.")
                if ignored:
                    detail += (f" {len(ignored)} non-mesh file(s) left in input/, "
                               "remove manually.")
                notify_result("ISSUES", detail)
            else:
                notify_result("SUCCESS",
                             f"{len(results)} mesh(es) repaired and verified "
                             f"watertight in {elapsed_secs()} s.")
            if results and not INTERRUPTED:
                try:
                    subprocess.run([str(OPEN_BIN), str(PROCESSED_DIR)], check=False)
                except Exception as exc:
                    log(f"WARNING: could not open {PROCESSED_DIR} in Finder ({exc}).")
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
