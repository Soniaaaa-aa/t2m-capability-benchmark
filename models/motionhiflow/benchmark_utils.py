"""
benchmark_utils.py — reusable logic for the MotionHiFlow benchmark pipeline.

This module holds everything that is stable enough not to belong in notebook cells:
run configuration, output-file lookup, motion standardisation, GIF rendering and the
per-prompt matching score.

Design rules:

- Nothing here modifies the MotionHiFlow repository. Generation still goes through the
  repository's own `run.sh` / `gen_t2m.py`; this module only prepares its inputs and
  post-processes its outputs. That keeps `git pull` conflict-free.
- Heavy dependencies (torch, spacy, the repo's evaluator) are imported lazily inside
  `MatchingScorer`, so the standardisation and rendering path works without them.
- Every function is pure enough to be called from a plain script as well as a notebook.

Usage:
    from benchmark_utils import RunConfig
    cfg = RunConfig(motion_length=100).make_dirs()
    file_manifest = cfg.build_file_manifest(manifest)
    standard_report = cfg.standardise(file_manifest)
    gif_report = cfg.render(standard_report)

Porting to another text-to-motion model:
    The post-processing here is model-agnostic as long as the model emits the HumanML3D
    layout — (T, 22, 3) joint positions and (T, 263) feature vectors. What differs between
    repositories is only naming and coordinate convention, so those are constructor
    arguments rather than edits:

        cfg = RunConfig(
            motion_length=100,
            root=Path("/content/OtherModel"),
            experiment_name="my_experiment",              # output sub-directory
            sample_pattern="sample{index}_r0_len{length}",  # output filename stem
            joints_subdir="joints",
            anim_subdir="animations",
            feature_suffix="_data",
            mirror_x=True,       # set False if the model already emits +X = Right
            n_joints=22,
            feature_dim=263,
        )

    Set `mirror_x` deliberately. Nothing can detect the correct value automatically: the
    facing check only measures the angle to +Z, which is unchanged by a left-right flip.
    Use `check_left_right_convention()` on a "walks to the left" clip to verify it.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "RunConfig",
    "HML3D_KINEMATIC_CHAIN",
    "EVAL_MAX_FRAMES",
    "EVAL_MIN_FRAMES",
    "mhf_folder_name",
    "parse_motion_length",
    "find_sample_file",
    "list_sample_folder",
    "write_prompt_file",
    "build_file_manifest",
    "standardise_motionhiflow",
    "estimate_hml3d_initial_forward",
    "standardise_batch",
    "check_standardisation",
    "check_left_right_convention",
    "get_scene_limits",
    "render_standard_gif",
    "render_batch",
    "package_gifs",
    "DEFAULT_ELEV",
    "DEFAULT_AZIM",
    "MatchingScorer",
    "DEFAULT_SAMPLE_PATTERN",
]

# Output filename stem produced by MotionHiFlow's gen_t2m.py. Override via RunConfig when
# porting to a repository that names its samples differently.
DEFAULT_SAMPLE_PATTERN = "sample{index}_r0_len{length}"

# Camera angle for the rendered GIFs. elev=30 looks down at the scene; a negative elevation
# views it from below, which makes left and right easy to misjudge by eye.
DEFAULT_ELEV = 30
DEFAULT_AZIM = -45


# Matches the {length} field including any format spec, e.g. "{length}" or "{length:04d}".
_LENGTH_FIELD_RE = re.compile(r"\{length(?::[^}]*)?\}")


def _format_stem(sample_pattern: str, index: int, length: int) -> str:
    """Render a filename stem for one concrete (index, length)."""
    return sample_pattern.format(index=index, length=int(length))


def _glob_stem(sample_pattern: str, index: int) -> str:
    """
    Render a filename stem whose length field is a glob wildcard.

    The length placeholder is replaced by a literal '*' BEFORE formatting, because a
    pattern such as "{length:04d}" cannot be formatted with a string and would raise.
    """
    return _LENGTH_FIELD_RE.sub("*", sample_pattern).format(index=index)


# ======================================================================================
# Run configuration
# ======================================================================================

@dataclass
class RunConfig:
    """
    All parameters and derived paths for one benchmark run.

    Only the first three fields are meant to be set by hand; everything else is derived.

    The frame count is baked into `run_tag`, so two runs at different lengths can never
    write into the same directory. That is what prevents the failure mode where a folder
    ends up holding both `sample0_r0_len196.npy` and `sample0_r0_len100.npy`, every file
    lookup becomes ambiguous, and the standardisation step silently produces zero rows.
    """

    motion_length: int = 100
    run_name: str = "pilot13"
    run_version: str = "v1"
    root: Path = Path("/content/MotionHiFlow")

    # --- repository layout: change these when porting to another model ---
    experiment_name: str = "t2m_tmdit_16d"          # output sub-directory of the run
    joints_subdir: str = "joints"                   # (T, n_joints, 3) positions
    anim_subdir: str = "animations"                 # (T, feature_dim) features plus mp4
    sample_pattern: str = DEFAULT_SAMPLE_PATTERN    # filename stem, formatted with index/length
    feature_suffix: str = "_data"                   # distinguishes the feature file
    exclude_suffixes: tuple = ("_ik",)              # stems to ignore, e.g. IK reconstructions
    length_multiple: int = 4                        # the model rounds lengths to this multiple

    # --- data convention: change these when the skeleton or axes differ ---
    n_joints: int = 22          # HumanML3D skeleton
    feature_dim: int = 263      # HumanML3D feature vector
    mirror_x: bool = True       # raw +X points left; mirror it so that +X = Right

    # --- derived; do not pass these in ---
    effective_length: int = field(init=False)
    run_tag: str = field(init=False)
    manifest_csv: Path = field(init=False)
    prompt_txt: Path = field(init=False)
    run_output_root: Path = field(init=False)
    result_root: Path = field(init=False)
    joint_root: Path = field(init=False)
    anim_root: Path = field(init=False)
    standard_root: Path = field(init=False)
    std_joint_root: Path = field(init=False)
    std_gif_root: Path = field(init=False)
    report_file: Path = field(init=False)
    score_csv: Path = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)

        # The model rounds lengths down to a multiple of length_multiple; the result is the
        # number that appears in the generated filenames.
        step = max(1, int(self.length_multiple))
        self.effective_length = (int(self.motion_length) // step) * step
        if self.effective_length < step:
            raise ValueError(
                f"motion_length is too small: {self.motion_length} "
                f"(must be at least {step})"
            )

        self.run_tag = f"{self.run_name}_len{self.effective_length}_{self.run_version}"

        # Generation paths.
        self.manifest_csv = self.root / f"{self.run_tag}_manifest.csv"
        self.prompt_txt = self.root / f"{self.run_tag}_prompts.txt"
        self.run_output_root = self.root / "benchmark_runs" / self.run_tag
        self.result_root = self.run_output_root / self.experiment_name
        self.joint_root = self.result_root / self.joints_subdir
        self.anim_root = self.result_root / self.anim_subdir

        # Standardisation paths.
        self.standard_root = self.root / "standardised" / self.run_tag
        self.std_joint_root = self.standard_root / "joints"
        self.std_gif_root = self.standard_root / "gifs"
        self.report_file = self.standard_root / "standardisation_report.csv"
        self.score_csv = self.standard_root / "matching_scores.csv"

    def make_dirs(self) -> "RunConfig":
        """Create the output directories. Returns self so it can be chained."""
        for d in (self.run_output_root, self.std_joint_root, self.std_gif_root):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def summary(self) -> str:
        """A short human-readable description of this run."""
        example = _format_stem(self.sample_pattern, 0, self.effective_length)
        lines = [
            f"Requested length   : {self.motion_length}",
            f"Effective length   : {self.effective_length} (e.g. {example}.npy)",
            f"RUN_TAG            : {self.run_tag}",
            f"Generation output  : {self.run_output_root}",
            f"Standardised output: {self.standard_root}",
            f"Skeleton           : {self.n_joints} joints, {self.feature_dim}-D features, "
            f"mirror_x={self.mirror_x}",
        ]
        # Warn when this directory already holds results from an earlier run.
        if self.joint_root.exists() and any(self.joint_root.rglob("*.npy")):
            lines += [
                "",
                "Note: this directory already holds results; re-running generation "
                "overwrites them.",
                "      Bump run_version to 'v2' if you want to keep them.",
            ]
        return "\n".join(lines)

    # -------------------------------------------------------------- convenience wrappers
    # These bind the module-level functions to this run's layout and data convention, so
    # a notebook cell stays one line. The functions themselves remain usable standalone.

    def find(self, index: int, prompt: str, kind: str = "joints"):
        """Locate one generated file using this run's naming convention."""
        root = self.anim_root if kind == "data" else self.joint_root
        return find_sample_file(
            root, index, prompt, kind=kind, expected_length=self.effective_length,
            sample_pattern=self.sample_pattern, feature_suffix=self.feature_suffix,
            exclude_suffixes=self.exclude_suffixes,
        )

    def build_file_manifest(self, manifest, verbose: bool = True):
        """Locate the generated joint file for every prompt in the manifest."""
        return build_file_manifest(
            manifest, self.joint_root, self.effective_length,
            sample_pattern=self.sample_pattern, feature_suffix=self.feature_suffix,
            exclude_suffixes=self.exclude_suffixes, verbose=verbose,
        )

    def standardise(self, file_manifest, verbose: bool = True):
        """Standardise every located clip into this run's output directory."""
        return standardise_batch(
            file_manifest, self.std_joint_root,
            mirror_x=self.mirror_x, n_joints=self.n_joints, verbose=verbose,
        )

    def render(self, standard_report, fps: int = 20, elev: float = DEFAULT_ELEV,
               azim: float = DEFAULT_AZIM, verbose: bool = True):
        """Render every standardised clip into this run's GIF directory."""
        return render_batch(standard_report, self.std_gif_root, fps=fps,
                            elev=elev, azim=azim, verbose=verbose)

    def score(self, scorer, manifest, strict: bool = True, verbose: bool = True):
        """Score every prompt with a MatchingScorer, using this run's naming convention."""
        return scorer.score_batch(
            manifest, self.anim_root, self.effective_length,
            sample_pattern=self.sample_pattern, feature_suffix=self.feature_suffix,
            strict=strict, verbose=verbose,
        )


# ======================================================================================
# Output-file lookup
# ======================================================================================

def mhf_folder_name(prompt: str) -> str:
    """Return the output sub-folder name for a prompt (same rule as gen_t2m.py)."""
    return prompt[:100].replace("/", "_").replace(" ", "_")


# Matches an {index} or {length} placeholder including any format spec.
_ANY_FIELD_RE = re.compile(r"\{(index|length)(?::[^}]*)?\}")


def _length_regex(sample_pattern: str) -> "re.Pattern":
    """
    Derive a regex that extracts the length field from a filename built with this pattern.

    The literal parts are escaped and the placeholders become digit groups, so a custom
    pattern such as "gen_{index:02d}_frames{length:04d}" is parsed just as correctly as the
    MotionHiFlow default.
    """
    parts, last = [], 0
    for m in _ANY_FIELD_RE.finditer(sample_pattern):
        parts.append(re.escape(sample_pattern[last:m.start()]))
        parts.append(r"(?P<length>\d+)" if m.group(1) == "length" else r"\d+")
        last = m.end()
    parts.append(re.escape(sample_pattern[last:]))
    return re.compile("".join(parts))


def parse_motion_length(path, sample_pattern: str = DEFAULT_SAMPLE_PATTERN,
                        strict: bool = True):
    """
    Parse the frame count out of a generated filename.

    The regex is derived from `sample_pattern`, so this keeps working after the naming
    convention is changed for another repository.

    strict=False returns None instead of raising, for callers that can fall back to the
    array's own length.
    """
    name = Path(path).name
    m = _length_regex(sample_pattern).search(name)
    if m is None:
        # Fall back to the common "_len<digits>" spelling before giving up.
        m = re.search(r"_len(?P<length>\d+)", name)
    if m is None:
        if strict:
            raise ValueError(
                f"Cannot parse length from filename: {path}\n"
                f"(pattern: {sample_pattern!r})"
            )
        return None
    return int(m.group("length"))


def find_sample_file(root, index: int, prompt: str, kind: str = "joints",
                     expected_length=None, sample_pattern: str = DEFAULT_SAMPLE_PATTERN,
                     feature_suffix: str = "_data", exclude_suffixes=("_ik",)):
    """
    Locate the file generated for one prompt.

    Args:
        root            : search root, e.g. <run>/<experiment>/joints
        index           : index of the prompt, substituted into sample_pattern
        kind            : "joints" -> the plain position file
                          "data"   -> the feature file, i.e. stem + feature_suffix
        expected_length : the requested frame count, already floored to the model's
                          length multiple. When given, an exact match on it wins.
        sample_pattern  : filename stem, formatted with {index} and {length}. The default
                          matches MotionHiFlow's gen_t2m.py; override it for another repo.
        feature_suffix  : what distinguishes the feature file from the position file
        exclude_suffixes: stems to ignore when looking for position files, such as the
                          "_ik" reconstructions MotionHiFlow writes alongside

    Returns:
        (Path | None, status) where status is "OK", "NOT_FOUND" or "AMBIGUOUS".

    Why expected_length matters:
        If several motion lengths were generated under one run tag, a single folder holds
        both sample0_r0_len196.npy and sample0_r0_len100.npy. A bare wildcard matches two
        files, every lookup is flagged AMBIGUOUS, and a downstream Status == "OK" filter
        silently yields zero rows.
    """
    suffix = feature_suffix if kind == "data" else ""
    folder = Path(root) / mhf_folder_name(prompt)
    if not folder.exists():
        return None, "NOT_FOUND"

    # 1) Prefer an exact match on this round's length.
    if expected_length is not None:
        exact = folder / f"{_format_stem(sample_pattern, index, expected_length)}{suffix}.npy"
        if exact.exists():
            return exact, "OK"

    # 2) Fall back to a wildcard match on the length field.
    candidates = sorted(folder.glob(f"{_glob_stem(sample_pattern, index)}{suffix}.npy"))
    if kind != "data":
        # Keep plain position files: drop the feature files and any excluded variants.
        drop = tuple(f"{x}.npy" for x in exclude_suffixes) + (f"{feature_suffix}.npy",)
        candidates = [p for p in candidates if not p.name.endswith(drop)]

    if len(candidates) == 1:
        return candidates[0], "OK"
    if len(candidates) == 0:
        return None, "NOT_FOUND"
    # With several candidates keep the newest one and flag it for manual inspection.
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest, "AMBIGUOUS"


def list_sample_folder(root, index: int, prompt: str) -> str:
    """Diagnostic helper: list everything inside one prompt's output folder."""
    folder = Path(root) / mhf_folder_name(prompt)
    if not folder.exists():
        return f"[folder missing] {folder}"
    names = sorted(p.name for p in folder.iterdir())
    return f"{folder}\n    " + ("\n    ".join(names) if names else "(empty)")


def write_prompt_file(manifest: pd.DataFrame, prompt_txt) -> Path:
    """
    Write the prompt file consumed by `run.sh gen ... text_path=...`.

    MotionHiFlow expects one "prompt#frames" line per entry.
    """
    prompt_txt = Path(prompt_txt)
    with open(prompt_txt, "w") as f:
        for _, row in manifest.iterrows():
            f.write(f"{row['Prompt']}#{int(row['Motion_Length'])}\n")
    return prompt_txt


def build_file_manifest(manifest: pd.DataFrame, joint_root, expected_length,
                        sample_pattern: str = DEFAULT_SAMPLE_PATTERN,
                        feature_suffix: str = "_data", exclude_suffixes=("_ik",),
                        verbose: bool = True) -> pd.DataFrame:
    """
    Locate the generated joint file for every prompt in the manifest.

    On failure it prints the offending folders' contents immediately, rather than letting
    an empty result surface two steps later with no explanation.

    Returns a DataFrame with Prompt_ID, Prompt, Requested_Length, Actual_Length,
    Joint_File and Status.
    """
    records = []
    for i, row in manifest.iterrows():
        joint_file, status = find_sample_file(
            joint_root, i, row["Prompt"], kind="joints", expected_length=expected_length,
            sample_pattern=sample_pattern, feature_suffix=feature_suffix,
            exclude_suffixes=exclude_suffixes,
        )
        records.append({
            "Prompt_ID": row["Prompt_ID"],
            "Prompt": row["Prompt"],
            "Requested_Length": int(row["Motion_Length"]),
            "Actual_Length": (parse_motion_length(joint_file, sample_pattern, strict=False)
                              if joint_file else None),
            "Joint_File": str(joint_file) if joint_file else None,
            "Status": status,
        })

    file_manifest = pd.DataFrame(records)
    if not verbose:
        return file_manifest

    print(file_manifest["Status"].value_counts())

    # MotionHiFlow may clamp the requested length; say so rather than hiding it.
    mismatch = file_manifest[
        file_manifest["Actual_Length"].notna()
        & (file_manifest["Actual_Length"] != expected_length)
    ]
    if not mismatch.empty:
        print(f"\nNote: {len(mismatch)} clip(s) do not have {expected_length} frames.")

    bad = file_manifest[file_manifest["Status"] != "OK"]
    if not bad.empty:
        print(f"\n{len(bad)} lookup(s) failed. Folder contents:")
        for _, row in bad.head(3).iterrows():
            idx = int(manifest.index[manifest["Prompt_ID"] == row["Prompt_ID"]][0])
            print(f"\n[{row['Prompt_ID']}] status = {row['Status']}")
            print("   ", list_sample_folder(joint_root, idx, row["Prompt"]))
        print(
            "\nCommon causes:\n"
            "  NOT_FOUND : changed motion_length without re-running generation\n"
            "  AMBIGUOUS : old files of another length under the same run tag"
        )

    return file_manifest


# ======================================================================================
# Standardisation
# ======================================================================================

def standardise_motionhiflow(motion, mirror_x: bool = True, n_joints: int = 22):
    """
    Standardise one motion clip.

    MotionHiFlow's raw joint positions share no common origin or axis convention, so a
    direct comparison is affected by the initial pose. Three corrections are applied:

    1. Ground alignment  — lift the lowest point of the whole clip to Y = 0.
    2. Origin alignment  — move the first-frame root XZ to (0, 0).
    3. Axis convention   — mirror X so that +X = Right (only when mirror_x is True).

    Final convention: +X = Right, -X = Left, +Z = Forward, -Z = Backward, +Y = Up.

    Args:
        motion   : (T, n_joints, 3) float array
        mirror_x : True for raw HumanML3D output, where +X points to the character's left.
                   Set False for a model that already emits +X = Right — nothing can detect
                   this automatically, see check_left_right_convention().
        n_joints : expected joint count; 22 for the HumanML3D skeleton

    Returns (motion_std, metadata)
    """
    motion = np.asarray(motion, dtype=np.float32)

    # --- Input validation ---
    if motion.ndim != 3:
        raise ValueError(f"Expected [T, J, 3], got {motion.shape}")
    if motion.shape[1] != n_joints:
        raise ValueError(f"Expected {n_joints} joints, got {motion.shape[1]}")
    if motion.shape[2] != 3:
        raise ValueError(f"Expected XYZ, got {motion.shape}")
    if not np.isfinite(motion).all():
        raise ValueError("Motion contains NaN or Inf.")

    motion_std = motion.copy()

    # --- 1. Ground alignment ---
    # Using the global minimum rather than the first frame keeps jumps and crouches from
    # sinking the character into the floor.
    original_floor = float(motion_std[:, :, 1].min())
    motion_std[:, :, 1] -= original_floor

    # --- 2. Origin alignment (joint 0 is the root / pelvis) ---
    original_root_x = float(motion_std[0, 0, 0])
    original_root_z = float(motion_std[0, 0, 2])
    motion_std[:, :, 0] -= original_root_x
    motion_std[:, :, 2] -= original_root_z

    # --- 3. Axis convention ---
    # In raw HumanML3D coordinates +X points to the character's left, while the benchmark
    # convention is +X = Right, hence the mirroring. A model that already emits +X = Right
    # must pass mirror_x=False, or left and right end up swapped throughout.
    if mirror_x:
        motion_std[:, :, 0] *= -1.0

    metadata = {
        "Original_Floor_Y": original_floor,
        "Original_Root_X": original_root_x,
        "Original_Root_Z": original_root_z,
        "New_Floor_Y": float(motion_std[:, :, 1].min()),
        "New_Root_X": float(motion_std[0, 0, 0]),
        "New_Root_Z": float(motion_std[0, 0, 2]),
        "Mirrored_X": bool(mirror_x),
        "X_Convention": "+X=Right, -X=Left",
        "Z_Convention": "+Z=Forward, -Z=Backward",
        "Y_Convention": "+Y=Up",
    }
    return motion_std, metadata


def estimate_hml3d_initial_forward(motion, n_frames: int = 5):
    """
    Estimate the initial facing direction.

    Build an "across" vector from the hip and shoulder left-right lines, then cross it with
    the up vector to obtain "forward" — the same rule as the HumanML3D preprocessing.

    Call this on the RAW motion, not the standardised one: standardisation mirrors X, which
    flips the sign of the resulting angle.

    Returns (forward_vector | None, angle_in_degrees_relative_to_+Z | None).
    """
    n = min(n_frames, len(motion))

    # HumanML3D joint indices.
    R_HIP, L_HIP = 2, 1
    R_SHOULDER, L_SHOULDER = 17, 16

    across_hip = motion[:n, R_HIP] - motion[:n, L_HIP]
    across_shoulder = motion[:n, R_SHOULDER] - motion[:n, L_SHOULDER]

    # Average over the first frames to reduce single-frame jitter.
    across = (across_hip + across_shoulder).mean(axis=0)
    across[1] = 0.0  # project onto the horizontal plane

    norm = np.linalg.norm(across)
    if norm < 1e-8:
        return None, None
    across /= norm

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = np.cross(up, across)
    forward[1] = 0.0
    forward /= (np.linalg.norm(forward) + 1e-8)

    # Angle relative to +Z; 0 degrees means facing +Z.
    angle_deg = float(np.degrees(np.arctan2(forward[0], forward[2])))
    return forward, angle_deg


def standardise_batch(file_manifest: pd.DataFrame, out_dir, facing_tolerance_deg: float = 15.0,
                      mirror_x: bool = True, n_joints: int = 22,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Standardise every located clip and save it as <out_dir>/<Prompt_ID>.npy.

    Rows without a file are skipped and reported. Rows flagged AMBIGUOUS are still
    processed (the newest candidate was already picked) but produce a warning — dropping
    them, as the original notebook did, is what turned a mixed-length folder into a silent
    "0 standardised" result.

    Raises RuntimeError when nothing could be standardised, so the failure surfaces here
    instead of in the rendering step.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = [], []

    for _, row in file_manifest.iterrows():
        if not row["Joint_File"]:
            skipped.append((row["Prompt_ID"], row["Status"]))
            continue
        if row["Status"] != "OK" and verbose:
            print(f"Warning: {row['Prompt_ID']} status={row['Status']}, "
                  f"using {Path(row['Joint_File']).name}")

        joint_file = Path(row["Joint_File"])
        motion = np.load(joint_file)

        motion_std, metadata = standardise_motionhiflow(motion, mirror_x=mirror_x,
                                                        n_joints=n_joints)
        forward, angle_deg = estimate_hml3d_initial_forward(motion)

        if angle_deg is None:
            facing_status = "UNABLE_TO_CHECK"
        elif abs(angle_deg) <= facing_tolerance_deg:
            facing_status = "OK_+Z"          # roughly facing +Z
        else:
            facing_status = "CHECK_FACING"   # needs manual inspection

        output_npy = out_dir / f"{row['Prompt_ID']}.npy"
        np.save(output_npy, motion_std)

        records.append({
            "Prompt_ID": row["Prompt_ID"],
            "Prompt": row["Prompt"],
            "Raw_Joint_File": str(joint_file),
            "Standard_Joint_File": str(output_npy),
            "Frames": len(motion_std),
            "Ground_Y": metadata["New_Floor_Y"],
            "Initial_Root_X": metadata["New_Root_X"],
            "Initial_Root_Z": metadata["New_Root_Z"],
            "Initial_Forward_X": forward[0] if forward is not None else np.nan,
            "Initial_Forward_Z": forward[2] if forward is not None else np.nan,
            "Facing_Angle_Deg": angle_deg,
            "Facing_Status": facing_status,
        })

    standard_report = pd.DataFrame(records)

    if verbose:
        print("\nTotal standardised:", len(standard_report))
        if skipped:
            print("Skipped:", skipped)

    if standard_report.empty:
        raise RuntimeError(
            "Nothing was standardised. Inspect file_manifest['Status'].\n"
            "  NOT_FOUND : re-run the generation step\n"
            "  AMBIGUOUS : regenerate under a fresh run tag"
        )

    return standard_report


def check_left_right_convention(motion_std, expect: str = "left",
                                min_displacement: float = 0.3):
    """
    Verify the left/right axis convention on a clip whose prompt names a direction.

    This is the one setting nothing can infer on its own. `estimate_hml3d_initial_forward`
    only measures the angle to +Z, which a left-right flip leaves unchanged, so a wrong
    `mirror_x` produces a perfectly plausible-looking result with left and right swapped.

    Run it on a STANDARDISED clip generated from a prompt like "A person walks to the
    left.": under the benchmark convention (+X = Right) that clip must move toward -X.

    Args:
        motion_std       : (T, J, 3) standardised motion
        expect           : "left" or "right", taken from the prompt
        min_displacement : metres of root travel required before the result is meaningful

    Returns (ok, dx) where dx is the root's X displacement. `ok` is None when the clip
    barely moves, in which case the test says nothing — pick a clip that walks further.
    """
    if expect not in ("left", "right"):
        raise ValueError('expect must be "left" or "right"')

    root = np.asarray(motion_std)[:, 0, :]
    dx = float(root[-1, 0] - root[0, 0])

    if abs(dx) < min_displacement:
        return None, dx  # too little lateral travel to conclude anything

    moved = "right" if dx > 0 else "left"
    return moved == expect, dx


def check_standardisation(standard_report: pd.DataFrame) -> None:
    """Print a sanity check. All three maxima should be ~0 if the alignment applied."""
    print("Facing status:")
    print(standard_report["Facing_Status"].value_counts())
    print()
    print("max |initial root X| :", standard_report["Initial_Root_X"].abs().max())
    print("max |initial root Z| :", standard_report["Initial_Root_Z"].abs().max())
    print("max |ground Y|       :", standard_report["Ground_Y"].abs().max())


# ======================================================================================
# Rendering
# ======================================================================================

# Kinematic chains of the HumanML3D 22-joint skeleton.
HML3D_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],      # right leg
    [0, 1, 4, 7, 10],      # left leg
    [0, 3, 6, 9, 12, 15],  # spine and head
    [9, 14, 17, 19, 21],   # right arm
    [9, 13, 16, 18, 20],   # left arm
]


def get_scene_limits(motion, margin: float = 1.0, min_span: float = 3.0):
    """
    Derive the scene extent from the root trajectory.

    margin   : padding around the trajectory, in metres
    min_span : minimum visible span, so in-place motions still get enough room
    """
    root = motion[:, 0, :]

    x_min = float(root[:, 0].min()) - margin
    x_max = float(root[:, 0].max()) + margin
    z_min = float(root[:, 2].min()) - margin
    z_max = float(root[:, 2].max()) + margin

    # Widen the view for small-displacement motions such as an in-place jump.
    if x_max - x_min < min_span:
        centre = (x_min + x_max) / 2
        x_min, x_max = centre - min_span / 2, centre + min_span / 2
    if z_max - z_min < min_span:
        centre = (z_min + z_max) / 2
        z_min, z_max = centre - min_span / 2, centre + min_span / 2

    # The vertical range comes from the whole skeleton, since raised hands exceed head height.
    y_min = 0.0
    y_max = float(motion[:, :, 1].max()) + 0.25

    return x_min, x_max, z_min, z_max, y_min, y_max


def render_standard_gif(motion, output_file, prompt_id: str, prompt: str,
                        fps: int = 20, elev: float = DEFAULT_ELEV,
                        azim: float = DEFAULT_AZIM, verbose: bool = True) -> Path:
    """
    Render one standardised motion clip to a GIF, with the axis directions drawn in.

    Note the axis mapping: matplotlib's 3D axes do not map one-to-one onto the motion axes.
    The motion's Y (height) is plotted on matplotlib's Z axis (vertical).

    The axis arrows and labels live in data space, so they rotate with the camera and stay
    truthful at any (elev, azim) — only the viewpoint changes, never the data.

    matplotlib is imported here rather than at module level so that importing this module
    never depends on a display backend being configured.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    x_min, x_max, z_min, z_max, y_min, y_max = get_scene_limits(motion)

    if verbose:
        print(f"{prompt_id}: X=[{x_min:.2f}, {x_max:.2f}], "
              f"Z=[{z_min:.2f}, {z_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Compute the ground grid once, outside the frame loop.
    GX, GZ = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
    GY = np.zeros_like(GX)

    L = 0.8  # length of the axis arrows
    AXES = [
        (( L, 0, 0), "+X RIGHT"),
        ((-L, 0, 0), "-X LEFT"),
        ((0,  L, 0), "+Z FORWARD"),
        ((0, -L, 0), "-Z BACKWARD"),
        ((0, 0,  L), "+Y UP"),
    ]

    def update(frame):
        ax.cla()
        joints = motion[frame]

        # Stick figure. Note the argument order: motion X -> plot X, motion Z -> plot Y,
        # motion Y -> plot Z.
        for chain in HML3D_KINEMATIC_CHAIN:
            p = joints[chain]
            ax.plot(p[:, 0], p[:, 2], p[:, 1], linewidth=2)
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=10)

        # Root trajectory projected onto the ground.
        root = motion[:frame + 1, 0, :]
        ax.plot(root[:, 0], root[:, 2], np.zeros(len(root)), linestyle="--", linewidth=1.5)

        ax.plot_wireframe(GX, GZ, GY, linewidth=0.3, alpha=0.35)

        for (dx, dy, dz), label in AXES:
            ax.quiver(0, 0, 0, dx, dy, dz, arrow_length_ratio=0.12)
            ax.text(dx, dy, dz, label)

        # Fixed extent and camera, so clips stay visually comparable.
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_zlim(y_min, y_max)
        ax.set_box_aspect((1, 1, 0.7))
        ax.view_init(elev=elev, azim=azim)

        ax.set_xlabel("X (+Right / -Left)")
        ax.set_ylabel("Z (+Forward / -Backward)")
        ax.set_zlabel("Y (+Up)")
        ax.grid(True)
        ax.set_title(f"{prompt_id}: {prompt}\nFrame {frame + 1}/{len(motion)}")

    animation = FuncAnimation(fig, update, frames=len(motion), interval=1000 / fps)
    animation.save(str(output_file), writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)  # must close, otherwise batch rendering leaks memory

    if verbose:
        print("Saved:", output_file)
    return output_file


def render_batch(standard_report: pd.DataFrame, gif_dir, fps: int = 20,
                 elev: float = DEFAULT_ELEV, azim: float = DEFAULT_AZIM,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Render every standardised clip and return a GIF report.

    The original notebook referenced a `gif_report` variable that was never created; this
    function is what produces it.
    """
    gif_dir = Path(gif_dir)
    gif_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in standard_report.iterrows():
        motion = np.load(row["Standard_Joint_File"])
        gif_file = gif_dir / f"{row['Prompt_ID']}.gif"

        if verbose:
            print(f"\nRendering {row['Prompt_ID']} ...")
        render_standard_gif(motion, gif_file, row["Prompt_ID"], row["Prompt"],
                            fps=fps, elev=elev, azim=azim, verbose=verbose)

        records.append({
            "Prompt_ID": row["Prompt_ID"],
            "GIF_File": str(gif_file),
            "GIF_Frames": len(motion),
        })

    return pd.DataFrame(records)


def package_gifs(gif_dir, zip_base) -> str:
    """Zip a GIF directory and return the path of the archive."""
    return shutil.make_archive(str(zip_base), "zip", root_dir=str(gif_dir))


# ======================================================================================
# Matching score
# ======================================================================================

# Length constraints of the official evaluator.
EVAL_MAX_FRAMES = 196   # hard cap imposed by the evaluator's architecture
EVAL_MIN_FRAMES = 40    # below this the score is unreliable (empirical)
MAX_TEXT_LEN = 20       # maximum HumanML3D token count


class MatchingScorer:
    """
    Per-prompt text-motion matching score, using the official HumanML3D evaluator.

    Lower is better. This is a diagnostic for individual prompts; it is not a substitute
    for the official full evaluation (`run.sh eval`), which reports FID / R-Precision.

    Important: a matching score's absolute value shifts systematically with motion length,
    so scores are only comparable across prompts at the SAME length. Never read a
    100-frame score against a 196-frame one.

    torch, spacy and the repository's evaluator are imported in __init__ rather than at
    module level, so this module stays importable in an environment that only needs the
    standardisation and rendering helpers.
    """

    def __init__(self, repo_root="/content/MotionHiFlow", device=None,
                 spacy_model="en_core_web_sm", feature_dim: int = 263,
                 evaluator_import="src.evaluators.t2m_eval_wrapper:EvaluatorModelWrapper",
                 vectorizer_import="src.utils.word_vectorizer:WordVectorizer",
                 metrics_import="src.utils.metrics:euclidean_distance_matrix",
                 evaluators_subdir="deps/evaluators", glove_subdir="deps/glove",
                 meta_subdir="deps/evaluators/t2m/Comp_v6_KLD005/meta"):
        """
        Args:
            repo_root         : the model repository, added to sys.path so its modules import
            feature_dim       : expected motion feature width (263 for HumanML3D)
            *_import          : "module.path:attribute" locations of the evaluator pieces.
                                The evaluator itself is the shared official HumanML3D one,
                                used by MDM, MLD, T2M-GPT and others — only where a given
                                repository stores it changes, so these are arguments.
            *_subdir          : where the checkpoints, GloVe vectors and normalisation
                                statistics live, relative to repo_root
        """
        import importlib
        import torch
        import spacy
        from types import SimpleNamespace

        # The repository must be importable for its evaluator modules to resolve.
        import sys
        repo_root = Path(repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        def _load(spec: str):
            """Resolve a 'module.path:attribute' string."""
            module_name, _, attr = spec.partition(":")
            return getattr(importlib.import_module(module_name), attr)

        EvaluatorModelWrapper = _load(evaluator_import)
        WordVectorizer = _load(vectorizer_import)
        euclidean_distance_matrix = _load(metrics_import)

        self.torch = torch
        self._euclidean_distance_matrix = euclidean_distance_matrix
        self.repo_root = repo_root
        self.feature_dim = int(feature_dim)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            self.nlp = spacy.load(spacy_model)
        except OSError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                f"spaCy model '{spacy_model}' is missing. "
                f"Run: python -m spacy download {spacy_model}"
            ) from exc

        # These hyper-parameters must match the official evaluator's training setup,
        # otherwise the scores are no longer comparable with published numbers.
        opt = SimpleNamespace(
            checkpoints_dir=str(repo_root / evaluators_subdir),
            dataset_name="t2m",
            dim_movement_enc_hidden=512,
            dim_movement_latent=512,
            unit_length=4,
            device=self.device,
        )
        self.eval_wrapper = EvaluatorModelWrapper(opt)
        self.w_vectorizer = WordVectorizer(str(repo_root / glove_subdir), "our_vab")

        # The evaluator has its own normalisation statistics — not the training mean/std.
        meta = repo_root / meta_subdir
        self.eval_mean = np.load(meta / "mean.npy")
        self.eval_std = np.load(meta / "std.npy")

    # ---------------------------------------------------------------- text preprocessing
    def prepare_text(self, prompt: str):
        """Convert a prompt into evaluator inputs, following the HumanML3D rules."""
        torch = self.torch
        doc = self.nlp(prompt.replace("-", ""))

        tokens = []
        for token in doc:
            word = token.text
            if not word.isalpha():          # drop punctuation
                continue
            # Lemmatise nouns and verbs, except "left": it is both a direction and the past
            # tense of "leave", and lemmatising would destroy the directional meaning.
            if token.pos_ in ["NOUN", "VERB"] and word != "left":
                word = token.lemma_
            tokens.append(f"{word}/{token.pos_}")

        # Pad to a fixed length: sos + tokens + eos, then pad with unk.
        if len(tokens) < MAX_TEXT_LEN:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens += ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
        else:
            tokens = ["sos/OTHER"] + tokens[:MAX_TEXT_LEN] + ["eos/OTHER"]
            sent_len = len(tokens)

        word_embeddings, pos_one_hots = [], []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            word_embeddings.append(word_emb)
            pos_one_hots.append(pos_oh)

        # unsqueeze(0) adds the batch dimension.
        word_embeddings = torch.tensor(np.array(word_embeddings), dtype=torch.float32).unsqueeze(0)
        pos_one_hots = torch.tensor(np.array(pos_one_hots), dtype=torch.float32).unsqueeze(0)
        sent_len = torch.LongTensor([sent_len])
        return word_embeddings, pos_one_hots, sent_len

    # ------------------------------------------------------------------------- scoring
    def score(self, prompt: str, motion_file, strict: bool = True,
              sample_pattern: str = DEFAULT_SAMPLE_PATTERN):
        """
        Score one prompt against one generated motion.

        motion_file   : a feature file of width self.feature_dim (263 for HumanML3D)
        strict        : True raises above EVAL_MAX_FRAMES; False truncates with a warning
        sample_pattern: naming convention, used to read the length out of the filename

        Returns (score, used_frames).
        """
        torch = self.torch
        word_embeddings, pos_one_hots, sent_len = self.prepare_text(prompt)

        raw_motion = np.load(motion_file)
        if raw_motion.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim}-D HumanML3D features, got {raw_motion.shape}"
            )

        # Trust the array's real frame count; the filename is only a cross-check, and an
        # unfamiliar naming convention must not break scoring.
        parsed = parse_motion_length(motion_file, sample_pattern, strict=False)
        m_length = min(parsed, len(raw_motion)) if parsed else len(raw_motion)

        # The original implementation truncated silently here: everything beyond 196 frames
        # was dropped without any notice, and the resulting score looked perfectly normal.
        if m_length > EVAL_MAX_FRAMES:
            msg = (f"Motion has {m_length} frames, above the evaluator cap of "
                   f"{EVAL_MAX_FRAMES}; only the first {EVAL_MAX_FRAMES} would be scored.")
            if strict:
                raise ValueError(msg + " Pass strict=False to allow truncation.")
            print("Warning:", msg)

        if m_length < EVAL_MIN_FRAMES:
            print(f"Warning: motion has only {m_length} frames "
                  f"(< {EVAL_MIN_FRAMES}) — the score may be unreliable.")

        motion_norm = (raw_motion.astype(np.float32) - self.eval_mean) / self.eval_std

        # The evaluator takes a fixed 196 frames, zero-padded.
        motion_input = np.zeros((EVAL_MAX_FRAMES, self.feature_dim), dtype=np.float32)
        used_frames = min(m_length, EVAL_MAX_FRAMES)
        motion_input[:used_frames] = motion_norm[:used_frames]

        motion_tensor = torch.tensor(motion_input, dtype=torch.float32).unsqueeze(0)
        m_length_tensor = torch.LongTensor([used_frames])

        with torch.no_grad():
            text_embedding, motion_embedding = self.eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, motion_tensor, m_length_tensor
            )

        dist_matrix = self._euclidean_distance_matrix(
            text_embedding.cpu().numpy(), motion_embedding.cpu().numpy()
        )
        return float(dist_matrix[0, 0]), used_frames

    def score_batch(self, manifest: pd.DataFrame, anim_root, expected_length,
                    sample_pattern: str = DEFAULT_SAMPLE_PATTERN,
                    feature_suffix: str = "_data",
                    strict: bool = True, verbose: bool = True) -> pd.DataFrame:
        """
        Score every prompt in the manifest and return a results table.

        Pass cfg.sample_pattern / cfg.feature_suffix when the repository names its output
        files differently from MotionHiFlow.
        """
        records = []
        for i, row in manifest.iterrows():
            data_file, status = find_sample_file(
                anim_root, i, row["Prompt"], kind="data", expected_length=expected_length,
                sample_pattern=sample_pattern, feature_suffix=feature_suffix,
            )
            if status != "OK":
                if verbose:
                    print(f"[{status}] {row['Prompt_ID']}: feature file not found")
                continue

            score, used_frames = self.score(row["Prompt"], data_file, strict=strict,
                                            sample_pattern=sample_pattern)

            records.append({
                "Prompt_ID": row["Prompt_ID"],
                "Capability": row.get("Capability"),
                "Prompt": row["Prompt"],
                "Motion_File": Path(data_file).name,
                # Record the frames that actually contributed, so it stays clear at which
                # length a score was computed.
                "Scored_Frames": used_frames,
                "Matching_Score": round(score, 4),
            })
            if verbose:
                print(f"{row['Prompt_ID']} | {score:.4f} | {used_frames:>3}f | {row['Prompt']}")

        df_scores = pd.DataFrame(records)

        if verbose and not df_scores.empty:
            lengths = sorted(df_scores["Scored_Frames"].unique())
            print(f"\nScored at: {lengths} frames")
            if len(lengths) > 1:
                print("Warning: mixed lengths in one table — these scores are not comparable.")
            print("Reminder: do not compare these against scores from another motion_length.")

        return df_scores
