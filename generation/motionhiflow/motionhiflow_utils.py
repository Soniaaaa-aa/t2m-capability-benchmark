"""
motionhiflow_utils.py

Helper functions for the MotionHiFlow side of the benchmark pipeline: run config,
finding the files gen_t2m.py wrote, standardising them to the shared contract,
rendering the check GIFs, and the per-prompt matching score.

Generation itself still goes through MotionHiFlow's own run.sh / gen_t2m.py — this
module only prepares inputs for it and processes what comes out, so we don't have to
touch anything inside the cloned repo.

torch/spacy/the repo's evaluator are only imported inside MatchingScorer, so the rest
of this file (standardisation, rendering) works without them installed.

Quick usage:
    from motionhiflow_utils import RunConfig
    cfg = RunConfig(motion_length=100).make_dirs()
    file_manifest = cfg.build_file_manifest(manifest)
    standard_report = cfg.standardise(file_manifest)
    gif_report = cfg.render(standard_report)

If we ever reuse this for another model: the standardisation/rendering code only
assumes HumanML3D-layout output, (T, 22, 3) joints and (T, 263) features. What
changes between repos is naming and axis convention, so those are constructor args:

    cfg = RunConfig(
        motion_length=100,
        root=Path("/content/OtherModel"),
        experiment_name="my_experiment",
        sample_pattern="sample{index}_r0_len{length}",
        mirror_x=True,   # False if that model already outputs +X = Right
    )

mirror_x can't be detected automatically — the facing check only looks at the angle
to +Z, which doesn't change under a left/right flip. Check it manually with
check_left_right_convention() on a clip whose prompt says "walks to the left".
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

# filename stem gen_t2m.py writes; override in RunConfig for a different repo
DEFAULT_SAMPLE_PATTERN = "sample{index}_r0_len{length}"

# camera for the GIFs — negative elevation looks from below and makes left/right
# confusing to eyeball, so keep this positive
DEFAULT_ELEV = 30
DEFAULT_AZIM = -45


# matches the {length} field, with or without a format spec
_LENGTH_FIELD_RE = re.compile(r"\{length(?::[^}]*)?\}")


def _format_stem(sample_pattern: str, index: int, length: int) -> str:
    return sample_pattern.format(index=index, length=int(length))


def _glob_stem(sample_pattern: str, index: int) -> str:
    # swap the length field for a literal * before formatting -- can't .format() a
    # string into "{length:04d}"
    return _LENGTH_FIELD_RE.sub("*", sample_pattern).format(index=index)


# ======================================================================================
# Run configuration
# ======================================================================================

@dataclass
class RunConfig:
    """
    Parameters + derived paths for one run. Only set the first few fields by hand,
    the rest is computed in __post_init__.

    run_tag bakes in the frame length, so two runs at different lengths never land
    in the same folder (otherwise a folder ends up with both sample0_r0_len196.npy
    and sample0_r0_len100.npy and every lookup after that is ambiguous).
    """

    motion_length: int = 100
    run_name: str = "pilot13"
    run_version: str = "v1"
    root: Path = Path("/content/MotionHiFlow")

    # repo layout -- change if porting to another model
    experiment_name: str = "t2m_tmdit_16d"
    joints_subdir: str = "joints"
    anim_subdir: str = "animations"
    sample_pattern: str = DEFAULT_SAMPLE_PATTERN
    feature_suffix: str = "_data"
    exclude_suffixes: tuple = ("_ik",)   # e.g. the IK reconstructions MHF also writes
    length_multiple: int = 4             # model rounds requested length to this

    # data convention
    n_joints: int = 22
    feature_dim: int = 263
    mirror_x: bool = True   # raw output has +X = left, flip it to +X = Right

    # derived, don't set these
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

        step = max(1, int(self.length_multiple))
        self.effective_length = (int(self.motion_length) // step) * step
        if self.effective_length < step:
            raise ValueError(
                f"motion_length is too small: {self.motion_length} "
                f"(must be at least {step})"
            )

        self.run_tag = f"{self.run_name}_len{self.effective_length}_{self.run_version}"

        self.manifest_csv = self.root / f"{self.run_tag}_manifest.csv"
        self.prompt_txt = self.root / f"{self.run_tag}_prompts.txt"
        self.run_output_root = self.root / "benchmark_runs" / self.run_tag
        self.result_root = self.run_output_root / self.experiment_name
        self.joint_root = self.result_root / self.joints_subdir
        self.anim_root = self.result_root / self.anim_subdir

        self.standard_root = self.root / "standardised" / self.run_tag
        self.std_joint_root = self.standard_root / "joints"
        self.std_gif_root = self.standard_root / "gifs"
        self.report_file = self.standard_root / "standardisation_report.csv"
        self.score_csv = self.standard_root / "matching_scores.csv"

    def make_dirs(self) -> "RunConfig":
        for d in (self.run_output_root, self.std_joint_root, self.std_gif_root):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def summary(self) -> str:
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
        if self.joint_root.exists() and any(self.joint_root.rglob("*.npy")):
            lines += [
                "",
                "Note: this directory already holds results; re-running generation overwrites them.",
                "      Bump run_version to 'v2' if you want to keep them.",
            ]
        return "\n".join(lines)

    # thin wrappers so a notebook cell is one line; the real functions below still
    # work standalone if called directly

    def find(self, index: int, prompt: str, kind: str = "joints"):
        root = self.anim_root if kind == "data" else self.joint_root
        return find_sample_file(
            root, index, prompt, kind=kind, expected_length=self.effective_length,
            sample_pattern=self.sample_pattern, feature_suffix=self.feature_suffix,
            exclude_suffixes=self.exclude_suffixes,
        )

    def build_file_manifest(self, manifest, verbose: bool = True):
        return build_file_manifest(
            manifest, self.joint_root, self.effective_length,
            sample_pattern=self.sample_pattern, feature_suffix=self.feature_suffix,
            exclude_suffixes=self.exclude_suffixes, verbose=verbose,
        )

    def standardise(self, file_manifest, verbose: bool = True):
        return standardise_batch(
            file_manifest, self.std_joint_root,
            mirror_x=self.mirror_x, n_joints=self.n_joints, verbose=verbose,
        )

    def render(self, standard_report, fps: int = 20, elev: float = DEFAULT_ELEV,
               azim: float = DEFAULT_AZIM, verbose: bool = True):
        return render_batch(standard_report, self.std_gif_root, fps=fps,
                            elev=elev, azim=azim, verbose=verbose)

    def score(self, scorer, manifest, strict: bool = True, verbose: bool = True):
        return scorer.score_batch(
            manifest, self.anim_root, self.effective_length,
            sample_pattern=self.sample_pattern, feature_suffix=self.feature_suffix,
            strict=strict, verbose=verbose,
        )


# ======================================================================================
# Output-file lookup
# ======================================================================================

def mhf_folder_name(prompt: str) -> str:
    """Output sub-folder name for a prompt -- same rule gen_t2m.py uses."""
    return prompt[:100].replace("/", "_").replace(" ", "_")


# matches {index} or {length}, with or without a format spec
_ANY_FIELD_RE = re.compile(r"\{(index|length)(?::[^}]*)?\}")


def _length_regex(sample_pattern: str) -> "re.Pattern":
    # escape the literal parts and turn the placeholders into digit groups, so this
    # also works for a custom pattern, not just the MotionHiFlow default
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
    Pull the frame count back out of a generated filename.

    strict=False returns None instead of raising, for callers that can fall back to
    the array's own length instead.
    """
    name = Path(path).name
    m = _length_regex(sample_pattern).search(name)
    if m is None:
        # fall back to the plain "_len<digits>" spelling before giving up
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
    Find the file generated for one prompt.

    kind="joints" -> plain position file, kind="data" -> feature file (stem +
    feature_suffix). expected_length, if given, is tried as an exact match first --
    matters because if several lengths were generated under the same run_tag, the
    wildcard match below picks up more than one file and the lookup is flagged
    AMBIGUOUS instead of silently returning the wrong one.

    Returns (Path | None, status) with status "OK" / "NOT_FOUND" / "AMBIGUOUS".
    """
    suffix = feature_suffix if kind == "data" else ""
    folder = Path(root) / mhf_folder_name(prompt)
    if not folder.exists():
        return None, "NOT_FOUND"

    if expected_length is not None:
        exact = folder / f"{_format_stem(sample_pattern, index, expected_length)}{suffix}.npy"
        if exact.exists():
            return exact, "OK"

    candidates = sorted(folder.glob(f"{_glob_stem(sample_pattern, index)}{suffix}.npy"))
    if kind != "data":
        drop = tuple(f"{x}.npy" for x in exclude_suffixes) + (f"{feature_suffix}.npy",)
        candidates = [p for p in candidates if not p.name.endswith(drop)]

    if len(candidates) == 1:
        return candidates[0], "OK"
    if len(candidates) == 0:
        return None, "NOT_FOUND"
    # more than one match -- take the newest and flag it so it gets a manual look
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest, "AMBIGUOUS"


def list_sample_folder(root, index: int, prompt: str) -> str:
    """Debug helper -- list what's actually in one prompt's output folder."""
    folder = Path(root) / mhf_folder_name(prompt)
    if not folder.exists():
        return f"[folder missing] {folder}"
    names = sorted(p.name for p in folder.iterdir())
    return f"{folder}\n    " + ("\n    ".join(names) if names else "(empty)")


def write_prompt_file(manifest: pd.DataFrame, prompt_txt) -> Path:
    """Write the prompt file run.sh gen ... text_path=... expects: "prompt#frames" per line."""
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

    Prints the folder contents right away for anything that failed, instead of just
    leaving an empty table for us to figure out later.
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
    Standardise one clip to the shared contract: ground at Y=0, first-frame root XZ
    at (0,0), +X = Right.

    MotionHiFlow's raw output has no fixed origin or axis convention on its own, so
    this applies three corrections:
      1. ground  -- shift the whole clip up so its lowest point sits at Y = 0
      2. origin  -- shift so the root at frame 0 is at XZ = (0, 0)
      3. mirror  -- flip X if mirror_x=True (raw HumanML3D has +X pointing left)

    mirror_x needs to be set correctly by hand -- there's no way to detect it from
    the data (see check_left_right_convention below).

    Returns (motion_std, metadata).
    """
    motion = np.asarray(motion, dtype=np.float32)

    if motion.ndim != 3:
        raise ValueError(f"Expected [T, J, 3], got {motion.shape}")
    if motion.shape[1] != n_joints:
        raise ValueError(f"Expected {n_joints} joints, got {motion.shape[1]}")
    if motion.shape[2] != 3:
        raise ValueError(f"Expected XYZ, got {motion.shape}")
    if not np.isfinite(motion).all():
        raise ValueError("Motion contains NaN or Inf.")

    motion_std = motion.copy()

    # ground: use the clip's global min, not just frame 0, so a jump or crouch
    # doesn't end up with the character sinking into the floor
    original_floor = float(motion_std[:, :, 1].min())
    motion_std[:, :, 1] -= original_floor

    # origin -- joint 0 is root/pelvis
    original_root_x = float(motion_std[0, 0, 0])
    original_root_z = float(motion_std[0, 0, 2])
    motion_std[:, :, 0] -= original_root_x
    motion_std[:, :, 2] -= original_root_z

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
    Estimate initial facing direction (same rule the HumanML3D preprocessing uses):
    build an "across" vector from the hip/shoulder left-right lines, cross with up
    to get "forward".

    Use the RAW motion here, not the standardised one -- standardising flips X,
    which flips the sign of the angle.

    Returns (forward_vector | None, angle_deg_relative_to_+Z | None).
    """
    n = min(n_frames, len(motion))

    R_HIP, L_HIP = 2, 1
    R_SHOULDER, L_SHOULDER = 17, 16

    across_hip = motion[:n, R_HIP] - motion[:n, L_HIP]
    across_shoulder = motion[:n, R_SHOULDER] - motion[:n, L_SHOULDER]

    across = (across_hip + across_shoulder).mean(axis=0)
    across[1] = 0.0  # flatten onto the ground plane

    norm = np.linalg.norm(across)
    if norm < 1e-8:
        return None, None
    across /= norm

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = np.cross(up, across)
    forward[1] = 0.0
    forward /= (np.linalg.norm(forward) + 1e-8)

    angle_deg = float(np.degrees(np.arctan2(forward[0], forward[2])))  # 0 deg = facing +Z
    return forward, angle_deg


def standardise_batch(file_manifest: pd.DataFrame, out_dir, facing_tolerance_deg: float = 15.0,
                      mirror_x: bool = True, n_joints: int = 22,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Standardise every file in file_manifest and save as <out_dir>/<Prompt_ID>.npy.

    Rows with no file are skipped (reported at the end). AMBIGUOUS rows still get
    processed using whatever candidate find_sample_file picked, just with a warning
    printed -- dropping them outright is what turned a mixed-length folder into a
    silent "0 standardised" before.
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
            facing_status = "OK_+Z"
        else:
            facing_status = "CHECK_FACING"

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
    Sanity-check the left/right axis on a clip whose prompt names a direction --
    this is the one thing nothing else here can verify on its own, since the facing
    check above doesn't change under a left/right flip.

    Run on a STANDARDISED clip generated from something like "A person walks to the
    left." Under our convention (+X = Right) that clip should move toward -X.

    Returns (ok, dx); ok is None if the clip barely moves (pick a different clip --
    not enough travel to say anything).
    """
    if expect not in ("left", "right"):
        raise ValueError('expect must be "left" or "right"')

    root = np.asarray(motion_std)[:, 0, :]
    dx = float(root[-1, 0] - root[0, 0])

    if abs(dx) < min_displacement:
        return None, dx

    moved = "right" if dx > 0 else "left"
    return moved == expect, dx


def check_standardisation(standard_report: pd.DataFrame) -> None:
    """Print a quick sanity check -- all three maxima should be ~0 after standardising."""
    print("Facing status:")
    print(standard_report["Facing_Status"].value_counts())
    print()
    print("max |initial root X| :", standard_report["Initial_Root_X"].abs().max())
    print("max |initial root Z| :", standard_report["Initial_Root_Z"].abs().max())
    print("max |ground Y|       :", standard_report["Ground_Y"].abs().max())


# ======================================================================================
# Rendering
# ======================================================================================

HML3D_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],      # right leg
    [0, 1, 4, 7, 10],      # left leg
    [0, 3, 6, 9, 12, 15],  # spine + head
    [9, 14, 17, 19, 21],   # right arm
    [9, 13, 16, 18, 20],   # left arm
]


def get_scene_limits(motion, margin: float = 1.0, min_span: float = 3.0):
    """Bounding box for the camera, from the root trajectory, padded by margin."""
    root = motion[:, 0, :]

    x_min = float(root[:, 0].min()) - margin
    x_max = float(root[:, 0].max()) + margin
    z_min = float(root[:, 2].min()) - margin
    z_max = float(root[:, 2].max()) + margin

    # widen for near-stationary clips (e.g. an in-place jump) so there's room to see it
    if x_max - x_min < min_span:
        centre = (x_min + x_max) / 2
        x_min, x_max = centre - min_span / 2, centre + min_span / 2
    if z_max - z_min < min_span:
        centre = (z_min + z_max) / 2
        z_min, z_max = centre - min_span / 2, centre + min_span / 2

    y_min = 0.0
    y_max = float(motion[:, :, 1].max()) + 0.25  # whole skeleton, raised hands go above head

    return x_min, x_max, z_min, z_max, y_min, y_max


def render_standard_gif(motion, output_file, prompt_id: str, prompt: str,
                        fps: int = 20, elev: float = DEFAULT_ELEV,
                        azim: float = DEFAULT_AZIM, verbose: bool = True) -> Path:
    """
    Render one standardised clip to a GIF with axis arrows drawn in for reference.

    matplotlib's 3D axes don't map 1:1 onto ours -- motion Y (height) goes on
    matplotlib's Z. Imported inside the function so this module doesn't need a
    display backend just to be imported.
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

    GX, GZ = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
    GY = np.zeros_like(GX)

    L = 0.8
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

        # note the swap: motion X -> plot X, motion Z -> plot Y, motion Y -> plot Z
        for chain in HML3D_KINEMATIC_CHAIN:
            p = joints[chain]
            ax.plot(p[:, 0], p[:, 2], p[:, 1], linewidth=2)
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], s=10)

        root = motion[:frame + 1, 0, :]
        ax.plot(root[:, 0], root[:, 2], np.zeros(len(root)), linestyle="--", linewidth=1.5)

        ax.plot_wireframe(GX, GZ, GY, linewidth=0.3, alpha=0.35)

        for (dx, dy, dz), label in AXES:
            ax.quiver(0, 0, 0, dx, dy, dz, arrow_length_ratio=0.12)
            ax.text(dx, dy, dz, label)

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
    plt.close(fig)  # otherwise rendering a batch leaks memory

    if verbose:
        print("Saved:", output_file)
    return output_file


def render_batch(standard_report: pd.DataFrame, gif_dir, fps: int = 20,
                 elev: float = DEFAULT_ELEV, azim: float = DEFAULT_AZIM,
                 verbose: bool = True) -> pd.DataFrame:
    """Render every standardised clip in standard_report and return a GIF report table."""
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
    """Zip up a GIF directory, return the archive path."""
    return shutil.make_archive(str(zip_base), "zip", root_dir=str(gif_dir))


# ======================================================================================
# Matching score
# ======================================================================================

EVAL_MAX_FRAMES = 196   # hard cap from the evaluator's architecture
EVAL_MIN_FRAMES = 40    # below this the score gets noisy (empirical, not a hard rule)
MAX_TEXT_LEN = 20       # max HumanML3D token count


class MatchingScorer:
    """
    Per-prompt text-motion matching score using the official HumanML3D evaluator.
    Lower = better match. This is just a per-prompt diagnostic, not a replacement
    for the repo's own full eval (run.sh eval, which gives FID / R-Precision).

    Scores shift with motion length, so only compare scores computed at the same
    length -- don't put a 100-frame score next to a 196-frame one.

    torch/spacy/the evaluator are imported in __init__, not at the top of the file,
    so the rest of this module still works without them installed.
    """

    def __init__(self, repo_root="/content/MotionHiFlow", device=None,
                 spacy_model="en_core_web_sm", feature_dim: int = 263,
                 evaluator_import="src.evaluators.t2m_eval_wrapper:EvaluatorModelWrapper",
                 vectorizer_import="src.utils.word_vectorizer:WordVectorizer",
                 metrics_import="src.utils.metrics:euclidean_distance_matrix",
                 evaluators_subdir="deps/evaluators", glove_subdir="deps/glove",
                 meta_subdir="deps/evaluators/t2m/Comp_v6_KLD005/meta"):
        # the *_import args point at the standard HumanML3D evaluator pieces (the
        # same one MDM/MLD/T2M-GPT etc. use) -- only its location in the repo
        # differs, hence these being arguments instead of hardcoded
        import importlib
        import torch
        import spacy
        from types import SimpleNamespace

        import sys
        repo_root = Path(repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        def _load(spec: str):
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
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{spacy_model}' is missing. "
                f"Run: python -m spacy download {spacy_model}"
            ) from exc

        # these have to match the evaluator's own training setup or the scores
        # aren't comparable to the published numbers
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

        # evaluator has its own normalisation stats, not the training mean/std
        meta = repo_root / meta_subdir
        self.eval_mean = np.load(meta / "mean.npy")
        self.eval_std = np.load(meta / "std.npy")

    def prepare_text(self, prompt: str):
        """Turn a prompt into evaluator input, following the HumanML3D preprocessing rules."""
        torch = self.torch
        doc = self.nlp(prompt.replace("-", ""))

        tokens = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            # lemmatise nouns/verbs, except "left" -- it's a direction, not just the
            # past tense of "leave", and lemmatising would wreck the meaning
            if token.pos_ in ["NOUN", "VERB"] and word != "left":
                word = token.lemma_
            tokens.append(f"{word}/{token.pos_}")

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

        word_embeddings = torch.tensor(np.array(word_embeddings), dtype=torch.float32).unsqueeze(0)
        pos_one_hots = torch.tensor(np.array(pos_one_hots), dtype=torch.float32).unsqueeze(0)
        sent_len = torch.LongTensor([sent_len])
        return word_embeddings, pos_one_hots, sent_len

    def score(self, prompt: str, motion_file, strict: bool = True,
              sample_pattern: str = DEFAULT_SAMPLE_PATTERN):
        """
        Score one prompt against one generated motion. motion_file should be a
        feature file (self.feature_dim wide, 263 for HumanML3D). strict=False
        truncates clips over EVAL_MAX_FRAMES instead of raising.

        Returns (score, used_frames).
        """
        torch = self.torch
        word_embeddings, pos_one_hots, sent_len = self.prepare_text(prompt)

        raw_motion = np.load(motion_file)
        if raw_motion.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim}-D HumanML3D features, got {raw_motion.shape}"
            )

        # trust the array's actual frame count over the filename -- the filename is
        # only a cross-check, shouldn't break scoring if the naming is unfamiliar
        parsed = parse_motion_length(motion_file, sample_pattern, strict=False)
        m_length = min(parsed, len(raw_motion)) if parsed else len(raw_motion)

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
        """Score every prompt in manifest, return a results table."""
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
