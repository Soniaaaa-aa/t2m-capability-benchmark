"""
eval_vlm.py — VLM fallback evaluator for unresolved requirements.

This evaluator is NOT a replacement for any primary evaluator.
It is called only after the normal rule/geometry evaluator returns pass_fail=None.

Protocol:
- standardised motion [T,22,3]
- 32 uniformly sampled frames
- 4x8 temporal strip
- one atomic requirement per VLM call
- OpenAI Responses API
- PASS / FAIL only
"""

from __future__ import annotations

import os
import json
import base64
import tempfile
from pathlib import Path

import numpy as np

from common import BaseEvaluator, validate_standardised_motion, requirement_value, register_evaluator


class VLMEvaluator(BaseEvaluator):
    EVALUATOR_NAME = "VLMEvaluator"

    DEFAULT_MODEL = "gpt-5.4"
    DEFAULT_N_FRAMES = 32
    DEFAULT_DETAIL = "high"

    # HumanML3D 22-joint skeleton chains.
    KINEMATIC_CHAINS = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]

    def __init__(self, model=None, n_frames=32, detail="high", client=None):
        self.model = model or os.getenv("T2M_VLM_MODEL", self.DEFAULT_MODEL)
        self.n_frames = int(n_frames)
        self.detail = detail

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "VLMEvaluator requires the 'openai' package. "
                    "Install it with: pip install openai"
                ) from e

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. "
                    "Set it in the runtime environment; never commit API keys."
                )
            client = OpenAI()

        self.client = client

    @classmethod
    def for_benchmark(cls):
        return cls(
            model=os.getenv("T2M_VLM_MODEL", cls.DEFAULT_MODEL),
            n_frames=int(os.getenv("T2M_VLM_N_FRAMES", cls.DEFAULT_N_FRAMES)),
            detail=os.getenv("T2M_VLM_DETAIL", cls.DEFAULT_DETAIL),
        )

    @staticmethod
    def _sample_indices(n_total, n_frames):
        if n_total < 1:
            raise ValueError("Motion contains no frames.")
        n = min(int(n_frames), int(n_total))
        return np.linspace(0, n_total - 1, n).round().astype(int)

    def _make_temporal_strip(self, motion):
        """
        Create the frozen 4x8 temporal-strip protocol.
        Uses X horizontally and Y vertically with shared global XY bounds.
        """
        import matplotlib.pyplot as plt

        motion = validate_standardised_motion(motion)
        idx = self._sample_indices(len(motion), self.n_frames)
        sampled = motion[idx]

        # Shared bounds so pose scale is comparable across panels.
        xs = sampled[:, :, 0]
        ys = sampled[:, :, 1]
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())

        xspan = max(xmax - xmin, 1e-6)
        yspan = max(ymax - ymin, 1e-6)
        margin = 0.08 * max(xspan, yspan)

        rows, cols = 4, 8
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
        axes = np.asarray(axes).reshape(-1)

        for panel, ax in enumerate(axes):
            ax.axis("off")
            if panel >= len(sampled):
                continue

            pose = sampled[panel]
            for chain in self.KINEMATIC_CHAINS:
                pts = pose[chain]
                ax.plot(pts[:, 0], pts[:, 1], "-o", linewidth=1.2, markersize=2.2)

            ax.set_xlim(xmin - margin, xmax + margin)
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{int(idx[panel])}", fontsize=7)

        fig.suptitle(
            "Temporal strip — chronological order left-to-right, top-to-bottom",
            fontsize=10,
        )
        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return Path(tmp.name), idx.tolist()

    @staticmethod
    def _image_data_url(path):
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _instruction(prompt, requirement):
        req_type = requirement.get("type", "unknown")
        expected = requirement_value(requirement)

        return f"""
You are evaluating ONE atomic requirement of a generated 3D human motion.

The image is a temporal strip of uniformly sampled motion frames in chronological
order from left-to-right and top-to-bottom.

Full text prompt:
{prompt}

Atomic requirement:
type = {req_type}
expected value = {json.dumps(expected, ensure_ascii=False)}

Judge ONLY whether the visible motion satisfies this atomic requirement.
Do not grade unrelated parts of the full prompt.

Return exactly one JSON object:
{{
  "pass_fail": "PASS" or "FAIL",
  "confidence": "high" or "medium" or "low",
  "reason": "brief explanation based only on visible motion evidence"
}}
""".strip()

    @staticmethod
    def _extract_json(text):
        text = (text or "").strip()

        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"VLM response did not contain a JSON object: {text!r}")

        data = json.loads(text[start:end + 1])
        pf = str(data.get("pass_fail", "")).upper()
        if pf not in {"PASS", "FAIL"}:
            raise ValueError(f"Invalid VLM pass_fail: {data.get('pass_fail')!r}")

        confidence = str(data.get("confidence", "")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "unknown"

        return {
            "pass_fail": pf,
            "confidence": confidence,
            "reason": str(data.get("reason", "")).strip(),
        }

    def evaluate(self, motion, requirement, evaluation_case):
        motion = validate_standardised_motion(motion)
        image_path, indices = self._make_temporal_strip(motion)

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._instruction(
                                evaluation_case.get("prompt", ""),
                                requirement,
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": self._image_data_url(image_path),
                            "detail": self.detail,
                        },
                    ],
                }],
            )

            parsed = self._extract_json(response.output_text)

            return {
                "requirement_id": requirement.get("id"),
                "requirement_type": requirement.get("type"),
                "expected_value": requirement_value(requirement),
                "evaluator_name": self.EVALUATOR_NAME,
                "pass_fail": parsed["pass_fail"],
                "score": None,
                "threshold_status": "vlm_fallback_frozen_protocol",
                "reason": parsed["reason"],
                "evidence": {
                    "judge": "vlm_fallback",
                    "model": self.model,
                    "confidence": parsed["confidence"],
                    "frame_sampling": "uniform",
                    "n_frames": len(indices),
                    "sampled_indices": indices,
                    "visual_input": "temporal_strip_4x8",
                    "detail": self.detail,
                },
            }
        finally:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass


# Registering makes the class discoverable by load_all_evaluators().
# It is intentionally NOT added to EVALUATION_CONFIG because it is fallback-only.
register_evaluator("VLMEvaluator", VLMEvaluator)
