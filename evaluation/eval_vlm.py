"""
eval_vlm.py — VLM fallback evaluator for unresolved requirements.

This evaluator is NOT a replacement for any primary evaluator.
It is called only after the normal rule/geometry evaluator returns
pass_fail=None.

Protocol:
- standardised motion [T,22,3]
- 32 uniformly sampled frames
- 4x8 temporal strip
- OpenAI Responses API
- PASS / FAIL only

Optimisation:
- Temporal strips are cached per (model, prompt_id)
- Multiple unresolved requirements from the same motion reuse
  exactly the same temporal strip
- evaluate_batch() evaluates ALL unresolved requirements from
  the same motion in ONE VLM API call
- evaluate() is retained for backward compatibility
"""

from __future__ import annotations

import os
import json
import base64
import tempfile
from pathlib import Path

import numpy as np

from common import (
    BaseEvaluator,
    validate_standardised_motion,
    requirement_value,
    register_evaluator,
)


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

    def __init__(
        self,
        model=None,
        n_frames=32,
        detail="high",
        client=None,
    ):

        self.model = model or os.getenv(
            "T2M_VLM_MODEL",
            self.DEFAULT_MODEL,
        )

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
                    "Set it in the runtime environment; "
                    "never commit API keys."
                )

            client = OpenAI()

        self.client = client

        # ----------------------------------------------------
        # TEMPORAL STRIP CACHE
        # ----------------------------------------------------
        # Key:
        #   (model, prompt_id)
        #
        # Value:
        #   {
        #       "image_data_url": ...,
        #       "indices": [...]
        #   }
        #
        # This allows multiple requirements belonging to the
        # same motion to reuse exactly the same visual evidence.
        # ----------------------------------------------------

        self._strip_cache = {}

    # ========================================================
    # BENCHMARK CONSTRUCTOR
    # ========================================================

    @classmethod
    def for_benchmark(cls):

        return cls(
            model=os.getenv(
                "T2M_VLM_MODEL",
                cls.DEFAULT_MODEL,
            ),
            n_frames=int(
                os.getenv(
                    "T2M_VLM_N_FRAMES",
                    cls.DEFAULT_N_FRAMES,
                )
            ),
            detail=os.getenv(
                "T2M_VLM_DETAIL",
                cls.DEFAULT_DETAIL,
            ),
        )

    # ========================================================
    # FRAME SAMPLING
    # ========================================================

    @staticmethod
    def _sample_indices(n_total, n_frames):

        if n_total < 1:
            raise ValueError(
                "Motion contains no frames."
            )

        n = min(
            int(n_frames),
            int(n_total),
        )

        return np.linspace(
            0,
            n_total - 1,
            n,
        ).round().astype(int)

    # ========================================================
    # TEMPORAL STRIP
    # ========================================================

    def _make_temporal_strip(self, motion):
        """
        Create the frozen 4x8 temporal-strip protocol.

        Uses X horizontally and Y vertically with shared
        global XY bounds.
        """

        import matplotlib.pyplot as plt

        motion = validate_standardised_motion(
            motion
        )

        idx = self._sample_indices(
            len(motion),
            self.n_frames,
        )

        sampled = motion[idx]

        # Shared bounds so pose scale is comparable
        # across panels.

        xs = sampled[:, :, 0]
        ys = sampled[:, :, 1]

        xmin = float(xs.min())
        xmax = float(xs.max())

        ymin = float(ys.min())
        ymax = float(ys.max())

        xspan = max(
            xmax - xmin,
            1e-6,
        )

        yspan = max(
            ymax - ymin,
            1e-6,
        )

        margin = 0.08 * max(
            xspan,
            yspan,
        )

        rows = 4
        cols = 8

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(16, 8),
        )

        axes = np.asarray(
            axes
        ).reshape(-1)

        for panel, ax in enumerate(axes):

            ax.axis("off")

            if panel >= len(sampled):
                continue

            pose = sampled[panel]

            for chain in self.KINEMATIC_CHAINS:

                pts = pose[chain]

                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    "-o",
                    linewidth=1.2,
                    markersize=2.2,
                )

            ax.set_xlim(
                xmin - margin,
                xmax + margin,
            )

            ax.set_ylim(
                ymin - margin,
                ymax + margin,
            )

            ax.set_aspect(
                "equal",
                adjustable="box",
            )

            ax.set_title(
                f"{int(idx[panel])}",
                fontsize=7,
            )

        fig.suptitle(
            "Temporal strip — chronological order "
            "left-to-right, top-to-bottom",
            fontsize=10,
        )

        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        )

        tmp.close()

        fig.savefig(
            tmp.name,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        return (
            Path(tmp.name),
            idx.tolist(),
        )

    # ========================================================
    # IMAGE ENCODING
    # ========================================================

    @staticmethod
    def _image_data_url(path):

        encoded = base64.b64encode(
            Path(path).read_bytes()
        ).decode("ascii")

        return (
            f"data:image/png;base64,{encoded}"
        )

    # ========================================================
    # TEMPORAL STRIP CACHE
    # ========================================================

    def _get_cached_temporal_strip(
        self,
        motion,
        evaluation_case,
    ):
        """
        Get temporal-strip visual evidence.

        If the same (model, prompt_id) has already been
        processed, reuse the cached image.
        """

        model_name = evaluation_case.get(
            "model",
            "unknown_model",
        )

        prompt_id = evaluation_case.get(
            "prompt_id",
            "unknown_prompt",
        )

        cache_key = (
            model_name,
            prompt_id,
        )

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        if cache_key in self._strip_cache:

            cached = self._strip_cache[
                cache_key
            ]

            return (
                cached["image_data_url"],
                cached["indices"],
                True,
            )

        # ----------------------------------------------------
        # CACHE MISS
        # ----------------------------------------------------

        image_path, indices = (
            self._make_temporal_strip(
                motion
            )
        )

        try:

            image_data_url = (
                self._image_data_url(
                    image_path
                )
            )

        finally:

            try:
                image_path.unlink(
                    missing_ok=True
                )

            except Exception:
                pass

        self._strip_cache[
            cache_key
        ] = {
            "image_data_url":
                image_data_url,
            "indices":
                indices,
        }

        return (
            image_data_url,
            indices,
            False,
        )

    # ========================================================
    # SINGLE REQUIREMENT PROMPT
    # ========================================================

    @staticmethod
    def _instruction(
        prompt,
        requirement,
    ):

        req_type = requirement.get(
            "type",
            "unknown",
        )

        expected = requirement_value(
            requirement
        )

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

    # ========================================================
    # BATCH REQUIREMENT PROMPT
    # ========================================================

    @staticmethod
    def _batch_instruction(
        prompt,
        requirements,
    ):
        """
        Create one prompt containing multiple unresolved
        requirements from the SAME motion.
        """

        items = []

        for item in requirements:

            req_index = item[
                "requirement_index"
            ]

            requirement = item[
                "requirement"
            ]

            items.append({
                "requirement_index":
                    req_index,
                "requirement_id":
                    requirement.get("id"),
                "type":
                    requirement.get(
                        "type",
                        "unknown",
                    ),
                "expected_value":
                    requirement_value(
                        requirement
                    ),
            })

        requirements_json = json.dumps(
            items,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
You are evaluating multiple atomic requirements of ONE generated 3D human motion.

IMPORTANT:
All requirements below belong to the SAME motion.

The image is a temporal strip of uniformly sampled motion frames in chronological
order from left-to-right and top-to-bottom.

Full text prompt:
{prompt}

Atomic requirements:
{requirements_json}

Evaluate EACH requirement independently.

For each requirement:
- Judge ONLY that atomic requirement.
- Do not grade unrelated parts of the full prompt.
- Use the SAME temporal-strip image as visual evidence.
- PASS means the visible motion satisfies that requirement.
- FAIL means the visible motion does not satisfy that requirement.

Return exactly one JSON object in this format:

{{
  "results": [
    {{
      "requirement_index": 0,
      "pass_fail": "PASS",
      "confidence": "high",
      "reason": "brief explanation based only on visible motion evidence"
    }}
  ]
}}

Rules:
- Return exactly one result for every supplied requirement.
- Keep the original requirement_index.
- pass_fail must be "PASS" or "FAIL".
- confidence must be "high", "medium", or "low".
- Do not omit any requirement.
- Do not add requirements.
- Return JSON only.
""".strip()

    # ========================================================
    # SINGLE JSON PARSER
    # ========================================================

    @staticmethod
    def _extract_json(text):

        text = (
            text or ""
        ).strip()

        if text.startswith("```"):

            text = text.strip(
                "`"
            ).strip()

            if text.lower().startswith(
                "json"
            ):
                text = text[
                    4:
                ].strip()

        start = text.find("{")
        end = text.rfind("}")

        if (
            start < 0
            or end < start
        ):

            raise ValueError(
                "VLM response did not contain "
                "a JSON object: "
                f"{text!r}"
            )

        data = json.loads(
            text[
                start:end + 1
            ]
        )

        pf = str(
            data.get(
                "pass_fail",
                "",
            )
        ).upper()

        if pf not in {
            "PASS",
            "FAIL",
        }:

            raise ValueError(
                "Invalid VLM pass_fail: "
                f"{data.get('pass_fail')!r}"
            )

        confidence = str(
            data.get(
                "confidence",
                "",
            )
        ).lower()

        if confidence not in {
            "high",
            "medium",
            "low",
        }:

            confidence = (
                "unknown"
            )

        return {
            "pass_fail":
                pf,
            "confidence":
                confidence,
            "reason":
                str(
                    data.get(
                        "reason",
                        "",
                    )
                ).strip(),
        }

    # ========================================================
    # BATCH JSON PARSER
    # ========================================================

    @staticmethod
    def _extract_batch_json(
        text,
        expected_indices,
    ):
        """
        Parse and validate a batch VLM response.
        """

        text = (
            text or ""
        ).strip()

        if text.startswith("```"):

            text = text.strip(
                "`"
            ).strip()

            if text.lower().startswith(
                "json"
            ):

                text = text[
                    4:
                ].strip()

        start = text.find("{")
        end = text.rfind("}")

        if (
            start < 0
            or end < start
        ):

            raise ValueError(
                "Batch VLM response did not "
                "contain a JSON object: "
                f"{text!r}"
            )

        data = json.loads(
            text[
                start:end + 1
            ]
        )

        raw_results = data.get(
            "results"
        )

        if not isinstance(
            raw_results,
            list,
        ):

            raise ValueError(
                "Batch VLM response "
                "does not contain a "
                "'results' list."
            )

        parsed = {}

        for item in raw_results:

            if not isinstance(
                item,
                dict,
            ):
                continue

            try:
                req_index = int(
                    item.get(
                        "requirement_index"
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            pf = str(
                item.get(
                    "pass_fail",
                    "",
                )
            ).upper()

            if pf not in {
                "PASS",
                "FAIL",
            }:

                raise ValueError(
                    "Invalid batch VLM "
                    "pass_fail for "
                    f"requirement "
                    f"{req_index}: "
                    f"{pf!r}"
                )

            confidence = str(
                item.get(
                    "confidence",
                    "",
                )
            ).lower()

            if confidence not in {
                "high",
                "medium",
                "low",
            }:

                confidence = (
                    "unknown"
                )

            parsed[
                req_index
            ] = {
                "pass_fail":
                    pf,
                "confidence":
                    confidence,
                "reason":
                    str(
                        item.get(
                            "reason",
                            "",
                        )
                    ).strip(),
            }

        expected_indices = set(
            int(x)
            for x in expected_indices
        )

        returned_indices = set(
            parsed.keys()
        )

        missing = (
            expected_indices
            - returned_indices
        )

        extra = (
            returned_indices
            - expected_indices
        )

        if missing:

            raise ValueError(
                "Batch VLM response "
                "omitted requirement "
                "indices: "
                f"{sorted(missing)}"
            )

        if extra:

            raise ValueError(
                "Batch VLM response "
                "returned unexpected "
                "requirement indices: "
                f"{sorted(extra)}"
            )

        return parsed

    # ========================================================
    # ORIGINAL SINGLE REQUIREMENT EVALUATION
    # ========================================================

    def evaluate(
        self,
        motion,
        requirement,
        evaluation_case,
    ):
        """
        Evaluate ONE atomic requirement.

        Retained for backward compatibility.

        Temporal-strip generation is cached per
        (model, prompt_id), but this method still
        performs one VLM API call per requirement.
        """

        motion = (
            validate_standardised_motion(
                motion
            )
        )

        image_data_url, indices, cache_hit = (
            self._get_cached_temporal_strip(
                motion,
                evaluation_case,
            )
        )

        response = (
            self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role":
                            "user",
                        "content": [
                            {
                                "type":
                                    "input_text",
                                "text":
                                    self._instruction(
                                        evaluation_case.get(
                                            "prompt",
                                            "",
                                        ),
                                        requirement,
                                    ),
                            },
                            {
                                "type":
                                    "input_image",
                                "image_url":
                                    image_data_url,
                                "detail":
                                    self.detail,
                            },
                        ],
                    }
                ],
            )
        )

        parsed = self._extract_json(
            response.output_text
        )

        return {
            "requirement_id":
                requirement.get(
                    "id"
                ),
            "requirement_type":
                requirement.get(
                    "type"
                ),
            "expected_value":
                requirement_value(
                    requirement
                ),
            "evaluator_name":
                self.EVALUATOR_NAME,
            "pass_fail":
                parsed[
                    "pass_fail"
                ],
            "score":
                None,
            "threshold_status":
                "vlm_fallback_frozen_protocol",
            "reason":
                parsed[
                    "reason"
                ],
            "evidence": {
                "judge":
                    "vlm_fallback",
                "model":
                    self.model,
                "confidence":
                    parsed[
                        "confidence"
                    ],
                "frame_sampling":
                    "uniform",
                "n_frames":
                    len(indices),
                "sampled_indices":
                    indices,
                "visual_input":
                    "temporal_strip_4x8",
                "detail":
                    self.detail,
                "temporal_strip_cache_hit":
                    cache_hit,
                "batch_size":
                    1,
            },
        }

    # ========================================================
    # NEW BATCH EVALUATION
    # ========================================================

    def evaluate_batch(
        self,
        motion,
        requirements,
        evaluation_case,
    ):
        """
        Evaluate ALL unresolved requirements belonging
        to ONE motion using ONE VLM API call.

        requirements format:

        [
            {
                "requirement_index": 0,
                "requirement": {...}
            },
            ...
        ]

        Returns:

        {
            requirement_index: result,
            ...
        }
        """

        if not requirements:
            return {}

        motion = (
            validate_standardised_motion(
                motion
            )
        )

        # ----------------------------------------------------
        # ONE shared temporal strip for the whole motion.
        # ----------------------------------------------------

        image_data_url, indices, cache_hit = (
            self._get_cached_temporal_strip(
                motion,
                evaluation_case,
            )
        )

        expected_indices = [
            int(
                item[
                    "requirement_index"
                ]
            )
            for item in requirements
        ]

        instruction = (
            self._batch_instruction(
                evaluation_case.get(
                    "prompt",
                    "",
                ),
                requirements,
            )
        )

        # ----------------------------------------------------
        # CRITICAL OPTIMISATION
        #
        # ONE API CALL for ALL unresolved requirements
        # belonging to this motion.
        # ----------------------------------------------------

        response = (
            self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role":
                            "user",
                        "content": [
                            {
                                "type":
                                    "input_text",
                                "text":
                                    instruction,
                            },
                            {
                                "type":
                                    "input_image",
                                "image_url":
                                    image_data_url,
                                "detail":
                                    self.detail,
                            },
                        ],
                    }
                ],
            )
        )

        parsed_results = (
            self._extract_batch_json(
                response.output_text,
                expected_indices,
            )
        )

        results = {}

        for item in requirements:

            req_index = int(
                item[
                    "requirement_index"
                ]
            )

            requirement = item[
                "requirement"
            ]

            parsed = (
                parsed_results[
                    req_index
                ]
            )

            results[
                req_index
            ] = {
                "requirement_id":
                    requirement.get(
                        "id"
                    ),
                "requirement_type":
                    requirement.get(
                        "type"
                    ),
                "expected_value":
                    requirement_value(
                        requirement
                    ),
                "evaluator_name":
                    self.EVALUATOR_NAME,
                "pass_fail":
                    parsed[
                        "pass_fail"
                    ],
                "score":
                    None,
                "threshold_status":
                    "vlm_batch_fallback_frozen_protocol",
                "reason":
                    parsed[
                        "reason"
                    ],
                "evidence": {
                    "judge":
                        "vlm_batch_fallback",
                    "model":
                        self.model,
                    "confidence":
                        parsed[
                            "confidence"
                        ],
                    "frame_sampling":
                        "uniform",
                    "n_frames":
                        len(indices),
                    "sampled_indices":
                        indices,
                    "visual_input":
                        "temporal_strip_4x8",
                    "detail":
                        self.detail,
                    "temporal_strip_cache_hit":
                        cache_hit,
                    "batch_size":
                        len(
                            requirements
                        ),
                },
            }

        return results


# ============================================================
# REGISTER
# ============================================================

# Registering makes the class discoverable by
# load_all_evaluators().
#
# It is intentionally NOT added to EVALUATION_CONFIG
# because it is fallback-only.

register_evaluator(
    "VLMEvaluator",
    VLMEvaluator,
)
