"""Sequential four-branch inference, verification on the unprompted video."""

from __future__ import annotations

import tempfile
import time
import math
from pathlib import Path

from selectvp.config import config_identity
from selectvp.preprocessing.runner import Preprocessor
from selectvp.selection.selector import BRANCHES, select_aligned
from selectvp.utils.io import artifact_key, fingerprint, id_key, index_by_id, read_json, write_json, write_run_manifest
from selectvp.utils.parsing import parse_response, valid_interval
from selectvp.utils.video import cut_video_segment


def infer(config, samples, universe, branches=BRANCHES, retry_failed=False, model=None):
    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    preprocessing = Preprocessor(config, samples, universe)
    # Validate every required cache before loading a large model.
    for sample in samples:
        for branch in branches:
            if branch != "bl":
                preprocessing.prompted(sample, branch)
    identity = fingerprint({"config": config_identity(config), "samples": samples,
                            "videos": {s["video"]: preprocessing.input_identity(s)["video_sha256"] for s in samples}})
    state_file = output / "run_identity.json"
    if state_file.exists() and read_json(state_file)["signature"] != identity:
        raise ValueError("Output directory belongs to different inputs/config; choose a new paths.output")
    write_json(state_file, {"signature": identity})
    write_run_manifest(output, config, samples, "inference")
    started = time.monotonic()
    for branch in branches:
        destination = output / f"{branch}.json"
        existing = index_by_id(read_json(destination)) if destination.exists() else {}
        expected = {id_key(s["id"]) for s in samples}
        if set(existing) - expected:
            raise ValueError(f"Unexpected IDs in resume file: {destination}")
        with tempfile.TemporaryDirectory(prefix="selectvp-") as directory:
            for sample in samples:
                key = id_key(sample["id"])
                if key in existing and (not retry_failed or existing[key].get("status") == "ok"):
                    continue
                if model is None:
                    from selectvp.models.qwen3_vl import Qwen3VL
                    model = Qwen3VL(config["model"])
                    write_json(output / "model.json", model.metadata())
                candidate = {"id": sample["id"], "branch": branch, "response": None, "pred_start": None,
                             "pred_end": None, "conf": 0.0, "status": "generation_failed"}
                try:
                    video, flags = (preprocessing.video(sample), None) if branch == "bl" else preprocessing.prompted(sample, branch)
                    text = config["prompts"][branch] + config["prompts"]["query"].format(sample["query"])
                    response = model.generate(video, text, flags)
                    start, end = parse_response(response)
                    candidate.update(response=response, pred_start=start, pred_end=end, status="invalid_interval")
                    if valid_interval(start, end):
                        segment = Path(directory) / f"{artifact_key(sample['id'])}.mp4"
                        # Always crop the original, unprompted 1 FPS input, also for OD.
                        candidate["status"] = "crop_failed"
                        if cut_video_segment(preprocessing.video(sample), segment, start, end):
                            candidate["status"] = "verification_failed"
                            score, verification_text = model.verify(segment, config["prompts"]["verification"].format(query=sample["query"]))
                            if not math.isfinite(score):
                                raise ValueError("Verification produced non-finite confidence")
                            candidate.update(conf=score, verification_response=verification_text, status="ok")
                except Exception as exc:
                    candidate["error"] = f"{type(exc).__name__}: {exc}"
                existing[key] = candidate
                write_json(destination, [existing[id_key(s["id"])] for s in samples if id_key(s["id"]) in existing])
                print(f"{branch} {sample['id']}: {candidate['status']} score={candidate['conf']:.6f}", flush=True)
    if all((output / f"{b}.json").exists() for b in BRANCHES):
        rows = {branch: read_json(output / f"{branch}.json") for branch in BRANCHES}
        selected = select_aligned(rows, [s["id"] for s in samples])
        write_json(output / "predictions.json", selected)
        failures = {branch: sum(x.get("status") != "ok" for x in records) for branch, records in rows.items()}
        write_json(output / "inference_report.json", {"num_samples": len(samples), "failed_candidates": failures,
                                                     "elapsed_seconds_this_invocation": time.monotonic()-started})
        return not any(failures.values())
    return all(row.get("status") == "ok" for branch in branches for row in read_json(output / f"{branch}.json"))
