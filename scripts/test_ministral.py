"""Load real cached weights and test text, UAV JSON and vision inference."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

# Avoid network lookups, including the separately cached FP8 CUDA kernel.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from agent.ministral import MODEL_ID, MinistralBackend
from agent.waypoint import parse_waypoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ministral_smoke.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model_path, "python": sys.version, "python_executable": sys.executable,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "accelerate", "kernels", "mujoco")},
        "tests": [],
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def check(name, callback):
        try:
            result = callback()
            report["tests"].append({"name": name, **result})
            print(name, json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            report["tests"].append({"name": name, "passed": False, "error": repr(exc)})
            print(name, repr(exc), flush=True)
        save()

    try:
        backend = MinistralBackend(args.model_path)
        report.update(gpu=torch.cuda.get_device_name(), fp8_tensors=backend.fp8_tensors,
                      load_seconds=backend.load_seconds)
        print("Model loaded:", json.dumps({k: report[k] for k in
              ("gpu", "fp8_tensors", "load_seconds")}), flush=True)

        def text_test():
            result = backend.generate("What is 2 + 3? Reply only with the number.", max_new_tokens=16)
            return result | {"passed": result["text"] == "5"}

        def action_test():
            result = backend.generate(
                'Current position is [0.0, 0.0, 1.0] in NWU coordinates. '
                'Move north by 0.5 meters and maintain altitude and yaw 0. '
                'Return only JSON with exactly these keys: "position" (absolute [x,y,z]), "yaw" (radians).',
                system=("You are a UAV waypoint planner using a world NWU coordinate frame. "
                        "North motion INCREASES X. West motion INCREASES Y. Up motion INCREASES Z. "
                        "For north motion keep Y and Z unchanged and add the distance to X. "
                        "Return a JSON object only, with no Markdown code fences or explanation."),
            )
            try:
                action = parse_waypoint(result["text"])
            except (json.JSONDecodeError, ValueError) as exc:
                return result | {"passed": False, "error": str(exc)}
            passed = (set(action) == {"position", "yaw"} and
                      action["position"] == [0.5, 0.0, 1.0] and action["yaw"] == 0)
            return result | {"parsed_action": action, "passed": passed,
                             "markdown_wrapper": result["text"].startswith("```")}

        def vision_test():
            image = Image.new("RGB", (224, 224), (255, 0, 0))
            result = backend.generate("What is the dominant color? Reply with one English word.",
                                      image=image, max_new_tokens=16)
            return result | {"passed": result["text"].lower().strip(". ") == "red"}

        def scene_test():
            import mujoco
            import numpy as np
            root = Path(__file__).resolve().parents[1]
            model = mujoco.MjModel.from_xml_path(str(root / "env/scene.xml"))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            with mujoco.Renderer(model, height=224, width=224) as renderer:
                renderer.update_scene(data, camera="fpv_camera")
                image = Image.fromarray(renderer.render().copy())
            image.save(args.output.parent / "mujoco_fpv.png")
            result = backend.generate("Briefly describe the visible scene and obstacles for a drone.",
                                      image=image, max_new_tokens=96)
            # This verifies rendering -> VLM execution, not navigation quality.
            return result | {"passed": bool(result["text"]) and bool(np.asarray(image).std() > 1),
                             "timestep": model.opt.timestep,
                             "nq": model.nq, "nu": model.nu,
                             "scope": "render_and_inference_only"}

        check("text_arithmetic", text_test)
        check("uav_waypoint_json", action_test)
        check("vision_color", vision_test)
        check("mujoco_fpv_vision", scene_test)
        report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["passed"] = all(test["passed"] for test in report["tests"])
        save()
    except Exception as exc:
        report.update(passed=False, error=repr(exc))
        save()
        raise
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
