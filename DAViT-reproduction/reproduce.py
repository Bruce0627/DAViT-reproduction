"""Evaluate the published DAViT checkpoints on the Kermany test split."""
import argparse
import csv
import json
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

REFERENCE = {
    "detection": {"tn": 218, "fp": 16, "fn": 8, "tp": 382},
    "type": {"tn": 240, "fp": 2, "fn": 46, "tp": 102},
}
COUNTS = {"detection": {0: 234, 1: 390}, "type": {0: 242, 1: 148}}
CHECKPOINTS = {
    "detection": "davit_pneumonia_detection.bin",
    "type": "davit_pneumonia_type.bin",
}


def metrics(tn, fp, fn, tp):
    def ratio(n, d):
        return n / d if d else 0.0
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": ratio(tp + tn, tn + fp + fn + tp),
        "precision": ratio(tp, tp + fp), "recall": recall,
        "specificity": specificity, "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "auc_hard_labels": (recall + specificity) / 2,
    }

def probability_auc(predictions):
    ordered = sorted(predictions, key=lambda row: row[3])
    neg_seen = pos_seen = 0
    concordance = 0.0
    for _, group in groupby(ordered, key=lambda row: row[3]):
        counts = Counter(row[1] for row in group)
        concordance += counts[1] * (neg_seen + counts[0] / 2)
        neg_seen += counts[0]
        pos_seen += counts[1]
    return concordance / (pos_seen * neg_seen)


def evaluate(args, task):
    weight_path = args.weights / CHECKPOINTS[task]
    for path in (args.upstream / "davit" / "model_davit.py", weight_path, args.data / "test"):
        if not path.exists():
            raise FileNotFoundError(path)

    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    sys.path.insert(0, str(args.upstream.resolve() / "davit"))
    from model_davit import Model

    backbone = args.backbone
    processor = AutoImageProcessor.from_pretrained(backbone)
    config = AutoConfig.from_pretrained(backbone)
    # The published downstream checkpoint contains the entire backbone.
    model = Model(AutoModel.from_config(config), processor, args)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    edge = processor.size["shortest_edge"]
    transform = Compose([
        Resize((edge, edge)), CenterCrop((edge, edge)), ToTensor(),
        Normalize(processor.image_mean, processor.image_std),
    ])
    rows = []
    for folder in ("NORMAL", "PNEUMONIA") if task == "detection" else ("PNEUMONIA",):
        for path in sorted((args.data / "test" / folder).rglob("*.jpeg")):
            if folder == "NORMAL":
                label = 0
            elif "bacteria" in path.name.lower():
                label = int(task == "detection")
            elif "virus" in path.name.lower():
                label = 1
            else:
                raise ValueError(f"No pneumonia type in {path}")
            rows.append((path, label))
    counts = Counter(label for _, label in rows)
    if counts != COUNTS[task]:
        raise ValueError(f"{task}: found {dict(counts)}, expected {COUNTS[task]}")

    def read_image(path):
        with Image.open(path) as image:
            return transform(image.convert("RGB"))

    predictions = []
    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        pixels = torch.stack([read_image(path) for path, _ in batch]).to(device)
        with torch.inference_mode():
            probabilities = model(pixels).cpu().tolist()
        for (path, label), (p0, p1) in zip(batch, probabilities):
            predictions.append((str(path.relative_to(args.data)), label, int(p1 > p0), p1))
        print(f"{task}: {len(predictions)}/{len(rows)}", flush=True)

    counts = Counter((label, pred) for _, label, pred, _ in predictions)
    report = metrics(counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])
    report["auc_probability"] = probability_auc(predictions)
    report["source"] = "fresh checkpoint inference"
    report["device"] = device
    reference = metrics(**REFERENCE[task])
    report["difference_from_author_log"] = {
        key: report[key] - reference[key]
        for key in ("accuracy", "precision", "recall", "specificity", "f1", "auc_hard_labels")
    }
    with (args.output / f"{task}_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "label", "prediction", "probability_class_1"])
        writer.writerows(predictions)

    if args.cam:
        import numpy as np
        report["cam_examples"] = []
        for path in args.cam:
            activations = []
            hook = model.cnn.register_forward_hook(lambda _m, _i, out: activations.append(out))
            try:
                pixels = read_image(path).unsqueeze(0).to(device)
                probabilities = model(pixels)
                target = args.target if args.target is not None else probabilities.argmax(1).item()
                grad, = torch.autograd.grad(probabilities[0, target], activations[0])
                cam = (grad.mean(dim=(2, 3), keepdim=True) * activations[0]).sum(1).relu().detach()
                cam = torch.nn.functional.interpolate(
                    cam[:, None], (edge, edge), mode="bilinear", align_corners=False
                )[0, 0]
                cam = ((cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-12)).cpu().numpy()
                with Image.open(path) as image:
                    base = np.asarray(
                        image.convert("RGB").resize((edge, edge), Image.Resampling.BILINEAR),
                        dtype=float,
                    )
                red = np.zeros_like(base)
                red[:, :, 0] = 255
                alpha = 0.65 * cam[:, :, None]
                name = f"{task}_cam_{path.stem}_class_{target}.png"
                Image.fromarray(np.uint8(base * (1 - alpha) + red * alpha)).save(args.output / name)
                report["cam_examples"].append({
                    "image": path.name,
                    "target": target,
                    "target_probability": probabilities[0, target].item(),
                    "heatmap": name,
                })
            finally:
                hook.remove()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--upstream", type=Path, default=Path("DAViT"))
    parser.add_argument("--data", type=Path, default=Path("DAViT/data/chest_xray"))
    parser.add_argument("--weights", type=Path, default=Path("DAViT/davit/saved_models/checkpoint-best-f1"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--backbone", default="facebook/dinov2-large")
    parser.add_argument("--task", choices=["detection", "type", "both"], default="both")
    parser.add_argument("--device")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cam", type=Path, action="append")
    parser.add_argument("--target", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.cam and (args.task == "both" or args.reference_only):
        parser.error("--cam needs checkpoint evaluation with one task")
    args.output.mkdir(parents=True, exist_ok=True)
    for task in REFERENCE if args.task == "both" else [args.task]:
        if args.reference_only:
            report = metrics(**REFERENCE[task])
            report["source"] = "reconstructed from published author log; no new inference"
        else:
            report = evaluate(args, task)
        name = f"{task}_{'reference' if args.reference_only else 'metrics'}.json"
        payload = json.dumps(report, indent=2) + "\n"
        (args.output / name).write_text(payload, encoding="utf-8")
        print(payload)


if __name__ == "__main__":
    main()
