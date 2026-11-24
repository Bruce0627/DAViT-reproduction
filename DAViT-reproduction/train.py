"""Run the author training loop for downstream tasks or a supplied CXR8 split."""
import argparse
import csv
import logging
import sys
from pathlib import Path

from reproduce import CHECKPOINTS


def load_manifest(path):
    groups = {"train": [], "val": []}
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"path", "split", "label"}.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest columns: path, split, label")
        for row in reader:
            image = (path.parent / row["path"]).resolve()
            split = row["split"]
            label = int(row["label"])
            if split not in groups or label not in (0, 1) or image in seen:
                raise ValueError(f"Invalid or duplicate row for {image}")
            if not image.is_file():
                raise FileNotFoundError(image)
            groups[split].append((image, label))
            seen.add(image)
    for split, rows in groups.items():
        if {label for _, label in rows} != {0, 1}:
            raise ValueError(f"{split} must contain both labels")
    return groups


def build_cxr8_datasets(groups, processor):
    import torch
    from PIL import Image
    from torchvision import transforms

    edge = processor.size["shortest_edge"]
    normalize = transforms.Normalize(processor.image_mean, processor.image_std)

    class ImageDataset:
        def __init__(self, rows, transform):
            self.rows = rows
            self.transform = transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            path, label = self.rows[index]
            with Image.open(path) as image:
                return self.transform(image.convert("RGB")), torch.tensor(label, dtype=torch.long)

    return (
        ImageDataset(groups["train"], transforms.Compose([
            transforms.RandomResizedCrop((edge, edge)), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize,
        ])),
        ImageDataset(groups["val"], transforms.Compose([
            transforms.Resize((edge, edge)), transforms.CenterCrop((edge, edge)),
            transforms.ToTensor(), normalize,
        ])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["detection", "type", "adapt"], required=True)
    parser.add_argument("--data", type=Path, default=Path("DAViT/data/chest_xray"))
    parser.add_argument("--cxr8-manifest", type=Path)
    parser.add_argument("--domain-weights", type=Path, default=Path(
        "DAViT/davit/saved_models/checkpoint-best-f1/domain_adapted_davit.bin"
    ))
    parser.add_argument("--upstream", type=Path, default=Path("DAViT"))
    parser.add_argument("--backbone", default="facebook/dinov2-large")
    parser.add_argument("--output-dir", default="training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()
    if min(args.train_batch_size, args.eval_batch_size, args.epochs, args.gpus) < 1:
        parser.error("batch sizes, epochs and gpus must be positive")
    if args.gpus > 1 and args.device != "cuda":
        parser.error("multiple GPUs require --device cuda")
    if args.task == "adapt":
        if args.cxr8_manifest is None:
            parser.error("--cxr8-manifest is required for adaptation")
        groups = load_manifest(args.cxr8_manifest)
    elif not args.domain_weights.is_file():
        raise FileNotFoundError(args.domain_weights)

    upstream = args.upstream.resolve() / "davit"
    if not (upstream / "main.py").is_file():
        raise FileNotFoundError(upstream / "main.py")
    args.model_name = CHECKPOINTS.get(args.task, "domain_adapted_davit.bin")
    out = Path(args.output_dir) / "checkpoint-best-f1" / args.model_name
    if out.exists():
        raise FileExistsError(f"{out} exists. Use a new --output-dir.")

    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    if args.device.startswith("cuda") and torch.cuda.device_count() < args.gpus:
        parser.error(f"requested {args.gpus} GPUs; found {torch.cuda.device_count()}")
    sys.path.insert(0, str(upstream))
    from main import TextDataset, set_seed, train
    from model_davit import Model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.n_gpu = args.gpus if args.device.startswith("cuda") else 0
    args.learning_rate = 1e-5
    args.adam_epsilon = 1e-8
    args.weight_decay = 0.0
    args.gradient_accumulation_steps = 1
    args.train_data_file = str(args.data / "train")
    args.eval_data_file = str(args.data / "val")
    args.classify_pneumonia_type = args.task == "type"
    set_seed(args)
    processor = AutoImageProcessor.from_pretrained(args.backbone)
    if args.task == "adapt":
        model = Model(AutoModel.from_pretrained(args.backbone), processor, args)
        train_data, val_data = build_cxr8_datasets(groups, processor)
    else:
        model = Model(AutoModel.from_config(AutoConfig.from_pretrained(args.backbone)), processor, args)
        state = torch.load(args.domain_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        train_data = TextDataset(processor, args, file_type="train")
        val_data = TextDataset(processor, args, file_type="val")

    train(args, train_data, model, processor, val_data)
    if not out.is_file():
        raise RuntimeError("Training did not produce a checkpoint")
    print(out.resolve())


if __name__ == "__main__":
    main()
