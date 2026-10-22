"""Generate the deterministic 40-image synthetic PII benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from common import ANNOTATION_VERSION, load_annotations, normalize_value


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
FONT_FILES = ("arial.ttf", "times.ttf", "calibri.ttf", "segoeui.ttf")
FONT_DIRECTORY_CANDIDATES = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/msttcorefonts"),
    Path("/usr/share/fonts/truetype/microsoft"),
)
DEV_NAMES = (
    "Avery Sample", "Mira Example", "Tarin Sample", "Niko Example",
    "Sana Sample", "Kian Example", "Liora Sample", "Darin Example",
    "Veda Sample", "Rafi Example",
)
HELDOUT_NAMES = ("Zora Example", "Ilan Sample", "Noor Example", "Oren Sample")
DIMENSIONS = (
    (1024, 768),
    (1280, 960),
    (1600, 1000),
    (2048, 1280),
    (3072, 1600),
    (1600, 2304),
)
PALETTES = (
    ((255, 255, 255), (20, 24, 30)),
    ((246, 247, 241), (42, 44, 48)),
    ((237, 243, 248), (35, 48, 60)),
    ((248, 239, 229), (68, 50, 42)),
)
NEGATIVE_LINES = (
    "QUANTITY 4821",
    "TOTAL 005.25",
    "ACCESS KEY X7",
    "REFERENCE 2025550188",
    "STATUS READY",
    "USER ID 004211",
)


def _composition(split: str) -> list[tuple[str, ...]]:
    if split == "development":
        return (
            [("person_name",)] * 6
            + [("email",)] * 6
            + [("phone",)] * 6
            + [("person_name", "email", "phone")] * 4
            + [()] * 6
        )
    if split == "heldout":
        return (
            [("person_name",)] * 2
            + [("email",)] * 2
            + [("phone",)] * 2
            + [("person_name", "email", "phone")] * 2
            + [()] * 4
        )
    raise ValueError("split must be development or heldout")


def _find_font_directory(override: Path | None) -> Path:
    candidates = (override,) if override is not None else FONT_DIRECTORY_CANDIDATES
    for directory in candidates:
        if directory is not None and all((directory / name).is_file() for name in FONT_FILES):
            return directory
    raise FileNotFoundError(
        "Required fonts not found; provide --font-dir containing " + ", ".join(FONT_FILES)
    )


def _font_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(split: str) -> dict[str, tuple[str, ...]]:
    names = DEV_NAMES if split == "development" else HELDOUT_NAMES
    count = len(names)
    if split == "development":
        email_domain = "example.com"
        phone_start = 100
    else:
        email_domain = "example.net"
        phone_start = 150
    return {
        "person_name": names,
        "email": tuple(f"synthetic-{split[:3]}-{index + 1:02d}@{email_domain}" for index in range(count)),
        "phone": tuple(
            f"+1 202-555-{phone_start + index:04d}" for index in range(count)
        ),
    }


def _draw_text(
    draw: ImageDraw.ImageDraw,
    value: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
) -> list[int]:
    draw.text((x, y), value, font=font, fill=color)
    return [int(number) for number in draw.textbbox((x, y), value, font=font)]


def _draw_negative(
    draw: ImageDraw.ImageDraw,
    *,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    width: int,
    height: int,
    split: str,
    index: int,
) -> None:
    lines = [NEGATIVE_LINES[(index + offset) % len(NEGATIVE_LINES)] for offset in range(3)]
    y_start = int(height * (0.20 if split == "development" else 0.30))
    line_gap = max(52, int(font.size * 1.55))
    for line_index, text in enumerate(lines):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        if split == "development":
            x = int(width * (0.08 + 0.08 * (line_index % 2)))
        else:
            x = max(24, (width - text_width) // 2)
        draw.text((x, y_start + line_index * line_gap), text, font=font, fill=color)


def _entity_rendering(
    category: str,
    value: str,
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    width: int,
    height: int,
    split: str,
    template_index: int,
    row_index: int,
    split_phone: bool,
) -> tuple[list[int], list[list[int]]]:
    y = int(height * (0.16 if split == "development" else 0.22)) + row_index * max(
        int(font.size * 1.9), int(height * 0.12)
    )
    if category == "phone" and split_phone:
        # Split after the exchange code so both fragments contain visible digits.
        first_piece, second_piece = value.rsplit("-", 1)
        first_box = draw.textbbox((0, 0), first_piece, font=font)
        second_box = draw.textbbox((0, 0), second_piece, font=font)
        first_width = first_box[2] - first_box[0]
        second_width = second_box[2] - second_box[0]
        gap = max(10, int(font.size * 0.22))
        if width >= 3000:
            x = max(24, 1600 - first_width - gap // 2)
        else:
            x = int(width * (0.12 if split == "development" else 0.19))
        first_region = _draw_text(draw, first_piece, x, y, font, color)
        second_region = _draw_text(draw, second_piece, x + first_width + gap, y, font, color)
        union = [
            min(first_region[0], second_region[0]),
            min(first_region[1], second_region[1]),
            max(first_region[2], second_region[2]),
            max(first_region[3], second_region[3]),
        ]
        return union, [first_region, second_region]

    text_box = draw.textbbox((0, 0), value, font=font)
    text_width = text_box[2] - text_box[0]
    available = max(1, width - text_width - 48)
    if split == "development":
        anchors = (0.08, 0.30, 0.53, 0.74)
        x = int(24 + available * anchors[template_index % len(anchors)])
    else:
        x = max(24, (width - text_width) // 2)
    region = _draw_text(draw, value, x, y, font, color)
    return region, []


def _make_image(
    split: str,
    index: int,
    kinds: tuple[str, ...],
    font_directory: Path,
) -> tuple[Image.Image, dict[str, Any], dict[str, str]]:
    template_index = index % 6
    template_id = f"{split}-layout-{template_index + 1:02d}"
    width, height = DIMENSIONS[(index * 5 + (1 if split == "heldout" else 0)) % len(DIMENSIONS)]
    font_path = font_directory / FONT_FILES[(index + (2 if split == "heldout" else 0)) % len(FONT_FILES)]
    font_size = 34 + (index % 4) * 8
    font = ImageFont.truetype(str(font_path), font_size)
    background, color = PALETTES[(index + (1 if split == "heldout" else 0)) % len(PALETTES)]
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    if index % 3 == 0:
        rng = random.Random((9301 if split == "development" else 1987) + index * 101)
        for _ in range(max(40, width * height // 180000)):
            x = rng.randrange(width)
            y = rng.randrange(height)
            delta = rng.choice((-8, -5, 5, 8))
            tint = tuple(max(0, min(255, channel + delta)) for channel in background)
            draw.rectangle((x, y, min(width - 1, x + 1), min(height - 1, y + 1)), fill=tint)

    image_id = f"{'dev' if split == 'development' else 'heldout'}-{index + 1:03d}"
    font_hash = _font_hash(font_path)
    metadata = {
        "annotation_version": ANNOTATION_VERSION,
        "image_id": image_id,
        "split": split,
        "image_file": f"images/{split}/{image_id}.png",
        "image_size": {"width": width, "height": height},
        "template_id": template_id,
        "generation": {
            "seed": (9301 if split == "development" else 1987) + index * 101,
            "font_file": font_path.name,
            "font_sha256": font_hash,
            "font_size": font_size,
            "text_color_rgb": list(color),
            "background_rgb": list(background),
            "texture": index % 3 == 0,
        },
        "entities": [],
    }
    if not kinds:
        _draw_negative(
            draw,
            font=font,
            color=color,
            width=width,
            height=height,
            split=split,
            index=index,
        )
        return image, metadata, {"font_file": font_path.name, "font_sha256": font_hash}

    values = _values(split)
    ordinals = {category: 0 for category in values}
    # The fixed composition order makes per-split identities disjoint by construction.
    for prior_kinds in _composition(split)[:index]:
        for prior_category in prior_kinds:
            ordinals[prior_category] += 1
    for row_index, category in enumerate(kinds):
        value = values[category][ordinals[category]]
        split_phone = category == "phone" and ((index + (split == "heldout")) % 3 == 0)
        bbox, fragments = _entity_rendering(
            category,
            value,
            draw=draw,
            font=font,
            color=color,
            width=width,
            height=height,
            split=split,
            template_index=template_index,
            row_index=row_index,
            split_phone=bool(split_phone),
        )
        metadata["entities"].append(
            {
                "category": category,
                "expected_value": value,
                "expected_normalized_value": normalize_value(category, value),
                "bbox_xyxy": bbox,
                "fragment_boxes_xyxy": fragments,
            }
        )
        ordinals[category] += 1
    return image, metadata, {"font_file": font_path.name, "font_sha256": font_hash}


def generate_dataset(
    output_dir: Path = DATA_ROOT,
    *,
    font_directory: Path | None = None,
) -> dict[str, object]:
    fonts = _find_font_directory(font_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    annotation_root = output_dir / "annotations"
    for split in ("development", "heldout"):
        (image_root / split).mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)

    all_metadata: dict[str, list[dict[str, Any]]] = {}
    font_records: dict[str, str] = {}
    for split in ("development", "heldout"):
        records: list[dict[str, Any]] = []
        for index, kinds in enumerate(_composition(split)):
            image, annotation, font_record = _make_image(split, index, kinds, fonts)
            image.save(image_root / split / f"{annotation['image_id']}.png", format="PNG", optimize=False)
            records.append(annotation)
            font_records[font_record["font_file"]] = font_record["font_sha256"]
        annotation_path = annotation_root / f"{split}.jsonl"
        annotation_path.write_text(
            "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records),
            encoding="utf-8",
        )
        all_metadata[split] = records

    manifest = {
        "dataset_version": ANNOTATION_VERSION,
        "generation_seed_policy": "fixed per split and image index",
        "font_sha256_by_file": font_records,
        "split_image_counts": {split: len(records) for split, records in all_metadata.items()},
        "split_entity_counts": {
            split: dict(Counter(entity["category"] for record in records for entity in record["entities"]))
            for split, records in all_metadata.items()
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_dataset(data_dir: Path = DATA_ROOT) -> dict[str, int]:
    totals: dict[str, int] = {}
    template_sets: dict[str, set[str]] = {}
    identity_sets: dict[str, set[str]] = {}
    expected_compositions = {
        "development": Counter(
            {("person_name",): 6, ("email",): 6, ("phone",): 6,
             ("person_name", "email", "phone"): 4, (): 6}
        ),
        "heldout": Counter(
            {("person_name",): 2, ("email",): 2, ("phone",): 2,
             ("person_name", "email", "phone"): 2, (): 4}
        ),
    }
    for split in ("development", "heldout"):
        records = load_annotations(data_dir / "annotations" / f"{split}.jsonl", expected_split=split)
        if len(records) != len(_composition(split)):
            raise ValueError("unexpected image count")
        actual_compositions = Counter(
            tuple(entity.category for entity in record.entities) for record in records
        )
        if actual_compositions != expected_compositions[split]:
            raise ValueError("unexpected category composition")
        template_sets[split] = {record.template_id for record in records}
        identity_sets[split] = {
            entity.normalized_value for record in records for entity in record.entities
        }
        for record in records:
            image_path = data_dir / record.image_file
            with Image.open(image_path) as image:
                if image.size != (record.width, record.height):
                    raise ValueError("annotation image dimensions do not match image file")
                image.verify()
        totals[split] = len(records)
    if template_sets["development"] & template_sets["heldout"]:
        raise ValueError("layout templates must be split-disjoint")
    if identity_sets["development"] & identity_sets["heldout"]:
        raise ValueError("synthetic identities and values must be split-disjoint")
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, help="directory containing the four documented TTF fonts")
    parser.add_argument("--validate-only", action="store_true", help="validate existing annotations and images")
    arguments = parser.parse_args()
    if arguments.validate_only:
        totals = validate_dataset()
        print(f"validated development={totals['development']} heldout={totals['heldout']} total={sum(totals.values())}")
        return 0
    manifest = generate_dataset(font_directory=arguments.font_dir)
    totals = validate_dataset()
    print(f"generated and validated {sum(totals.values())} synthetic images")
    print("category instance counts:", json.dumps(manifest["split_entity_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
