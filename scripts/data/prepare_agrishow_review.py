from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.parse
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from project_paths import DATA_DIR, OUTPUTS_DIR

TARGET_ID = "agrishow2022_person_01"
ALLOWED_PRESENCE = {"", "present", "absent", "uncertain"}
ANNOTATION_COLUMNS = (
    "image_id",
    "file_name",
    "target_id",
    "presence",
    "review_status",
    "notes",
    "commons_page_url",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the independent manual-presence review for Agrishow 2022."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATA_DIR / "raw" / "agrishow_2022",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR
        / "experiments"
        / "ex-020_agrishow_annotation_review",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=DATA_DIR
        / "annotations"
        / "agrishow_2022_presence_review.csv",
    )
    parser.add_argument("--thumbnail-width", type=int, default=640)
    parser.add_argument("--thumbnail-height", type=int, default=480)
    parser.add_argument("--overwrite-thumbnails", action="store_true")
    parser.add_argument("--overwrite-annotations", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 124:
        raise RuntimeError(f"Expected 124 selected records, found {len(rows)}.")
    image_ids = [row["image_id"] for row in rows]
    file_names = [row["file_name"] for row in rows]
    if len(set(image_ids)) != len(rows) or len(set(file_names)) != len(rows):
        raise RuntimeError("Inventory contains duplicate image IDs or file names.")
    return rows


def validate_images(rows: list[dict[str, str]], image_dir: Path) -> None:
    expected_names = {row["file_name"] for row in rows}
    actual_names = {path.name for path in image_dir.iterdir() if path.is_file()}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise RuntimeError(
            f"Dataset mismatch: missing={len(missing)} unexpected={len(unexpected)}."
        )
    for row in rows:
        image_path = image_dir / row["file_name"]
        if image_path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Size mismatch: {image_path}")


def write_annotation_template(
    path: Path,
    rows: list[dict[str, str]],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        if len(existing) != len(rows):
            raise RuntimeError(
                f"Existing annotation file has {len(existing)} rows, expected {len(rows)}."
            )
        for row in existing:
            if row.get("presence", "") not in ALLOWED_PRESENCE:
                raise RuntimeError(
                    f"Invalid presence value for {row.get('image_id')}: "
                    f"{row.get('presence')!r}"
                )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "target_id": TARGET_ID,
                    "presence": "",
                    "review_status": "pending",
                    "notes": "",
                    "commons_page_url": row["page_url"],
                }
            )


def make_thumbnails(
    rows: list[dict[str, str]],
    image_dir: Path,
    thumbnail_dir: Path,
    max_size: tuple[int, int],
    overwrite: bool,
) -> None:
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(rows, start=1):
        source = image_dir / row["file_name"]
        destination = thumbnail_dir / f"{row['image_id']}.jpg"
        if destination.exists() and not overwrite:
            continue
        temporary = destination.with_suffix(".jpg.part")
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            image.save(temporary, format="JPEG", quality=88, optimize=True)
        temporary.replace(destination)
        print(f"[{number:03d}/{len(rows):03d}] thumbnail: {row['image_id']}")


def relative_url(target: Path, base_dir: Path) -> str:
    relative = os.path.relpath(target, base_dir).replace("\\", "/")
    return urllib.parse.quote(relative, safe="/._-")


def write_review_html(
    path: Path,
    rows: list[dict[str, str]],
    image_dir: Path,
    thumbnail_dir: Path,
) -> None:
    review_rows = []
    for row in rows:
        review_rows.append(
            {
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "thumbnail": relative_url(
                    thumbnail_dir / f"{row['image_id']}.jpg", path.parent
                ),
                "original": relative_url(image_dir / row["file_name"], path.parent),
                "commons_page_url": row["page_url"],
            }
        )
    records_json = json.dumps(review_rows, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EX-020 — Revisão Agrishow 2022</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; background: #f4f6f3; color: #172018; }
header { position: sticky; top: 0; z-index: 2; padding: 14px 20px; background: #173f2a; color: white; }
header h1 { margin: 0 0 6px; font-size: 21px; }
header p { margin: 4px 0; }
#toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button, a.action { border: 1px solid #829087; border-radius: 6px; padding: 7px 10px; background: white; color: #172018; cursor: pointer; text-decoration: none; }
button.selected { color: white; border-color: #173f2a; background: #24613f; }
#gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 14px; padding: 16px; }
.card { background: white; border: 2px solid transparent; border-radius: 9px; padding: 10px; box-shadow: 0 2px 8px #0001; }
.card.answered { border-color: #4f8a63; }
.card img { width: 100%; height: 260px; object-fit: contain; background: #111; }
.image-id { font: 13px Consolas, monospace; overflow-wrap: anywhere; margin: 8px 0; }
.choices { display: flex; gap: 6px; flex-wrap: wrap; }
.links { display: flex; gap: 10px; margin: 9px 0; font-size: 14px; }
.notes { width: 100%; box-sizing: border-box; min-height: 55px; }
.warning { color: #ffe59a; }
</style>
</head>
<body>
<header>
  <h1>EX-020 — presença da pessoa-alvo</h1>
  <p>Marque pela imagem, sem consultar o ranking dos modelos. Abra o original quando o rosto estiver pequeno.</p>
  <div id="toolbar">
    <strong id="progress">0/124 revisadas</strong>
    <button id="export">Exportar CSV</button>
    <button id="clear">Limpar marcações locais</button>
    <span class="warning">O navegador guarda rascunho local; exporte o CSV ao terminar.</span>
  </div>
</header>
<main id="gallery"></main>
<script>
const records = __RECORDS__;
const targetId = "agrishow2022_person_01";
const storageKey = "ex020-agrishow-presence-v1";
let state = JSON.parse(localStorage.getItem(storageKey) || "{}");

function save() { localStorage.setItem(storageKey, JSON.stringify(state)); renderProgress(); }
function renderProgress() {
  const answered = records.filter(r => state[r.image_id]?.presence).length;
  document.getElementById("progress").textContent = `${answered}/${records.length} revisadas`;
}
function setPresence(id, value, card) {
  state[id] = state[id] || {};
  state[id].presence = value;
  card.querySelectorAll("button[data-value]").forEach(button => {
    button.classList.toggle("selected", button.dataset.value === value);
  });
  card.classList.add("answered");
  save();
}
function render() {
  const gallery = document.getElementById("gallery");
  records.forEach(record => {
    const card = document.createElement("article");
    card.className = "card" + (state[record.image_id]?.presence ? " answered" : "");
    const image = document.createElement("img");
    image.src = record.thumbnail;
    image.loading = "lazy";
    image.alt = record.image_id;
    card.appendChild(image);
    const id = document.createElement("div");
    id.className = "image-id";
    id.textContent = record.image_id;
    card.appendChild(id);
    const choices = document.createElement("div");
    choices.className = "choices";
    [["present", "Presente"], ["absent", "Ausente"], ["uncertain", "Incerto"]].forEach(([value, label]) => {
      const button = document.createElement("button");
      button.dataset.value = value;
      button.textContent = label;
      if (state[record.image_id]?.presence === value) button.classList.add("selected");
      button.onclick = () => setPresence(record.image_id, value, card);
      choices.appendChild(button);
    });
    card.appendChild(choices);
    const links = document.createElement("div");
    links.className = "links";
    [[record.original, "Abrir original"], [record.commons_page_url, "Ver fonte"]].forEach(([href, label]) => {
      const anchor = document.createElement("a");
      anchor.className = "action";
      anchor.href = href;
      anchor.target = "_blank";
      anchor.textContent = label;
      links.appendChild(anchor);
    });
    card.appendChild(links);
    const notes = document.createElement("textarea");
    notes.className = "notes";
    notes.placeholder = "Observação sobre oclusão, escala ou dúvida";
    notes.value = state[record.image_id]?.notes || "";
    notes.oninput = () => {
      state[record.image_id] = state[record.image_id] || {};
      state[record.image_id].notes = notes.value;
      save();
    };
    card.appendChild(notes);
    gallery.appendChild(card);
  });
  renderProgress();
}
function csvCell(value) { return `"${String(value || "").replaceAll('"', '""')}"`; }
document.getElementById("export").onclick = () => {
  const header = ["image_id", "file_name", "target_id", "presence", "review_status", "notes", "commons_page_url"];
  const lines = [header.map(csvCell).join(",")];
  records.forEach(record => {
    const answer = state[record.image_id] || {};
    const row = [record.image_id, record.file_name, targetId, answer.presence || "", answer.presence ? "first_review_completed" : "pending", answer.notes || "", record.commons_page_url];
    lines.push(row.map(csvCell).join(","));
  });
  const blob = new Blob(["\\ufeff" + lines.join("\\r\\n")], {type: "text/csv;charset=utf-8"});
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = "agrishow_2022_presence_review.csv";
  anchor.click();
  URL.revokeObjectURL(anchor.href);
};
document.getElementById("clear").onclick = () => {
  if (confirm("Apagar todas as marcações salvas neste navegador?")) {
    state = {};
    localStorage.removeItem(storageKey);
    location.reload();
  }
};
render();
</script>
</body>
</html>
""".replace("__RECORDS__", records_json)
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    annotations_path = args.annotations_path.resolve()
    inventory_path = dataset_dir / "metadata" / "agrishow_2022_selected.csv"
    image_dir = dataset_dir / "images"
    thumbnail_dir = output_dir / "thumbnails"
    html_path = output_dir / "review.html"

    if args.thumbnail_width <= 0 or args.thumbnail_height <= 0:
        raise ValueError("Thumbnail dimensions must be positive.")
    rows = read_inventory(inventory_path)
    validate_images(rows, image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_annotation_template(annotations_path, rows, args.overwrite_annotations)
    make_thumbnails(
        rows,
        image_dir,
        thumbnail_dir,
        (args.thumbnail_width, args.thumbnail_height),
        args.overwrite_thumbnails,
    )
    write_review_html(html_path, rows, image_dir, thumbnail_dir)

    manifest = {
        "execution_id": "EX-020",
        "status": "prepared_for_first_manual_review",
        "target_id": TARGET_ID,
        "selected_images": len(rows),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "annotations_path": str(annotations_path),
        "review_html": str(html_path),
        "allowed_presence": sorted(ALLOWED_PRESENCE - {""}),
        "ground_truth_independent_of_model_scores": True,
    }
    manifest_path = output_dir / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EX-020 preparation: APPROVED")
    print(f"Images validated: {len(rows)}")
    print(f"Annotation template: {annotations_path}")
    print(f"Review gallery: {html_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
