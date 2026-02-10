from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import median
from time import perf_counter
from typing import Callable, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

try:
    import pytesseract
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "pytesseract no está disponible. Instala dependencias: pip install -r requirements.txt"
    ) from exc

try:
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "reportlab no está disponible. Instala dependencias: pip install -r requirements.txt"
    ) from exc


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


ProgressCallback = Optional[Callable[[str, int, int], None]]


class CancelledError(RuntimeError):
    pass


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def sanitize_filename(name: str, fallback: str = "output") -> str:
    name = (name or "").strip()
    if not name:
        return fallback
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[\\/:*?\"<>|]+", " ", name).strip()
    return name[:120] if name else fallback


def is_usable_title(title: str) -> bool:
    """Heurística para filtrar títulos basura del OCR.

    Si el OCR devuelve algo muy corto o sin suficientes caracteres alfanuméricos,
    preferimos NO usarlo para nombrar archivos.
    """

    t = (title or "").strip()
    if not t:
        return False
    alnum = sum(ch.isalnum() for ch in t)
    # Threshold conservador: evita nombres como "EN" o fragmentos raros.
    return alnum >= 6


def pick_first_title_line(ocr_text: str) -> str:
    lines = [ln.strip() for ln in (ocr_text or "").splitlines()]
    # Keep any line with some meaningful characters; OCR may include accents or non A-Z text.
    lines = [ln for ln in lines if ln and any(ch.isalnum() for ch in ln)]
    if not lines:
        return ""
    # Prefer a longer, more title-like line.
    return sorted(
        lines,
        key=lambda x: (sum(ch.isalnum() for ch in x), len(x)),
        reverse=True,
    )[0]


def extract_title_from_first_image(img: Image.Image, lang: str) -> str:
    """Heurística: intenta detectar el título priorizando texto grande en la parte superior."""
    rgb = img.convert("RGB")
    w, h = rgb.size

    # 1) Intento robusto con image_to_data: busca tokens con mayor altura en el tercio superior.
    try:
        data = pytesseract.image_to_data(rgb, lang=lang, output_type=pytesseract.Output.DICT)
        n = len(data.get("text", []))
        candidates: list[tuple[float, str]] = []
        for i in range(n):
            word = (data["text"][i] or "").strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if conf < 40:
                continue
            top = int(data["top"][i])
            height = int(data["height"][i])
            if top > int(h * 0.45):
                continue
            if not any(ch.isalnum() for ch in word):
                continue
            # Score: larger text and higher confidence
            score = float(height) * 2.0 + conf
            candidates.append((score, word))

        if candidates:
            # Join top words (best score) and nearby ones in the same top region.
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, _ = candidates[0]
            top_words = [wrd for sc, wrd in candidates if sc >= best_score * 0.75]
            title_guess = " ".join(top_words)
            title_guess = re.sub(r"\s+", " ", title_guess).strip()
            if title_guess:
                return title_guess
    except Exception:
        pass

    # 2) Fallback: OCR completo a texto y escoge una línea "título".
    try:
        ocr_text = pytesseract.image_to_string(rgb, lang=lang)
        return pick_first_title_line(ocr_text)
    except Exception:
        return ""


def get_default_font_path() -> Optional[str]:
    win = os.environ.get("WINDIR", "C:\\Windows")
    candidates = [
        os.path.join(win, "Fonts", "arial.ttf"),
        os.path.join(win, "Fonts", "segoeui.ttf"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def try_translate_en_to_es(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # Optional dependency
    try:
        from deep_translator import GoogleTranslator  # type: ignore

        return GoogleTranslator(source="en", target="es").translate(text)
    except Exception:
        return text


@dataclass(frozen=True)
class LineBox:
    left: int
    top: int
    right: int
    bottom: int
    text: str
    conf: float


def ocr_line_boxes(img: Image.Image, lang: str = "eng") -> list[LineBox]:
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

    n = len(data.get("text", []))
    groups: dict[tuple[int, int, int], dict] = {}

    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if conf < 50:
            continue

        # Skip very noisy tokens
        if not re.search(r"[A-Za-z]", word):
            continue

        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))

        left = int(data["left"][i])
        top = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        right = left + w
        bottom = top + h

        if key not in groups:
            groups[key] = {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "text": [word],
                "conf": [conf],
            }
        else:
            g = groups[key]
            g["left"] = min(g["left"], left)
            g["top"] = min(g["top"], top)
            g["right"] = max(g["right"], right)
            g["bottom"] = max(g["bottom"], bottom)
            g["text"].append(word)
            g["conf"].append(conf)

    line_boxes: list[LineBox] = []
    for g in groups.values():
        text = " ".join(g["text"]).strip()
        conf = float(mean(g["conf"])) if g["conf"] else 0.0
        if text:
            line_boxes.append(
                LineBox(
                    left=int(g["left"]),
                    top=int(g["top"]),
                    right=int(g["right"]),
                    bottom=int(g["bottom"]),
                    text=text,
                    conf=conf,
                )
            )

    # Stable ordering: top-to-bottom then left-to-right
    line_boxes.sort(key=lambda b: (b.top, b.left))
    return line_boxes


def resize_for_ocr(img_rgb: Image.Image, max_width: int) -> tuple[Image.Image, float, float]:
    """Devuelve (img_para_ocr, scale_x, scale_y).

    scale_x/scale_y permiten reescalar coordenadas OCR hacia la imagen original.
    """

    if max_width <= 0:
        return img_rgb, 1.0, 1.0

    w, h = img_rgb.size
    if w <= max_width:
        return img_rgb, 1.0, 1.0

    new_w = int(max_width)
    new_h = int(h * (new_w / w))
    resized = img_rgb.resize((new_w, new_h), resample=Image.BILINEAR)
    scale_x = w / new_w
    scale_y = h / new_h
    return resized, scale_x, scale_y


def _flattened_pixels(img: Image.Image) -> list[tuple[int, int, int]]:
    # Pillow 12 deprecates getdata(); use get_flattened_data() when available.
    if hasattr(img, "get_flattened_data"):
        return list(img.get_flattened_data())  # type: ignore[attr-defined]
    return list(img.getdata())


def median_background_color(img_rgb: Image.Image, box: tuple[int, int, int, int], pad: int = 3) -> tuple[int, int, int]:
    w, h = img_rgb.size
    l, t, r, b = box
    l2 = max(l - pad, 0)
    t2 = max(t - pad, 0)
    r2 = min(r + pad, w)
    b2 = min(b + pad, h)

    # Collect thin border samples around the box using PIL crops (no NumPy dependency).
    samples: list[tuple[int, int, int]] = []
    if t2 < t:
        samples.extend(_flattened_pixels(img_rgb.crop((l2, t2, r2, t))))
    if b < b2:
        samples.extend(_flattened_pixels(img_rgb.crop((l2, b, r2, b2))))
    if l2 < l:
        samples.extend(_flattened_pixels(img_rgb.crop((l2, t2, l, b2))))
    if r < r2:
        samples.extend(_flattened_pixels(img_rgb.crop((r, t2, r2, b2))))

    if not samples:
        samples = _flattened_pixels(img_rgb.crop((l2, t2, r2, b2)))

    if not samples:
        return (255, 255, 255)

    rs = [p[0] for p in samples]
    gs = [p[1] for p in samples]
    bs = [p[2] for p in samples]
    return (int(median(rs)), int(median(gs)), int(median(bs)))


def draw_translated_overlay(
    img: Image.Image,
    lang_ocr: str = "eng",
    font_path: Optional[str] = None,
    ocr_max_width: int = 1600,
) -> tuple[Image.Image, list[LineBox]]:
    base = img.convert("RGB")
    draw = ImageDraw.Draw(base)

    ocr_img, scale_x, scale_y = resize_for_ocr(base, max_width=ocr_max_width)
    boxes_small = ocr_line_boxes(ocr_img, lang=lang_ocr)
    # Scale boxes back to original image coordinates
    boxes = [
        LineBox(
            left=int(b.left * scale_x),
            top=int(b.top * scale_y),
            right=int(b.right * scale_x),
            bottom=int(b.bottom * scale_y),
            text=b.text,
            conf=b.conf,
        )
        for b in boxes_small
    ]

    if font_path is None:
        font_path = get_default_font_path()

    for box in boxes:
        bbox = (box.left, box.top, box.right, box.bottom)
        bg = median_background_color(base, bbox, pad=4)

        # Cover original text region
        draw.rectangle(bbox, fill=bg)

        translated = try_translate_en_to_es(box.text)
        if not translated:
            translated = box.text

        # Choose font size to fit height, then shrink if too wide
        target_h = max(10, int((box.bottom - box.top) * 0.9))
        if font_path and os.path.exists(font_path):
            size = target_h
            font = ImageFont.truetype(font_path, size=size)
            max_w = max(10, box.right - box.left)
            while size > 8:
                w_text = draw.textlength(translated, font=font)
                if w_text <= max_w:
                    break
                size -= 1
                font = ImageFont.truetype(font_path, size=size)
        else:
            font = ImageFont.load_default()

        # Text color: choose dark/white based on background luminance
        lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
        fg = (0, 0, 0) if lum > 140 else (255, 255, 255)

        draw.text((box.left, box.top), translated, fill=fg, font=font)

    return base, boxes


def extract_images_from_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        # Keep only known image extensions
        names = [n for n in names if Path(n).suffix.lower() in IMAGE_EXTS]
        names.sort(key=natural_sort_key)

        extracted: list[Path] = []
        for name in names:
            target = out_dir / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)

    return extracted


def _translate_one_image(
    img_path: str,
    out_path: str,
    *,
    tesseract_cmd: Optional[str],
    ocr_lang: str,
    ocr_max_width: int,
) -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    img = Image.open(img_path).convert("RGB")
    out_img, _ = draw_translated_overlay(img, lang_ocr=ocr_lang, ocr_max_width=ocr_max_width)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path, format="PNG", optimize=True)
    return out_path


def convert_zip_to_pdfs(
    zip_path: Path,
    out_dir: Path,
    *,
    tesseract_cmd: Optional[str] = None,
    generate_es: bool = True,
    ocr_lang: str = "eng",
    title: str = "",
    title_ocr_lang: str = "eng+spa",
    ocr_max_width: int = 1600,
    workers: int = 1,
    resume: bool = True,
    progress: ProgressCallback = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Convierte un ZIP a PDFs EN/ES.

    progress(stage: str, current: int, total: int) se llama durante el proceso.
    Stages: extract, title, pdf_en, translate, pdf_es
    """

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    if generate_es and not is_tesseract_available():
        raise RuntimeError(
            "No se encontró Tesseract OCR. Instálalo o indica --tesseract-cmd."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = out_dir / "pdf"
    translated_dir = out_dir / "translated"

    def tick(stage: str, current: int, total: int):
        if progress is not None:
            progress(stage, current, total)

    started = perf_counter()

    with tempfile.TemporaryDirectory(prefix="zip_images_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        tick("extract", 0, 1)
        images = extract_images_from_zip(zip_path, temp_dir_path)
        tick("extract", 1, 1)
        if not images:
            raise ValueError("El ZIP no contiene imágenes soportadas.")

        pages = len(images)

        # Title (override > OCR). If OCR is not reliable, fall back to ZIP name only.
        title = (title or "").strip()
        title_from_override = bool(title)

        if not title:
            tick("title", 0, 1)
            try:
                first_img = Image.open(images[0]).convert("RGB")
                title = extract_title_from_first_image(first_img, lang=title_ocr_lang)
            except Exception:
                title = ""
            tick("title", 1, 1)

        if not title_from_override and not is_usable_title(title):
            title = ""

        safe_title = sanitize_filename(title, fallback="")
        zip_stem = sanitize_filename(zip_path.stem, fallback="zip")
        base_name = f"{safe_title} - {zip_stem}" if safe_title else zip_stem

        # 1) PDF original (EN)
        tick("pdf_en", 0, 1)
        pdf_en_path = pdf_dir / f"{base_name} - EN.pdf"
        build_pdf_from_images(images, pdf_path=pdf_en_path, title=f"{base_name} - EN")
        tick("pdf_en", 1, 1)

        outputs: dict[str, Path] = {"en": pdf_en_path}

        if generate_es:
            translated_dir.mkdir(parents=True, exist_ok=True)
            translated_images: list[Path] = []
            total = pages

            # Precompute output paths and apply resume
            jobs: list[tuple[int, Path, Path]] = []
            for i, img_path in enumerate(images, start=1):
                out_path = translated_dir / f"{i:03d}_{img_path.stem}.png"
                if resume and out_path.exists():
                    translated_images.append(out_path)
                else:
                    jobs.append((i, img_path, out_path))

            done = len(translated_images)
            tick("translate", done, total)

            # Parallel translation
            max_workers = max(1, int(workers or 1))
            if jobs:
                if max_workers == 1:
                    for (i, img_path, out_path) in jobs:
                        _translate_one_image(
                            str(img_path),
                            str(out_path),
                            tesseract_cmd=tesseract_cmd,
                            ocr_lang=ocr_lang,
                            ocr_max_width=ocr_max_width,
                        )
                        translated_images.append(out_path)
                        done += 1
                        tick("translate", done, total)
                else:
                    futures = []
                    with ProcessPoolExecutor(max_workers=max_workers) as ex:
                        for (_i, img_path, out_path) in jobs:
                            futures.append(
                                ex.submit(
                                    _translate_one_image,
                                    str(img_path),
                                    str(out_path),
                                    tesseract_cmd=tesseract_cmd,
                                    ocr_lang=ocr_lang,
                                    ocr_max_width=ocr_max_width,
                                )
                            )
                        for fut in as_completed(futures):
                            _ = fut.result()
                            done += 1
                            tick("translate", done, total)

                    # Ensure order in PDF: by filename prefix index
                    translated_images = sorted(
                        list(translated_dir.glob("*.png")),
                        key=lambda p: p.name,
                    )
            else:
                # All images reused
                translated_images = sorted(translated_images, key=lambda p: p.name)

            tick("pdf_es", 0, 1)
            pdf_es_path = pdf_dir / f"{base_name} - ES.pdf"
            build_pdf_from_images(translated_images, pdf_path=pdf_es_path, title=f"{base_name} - ES")
            tick("pdf_es", 1, 1)
            outputs["es"] = pdf_es_path

        meta: dict[str, object] = {
            "pages": pages,
            "seconds": round(perf_counter() - started, 3),
            "zip": str(zip_path),
            "base_name": base_name,
            "workers": int(workers or 1),
            "ocr_max_width": int(ocr_max_width or 0),
            "resume": bool(resume),
            "generated_es": bool(generate_es),
        }
        return outputs, meta


def build_pdf_from_images(image_paths: Iterable[Path], pdf_path: Path, title: str) -> None:
    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("No se encontraron imágenes para generar el PDF")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    first_img = Image.open(image_paths[0]).convert("RGB")
    w0, h0 = first_img.size

    c = canvas.Canvas(str(pdf_path), pagesize=(w0, h0))
    if title:
        c.setTitle(title)

    for p in image_paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        c.setPageSize((w, h))
        c.drawImage(ImageReader(img), 0, 0, width=w, height=h)
        c.showPage()

    c.save()


def is_tesseract_available() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convierte un ZIP de imágenes a PDF; opcionalmente traduce EN->ES estilo Lens."
    )
    parser.add_argument("--zip", dest="zip_path", required=True, help="Ruta al archivo .zip")
    parser.add_argument("--out", dest="out_dir", default="output", help="Carpeta de salida")
    parser.add_argument(
        "--tesseract-cmd",
        dest="tesseract_cmd",
        default=None,
        help="Ruta a tesseract.exe si no está en PATH",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Genera SOLO el PDF original (EN) con las imágenes originales (sin traducción/overlay)",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Idioma(s) de OCR para Tesseract (ej: eng, eng+spa)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Fuerza el título del PDF (si el OCR no lo detecta bien)",
    )
    parser.add_argument(
        "--title-ocr-lang",
        default="eng+spa",
        help="Idioma(s) de OCR para detectar el título en la primera imagen (default: eng+spa)",
    )
    parser.add_argument(
        "--ocr-max-width",
        type=int,
        default=1600,
        help="Ancho máximo (px) para OCR por página (acelera); 0 desactiva (default: 1600)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Número de procesos para traducir páginas en paralelo (default: 1)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="No reutiliza imágenes traducidas existentes en output/translated",
    )

    args = parser.parse_args(argv)

    zip_path = Path(args.zip_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not zip_path.exists():
        print(f"No existe: {zip_path}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        outputs, _meta = convert_zip_to_pdfs(
            zip_path,
            out_dir,
            tesseract_cmd=args.tesseract_cmd,
            generate_es=(not args.no_overlay),
            ocr_lang=args.ocr_lang,
            title=(args.title or ""),
            title_ocr_lang=args.title_ocr_lang,
            ocr_max_width=args.ocr_max_width,
            workers=args.workers,
            resume=(not args.no_resume),
        )
    except RuntimeError as exc:
        print(
            f"{exc}\n\n"
            "Para usar el modo 'Lens' necesitas instalar Tesseract y/o indicar la ruta con --tesseract-cmd, por ejemplo:\n"
            "  --tesseract-cmd \"C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe\"\n"
            "Alternativa: ejecuta con --no-overlay para generar solo el PDF original (EN).",
            file=sys.stderr,
        )
        return 4
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 5

    print(f"OK (EN): {outputs['en']}")
    if 'es' in outputs:
        print(f"OK (ES): {outputs['es']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
