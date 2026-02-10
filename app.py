from __future__ import annotations

import io
import os
import re
import tempfile
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

import zip_to_lens_pdf

import zipfile


# Streamlit puede recargar módulos en caliente; si el módulo falló a medio importar,
# algunas referencias pueden no existir. Este alias evita que el `except` reviente.
CancelledError = getattr(zip_to_lens_pdf, "CancelledError", RuntimeError)


def detect_tesseract_path() -> Optional[str]:
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    return None


def sanitize_zip_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = re.sub(r"[\\/:*?\"<>|]+", " ", stem).strip()
    return stem or "zip"


st.set_page_config(page_title="ZIP → PDF (Lens EN→ES)", layout="wide")

st.title("ZIP → PDFs (EN/ES) tipo Lens")

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    uploaded = st.file_uploader("Sube un .zip con imágenes", type=["zip"], accept_multiple_files=False)

    st.caption(
        "Genera 2 PDFs: uno original (EN) y otro traducido (ES). "
        "Si el OCR no detecta un título usable, el nombre se basa en el nombre del ZIP."
    )

    with st.expander("Opciones", expanded=True):
        generate_es = st.checkbox("Generar PDF traducido (ES)", value=True)
        title_override = st.text_input(
            "Título (opcional; si lo dejas vacío intenta OCR en la 1ª imagen)",
            value="",
            placeholder="Conceptos de Agentforce",
        )

        ocr_lang = st.text_input("OCR para texto (líneas)", value="eng")
        title_ocr_lang = st.text_input("OCR para título (1ª imagen)", value="eng+spa")

        st.markdown("**Rendimiento**")
        ocr_max_width = st.slider(
            "OCR max width (px)",
            min_value=0,
            max_value=3000,
            value=1600,
            step=100,
            help="Reduce el tamaño para OCR (más rápido). 0 = sin reescalar.",
        )
        workers = st.slider(
            "Workers (paralelo)",
            min_value=1,
            max_value=max(1, (os.cpu_count() or 4)),
            value=min(4, max(1, (os.cpu_count() or 4) // 2)),
            step=1,
            help="Procesos en paralelo para traducir páginas. Más = más rápido (hasta cierto punto).",
        )
        resume = st.checkbox(
            "Reutilizar páginas ya traducidas (resume)",
            value=True,
            help="Si vuelves a correr el mismo ZIP en la misma carpeta, evita reprocesar páginas ya generadas.",
        )

        auto_tess = detect_tesseract_path() or ""
        tesseract_cmd = st.text_input(
            "Ruta a tesseract.exe (necesario para ES)",
            value=auto_tess,
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )

        out_dir = st.text_input("Carpeta de salida", value=str(Path("output").resolve()))

    if uploaded is not None:
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded.getvalue()), "r") as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                imgs = [n for n in names if Path(n).suffix.lower() in zip_to_lens_pdf.IMAGE_EXTS]
                st.info(f"Páginas detectadas (imágenes soportadas): {len(imgs)}")
        except Exception:
            st.warning("No pude leer el ZIP para contar imágenes.")

with col_right:
    st.subheader("Ejecución")
    if "cancel_requested" not in st.session_state:
        st.session_state.cancel_requested = False

    run = st.button("Generar PDFs", type="primary", use_container_width=True, disabled=uploaded is None)
    cancel = st.button("Cancelar", use_container_width=True, disabled=uploaded is None)

    if cancel:
        st.session_state.cancel_requested = True

    if uploaded is None:
        st.info("Sube un ZIP para habilitar la ejecución.")

if run and uploaded is not None:
    zip_stem = sanitize_zip_stem(uploaded.name)

    # Reset cancel flag at start
    st.session_state.cancel_requested = False

    if generate_es and not tesseract_cmd:
        st.error("Para generar ES necesitas Tesseract. Indica la ruta a tesseract.exe.")
        st.stop()

    progress_bar = st.progress(0, text="Preparando…")
    status = st.empty()

    def progress_cb(stage: str, current: int, total: int):
        if st.session_state.get("cancel_requested"):
            raise CancelledError("Cancelado por el usuario")

        # Stage-specific messages
        if stage == "extract":
            progress_bar.progress(1, text="Extrayendo imágenes…")
            status.write("")
            return
        if stage == "title":
            progress_bar.progress(2, text="Detectando título (OCR)…")
            status.write("")
            return
        if stage == "pdf_en":
            progress_bar.progress(5, text="Generando PDF EN…")
            status.write("")
            return
        if stage == "translate":
            # Map translate progress into 5..95
            frac = 0 if total <= 0 else (current / total)
            pct = 5 + int(frac * 90)
            progress_bar.progress(min(95, max(5, pct)), text=f"Traduciendo (ES): {current}/{total} páginas…")
            status.write(f"Procesadas {current} de {total} páginas")
            return
        if stage == "pdf_es":
            progress_bar.progress(98, text="Generando PDF ES…")
            status.write("")
            return

    try:
        with tempfile.TemporaryDirectory(prefix="ui_zip_") as td:
            temp_zip = Path(td) / f"{zip_stem}.zip"
            temp_zip.write_bytes(uploaded.getvalue())

            outputs, meta = zip_to_lens_pdf.convert_zip_to_pdfs(
                temp_zip,
                Path(out_dir),
                tesseract_cmd=(tesseract_cmd if generate_es else None),
                generate_es=generate_es,
                ocr_lang=ocr_lang,
                title=title_override.strip(),
                title_ocr_lang=title_ocr_lang,
                ocr_max_width=int(ocr_max_width),
                workers=int(workers),
                resume=bool(resume),
                progress=progress_cb,
            )

        progress_bar.progress(100, text="Completado")
        st.success("Listo")

        st.subheader("Descargas")
        en_path = outputs.get("en")
        if en_path and en_path.exists():
            st.download_button(
                label=f"Descargar EN: {en_path.name}",
                data=en_path.read_bytes(),
                file_name=en_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
        es_path = outputs.get("es")
        if es_path and es_path.exists():
            st.download_button(
                label=f"Descargar ES: {es_path.name}",
                data=es_path.read_bytes(),
                file_name=es_path.name,
                mime="application/pdf",
                use_container_width=True,
            )

        st.caption(f"Salida: {(Path(out_dir) / 'pdf').resolve()}")

        st.subheader("Resumen")
        st.write(
            {
                "páginas": meta.get("pages"),
                "segundos": meta.get("seconds"),
                "workers": meta.get("workers"),
                "ocr_max_width": meta.get("ocr_max_width"),
                "resume": meta.get("resume"),
            }
        )

        translated_dir = Path(out_dir) / "translated"
        imgs = sorted(translated_dir.glob("*.png"))
        if imgs:
            st.subheader("Vista previa (traducidas)")
            st.image([str(imgs[0])], caption=[imgs[0].name], use_container_width=True)
    except CancelledError as exc:
        progress_bar.progress(100, text="Cancelado")
        st.warning(str(exc))
        st.stop()
    except Exception as exc:
        progress_bar.progress(100, text="Error")
        st.error(f"Error: {exc}")
        st.stop()

st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
