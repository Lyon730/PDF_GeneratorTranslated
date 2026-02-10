# ZIP de imágenes → PDF tipo Lens (EN→ES)

Convierte un `.zip` con imágenes en un PDF (1 página por imagen). Además, intenta:

- Detectar el texto **de la primera imagen** por OCR y usarlo como **título** del PDF.
- Crear una versión “tipo Google Lens”: conserva el fondo y **reemplaza** el texto en inglés por su traducción al español.

## Requisitos

- Python 3.10+ recomendado.
- **Tesseract OCR instalado** (pytesseract es solo el wrapper).
  - Windows: instala Tesseract (por ejemplo, el instalador oficial/UB Mannheim) y verifica que `tesseract.exe` esté en el PATH.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## UI (carga de ZIP)

Levanta una interfaz web local para subir el ZIP y descargar los PDFs:

```powershell
.\.venv\Scripts\python -m streamlit run .\app.py
```

La UI muestra:

- Número de páginas detectadas (imágenes en el ZIP)
- Barra de progreso y contador durante la traducción
- Botón para cancelar

## Uso

Ejemplo básico (genera **2 PDFs**: original EN + traducido ES):

```powershell
python .\zip_to_lens_pdf.py --zip ".\mi_zip_imagenes.zip" --out .\output
```

Si el título no se detecta bien por OCR, puedes forzarlo:

```powershell
python .\zip_to_lens_pdf.py --zip ".\mi_zip_imagenes.zip" --out .\output --title "Conceptos de Agentforce"
```

Opciones de rendimiento:

```powershell
python .\zip_to_lens_pdf.py --zip ".\mi_zip_imagenes.zip" --out .\output --workers 4 --ocr-max-width 1600
```

Si vuelves a ejecutar sobre el mismo `--out`, por defecto reutiliza páginas ya traducidas (resume). Para desactivar:

```powershell
python .\zip_to_lens_pdf.py --zip ".\mi_zip_imagenes.zip" --out .\output --no-resume
```

Si Tesseract no está en PATH:

```powershell
python .\zip_to_lens_pdf.py --zip ".\mi_zip_imagenes.zip" --out .\output --tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

Solo PDF con imágenes originales (solo EN; sin overlay):

```powershell
python .\zip_to_lens_pdf.py --zip ".\mi_zip_imagenes.zip" --out .\output --no-overlay
```

## Salidas

En la carpeta `--out`:

- `translated/` (PNG por imagen, con el texto traducido superpuesto)
- `pdf/` (PDF final)

Los PDFs se nombran como:

- `Título - nombreDelZip - EN.pdf`
- `Título - nombreDelZip - ES.pdf`

Si el OCR no detecta un título usable, se usa solo:

- `nombreDelZip - EN.pdf`
- `nombreDelZip - ES.pdf`

## Notas importantes

- La traducción usa `deep-translator` (Google Translate) **si está disponible**; puede requerir internet. Si falla, deja el texto original.
- El estilo “Lens” es una aproximación: cubre el texto detectado y dibuja la traducción en la misma zona.
