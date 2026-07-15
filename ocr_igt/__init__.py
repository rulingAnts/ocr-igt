"""ocr_igt — OCR of handwritten Fayu interlinear notebooks into FLEx .flextext.

Two-step workflow:

    ocr-igt ocr   <images/pdfs>   ->  editable *.igt.json sidecars
    ocr-igt build <sidecars>      ->  a single .flextext for FLEx import

The intermediate JSON is meant to be hand-corrected before `build`, because
the source scans are sloppy and OCR (especially Tesseract) will be imperfect.
"""

__version__ = "0.1.0"
