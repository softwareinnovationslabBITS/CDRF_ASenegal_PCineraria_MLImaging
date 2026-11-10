"""Configuration for AS-vs-PC classification on NAS data."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NAS_ROOT = os.environ.get("DATA_DIR", "/media/NAS01/CDRF_BioCSIS/02_CDRF_DataCollection")

AS_BATCHES = [
    {
        "batch_name": "AS-BATCH-1 (250)",
        "species": "AS",
        "excel_path": os.path.join(
            NAS_ROOT,
            "AS-BATCH-1 (250-PLANTS)/AS-B-1 (FINAL EXCEL)/AS-B-1(FINAL EXCEL).xlsx",
        ),
        "image_dir": os.path.join(
            NAS_ROOT,
            "AS-BATCH-1 (250-PLANTS)/ALL PRE-PROCESSED IMAGES-AS-B-1/",
        ),
    },
    {
        "batch_name": "AS-BATCH-1 (OPT-65)",
        "species": "AS",
        "excel_path": os.path.join(NAS_ROOT, "AS-BATCH-1 (OPT)-65 PLANTS/Final_Excel-65pots.xlsx"),
        "image_dir": os.path.join(NAS_ROOT, "AS-BATCH-1 (OPT)-65 PLANTS/Final_PreProcessed/"),
    },
    {
        "batch_name": "AS-BATCH-2 (200)",
        "species": "AS",
        "excel_path": os.path.join(
            NAS_ROOT,
            "ALL EXCELS (AS & PC)/Acacia senegal all excels/AS-BATCH-2.xlsx",
        ),
        "image_dir": os.path.join(
            NAS_ROOT,
            "AS-BATCH-2 (200-PLANTS)/ALL PRE-PROCESSED IMAGES-AS-B-2/",
        ),
    },
    {
        "batch_name": "AS-BATCH-3 (256)",
        "species": "AS",
        "excel_path": os.path.join(NAS_ROOT, "AS-BATCH-3 (256-PLANTS)/AS-B-3.xlsx"),
        "image_dir": os.path.join(
            NAS_ROOT,
            "AS-BATCH-3 (256-PLANTS)/ALL PREPROCESSED IMAGES-AS-B-3/",
        ),
    },
]

PC_BATCHES = [
    {
        "batch_name": "PC-BATCH-1 (220)",
        "species": "PC",
        "excel_path": os.path.join(
            NAS_ROOT,
            "PC-BATCH-1 (220-PLANTS)/PC-B-1 (FINAL EXCEL)/PC-B-1(FINAL EXCEL).xlsx",
        ),
        "image_dir": os.path.join(
            NAS_ROOT,
            "PC-BATCH-1 (220-PLANTS)/ALL PRE-PROCESSED IMAGES-PC-B-1/",
        ),
    },
    {
        "batch_name": "PC-BATCH-2 (200)",
        "species": "PC",
        "excel_path": os.path.join(NAS_ROOT, "PC-BATCH-2 (200-PLANTS)/Pc-B-2 excel.xlsx"),
        "image_dir": os.path.join(NAS_ROOT, "PC-BATCH-2 (200-PLANTS)/ALL PRE-PROCESSED IMAGES-PC-BATCH-2/"),
    },
    {
        "batch_name": "PC-BATCH-3 (240)",
        "species": "PC",
        "excel_path": os.path.join(NAS_ROOT, "PC-BATCH-3 (240 PLANTS)/Pc-B-3.xlsx"),
        "image_dir": os.path.join(NAS_ROOT, "PC-BATCH-3 (240 PLANTS)/ALL PRE-PROCESSED IMAGES-PC-B-3/"),
    },
]

BATCH_CONFIGS = AS_BATCHES + PC_BATCHES

IMAGE_SIZE = 224
SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 10
LR = 1e-3

TARGET_CLASSES = ["AS", "PC"]
CLASS_TO_LABEL = {"AS": 0, "PC": 1}

IMAGE_NAME_CANDIDATES = [
    "ImageName",
    "Image",
    "image",
    "Image Name",
    "Filename",
    "File",
    "Image_Name",
]

CACHE_IMAGES = os.path.join(BASE_DIR, "cache_images.npy")
CACHE_LABELS = os.path.join(BASE_DIR, "cache_labels.npy")
CACHE_FILENAMES = os.path.join(BASE_DIR, "cache_filenames.npy")
CACHE_BATCHES = os.path.join(BASE_DIR, "cache_batches.npy")
SPLIT_FILE = os.path.join(BASE_DIR, "split_indices.npz")
