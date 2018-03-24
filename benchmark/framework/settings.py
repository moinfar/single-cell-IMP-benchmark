import os

DEBUG = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(os.path.join(BASE_DIR, "files"))

CACHE_DIR = os.path.join(os.path.join(FILES_DIR, "cache"))
RESULTS_DIR = os.path.join(os.path.join(FILES_DIR, "results"))
STORAGE_DIR = os.path.join(os.path.join(FILES_DIR, "storage"))
IO_DIR = os.path.join(os.path.join(FILES_DIR, "io"))
