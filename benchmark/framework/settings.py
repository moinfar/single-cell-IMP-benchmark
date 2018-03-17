import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(os.path.join(BASE_DIR, "cache"))
RESULTS_DIR = os.path.join(os.path.join(BASE_DIR, "results"))
