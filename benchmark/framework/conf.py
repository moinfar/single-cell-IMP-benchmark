import importlib
import sys

import lazy_object_proxy


def load_settings():
    """
    A function which loads settings from settings.py.
    :return:
    """
    settings_path = "framework.settings"
    try:
        loaded_settings = importlib.import_module(settings_path)
        print("Settings loaded successfully.", file=sys.stderr)
    except ImportError as e:
        print("Failed to load settings.", file=sys.stderr)
        raise e

    return loaded_settings


# Load settings once
settings = lazy_object_proxy.Proxy(load_settings)
