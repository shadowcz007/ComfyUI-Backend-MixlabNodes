import importlib.util
import glob
import os
import sys
import __main__
import filecmp
import shutil

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# python = sys.executable

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

nodes = get_ext_dir("nodes")
files = os.listdir(nodes)
for file in files:
    if not file.endswith(".py"):
        continue
    name = os.path.splitext(file)[0]

    imported_module = importlib.import_module(".nodes.{}".format(name), __name__)
    try:
        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}
    except:
        pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
