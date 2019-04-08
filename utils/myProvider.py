import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]