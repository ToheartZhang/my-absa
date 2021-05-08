import os

join = os.path.join

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = join(MAIN_PATH, 'data')
MODEL_PATH = join(MAIN_PATH, 'models')
