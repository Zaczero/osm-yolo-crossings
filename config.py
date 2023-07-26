import os
import secrets
from pathlib import Path

from tinydb import TinyDB

from orjson_storage import ORJSONStorage

SEED = 42

SAVE_IMG = os.getenv('SAVE_IMG', '0') == '1'
DRY_RUN = os.getenv('DRY_RUN', '0') == '1'
SKIP_CONSTRUCTION = os.getenv('SKIP_CONSTRUCTION', '1') == '1'

DAILY_IMPORT_SPEED = float(os.getenv('DAILY_IMPORT_SPEED', '300'))
SLEEP_AFTER_ONE_IMPORT = 86400 / DAILY_IMPORT_SPEED if DAILY_IMPORT_SPEED > 0 else 0
SLEEP_AFTER_GRID_ITER = float(os.getenv('SLEEP_DAYS_AFTER_GRID_ITER', '30')) * 24 * 3600

if DRY_RUN:
    print('🦺 TEST MODE 🦺')
else:
    print('🔴 PRODUCTION MODE 🔴')

# Dedicated instance unavailable? Pick one from the public list:
# https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances
OVERPASS_API_INTERPRETER = os.getenv('OVERPASS_API_INTERPRETER', 'https://overpass.monicz.dev/api/interpreter')

OSM_USERNAME = os.getenv('OSM_USERNAME')
OSM_PASSWORD = os.getenv('OSM_PASSWORD')
assert OSM_USERNAME and OSM_PASSWORD, 'OSM credentials not set'

SEARCH_RELATION = 49715  # Poland

CPU_COUNT = min(int(os.getenv('CPU_COUNT', '1')), len(os.sched_getaffinity(0)))

SCORER_VERSION = 1  # changing this will invalidate previous results

VERSION = '1.0'
CREATED_BY = f'osm-yolo-crossings {VERSION}'
WEBSITE = 'https://github.com/Zaczero/osm-yolo-crossings'
USER_AGENT = f'osm-yolo-crossings/{VERSION} (+{WEBSITE})'

CHANGESET_ID_PLACEHOLDER = f'__CHANGESET_ID_PLACEHOLDER__{secrets.token_urlsafe(8)}__'

DEFAULT_CHANGESET_TAGS = {
    'comment': 'Import przejść dla pieszych z ortofotomapy',
    'created_by': CREATED_BY,
    'import': 'yes',
    'source': 'aerial imagery',
    'website': WEBSITE,
    # TODO: 'website:import': 'https://wiki.openstreetmap.org/wiki/BDOT10k_buildings_import',
}

CONFIDENCE = 0.997

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = DATA_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

IMAGES_DIR = Path('images')
IMAGES_DIR.mkdir(exist_ok=True)

DATASET_DIR = Path('dataset')
DATASET_DIR.mkdir(exist_ok=True)

YOLO_DATASET_DIR = DATASET_DIR / 'YOLO'
YOLO_DATASET_DIR.mkdir(exist_ok=True)

ATTRIB_DATASET_DIR = DATASET_DIR / 'ATTRIB'
ATTRIB_DATASET_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path('model')
MODEL_DIR.mkdir(exist_ok=True)

YOLO_MODEL_PATH = MODEL_DIR / 'yolo.keras'
YOLO_MODEL_RESOLUTION = 224
YOLO_CONFIDENCE = 0.5

ATTRIB_MODEL_PATH = MODEL_DIR / 'attrib.keras'
ATTRIB_MODEL_RESOLUTION = 224
ATTRIB_CONFIDENCE = 0.5

DB_PATH = DATA_DIR / 'db.json'
DB = TinyDB(DB_PATH, storage=ORJSONStorage)
DB_ADDED = DB.table('added')
DB_ADDED_INDEX = DB.table('added_index')
DB_GRID = DB.table('grid')
