import os
from pathlib import Path

from pymongo import ASCENDING, GEOSPHERE, MongoClient
from tinydb import TinyDB

from orjson_storage import ORJSONStorage

SEED = 42

SAVE_IMG = os.getenv('SAVE_IMG', '0') == '1'
DRY_RUN = os.getenv('DRY_RUN', '0') == '1'

DAILY_IMPORT_SPEED = float(os.getenv('DAILY_IMPORT_SPEED', '300'))
MIN_IMPORT_SIZE = int(os.getenv('MIN_IMPORT_SIZE', '30'))
MIN_SLEEP_AFTER_IMPORT = float(os.getenv('MIN_SLEEP_AFTER_IMPORT', '300'))  # let overpass-api update
SLEEP_AFTER_GRID_ITER = float(os.getenv('SLEEP_DAYS_AFTER_GRID_ITER', '30')) * 24 * 3600
PROCESS_NICE = int(os.getenv('PROCESS_NICE', '15'))
RETRY_TIME_LIMIT = float(os.getenv('RETRY_TIME_LIMIT', '10800'))  # 3h
BACKLOG_FACTOR = int(os.getenv('BACKLOG_FACTOR', '4'))

if DRY_RUN:
    print('🦺 TEST MODE 🦺')
else:
    print('🔴 PRODUCTION MODE 🔴')

# Dedicated instance unavailable? Pick one from the public list:
# https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances
OVERPASS_API_INTERPRETER = os.getenv('OVERPASS_API_INTERPRETER', 'https://overpass.monicz.dev/api/interpreter')

OSM_USERNAME = os.getenv('OSM_USERNAME')
OSM_PASSWORD = os.getenv('OSM_PASSWORD')

if not OSM_PASSWORD or not OSM_PASSWORD:
    print('⚠️ OpenStreetMap credentials are not set')

SEARCH_RELATION = 49715  # Poland

CPU_COUNT = min(int(os.getenv('CPU_COUNT', '1')), len(os.sched_getaffinity(0)))

SCORER_VERSION = 1  # changing this will invalidate previous results

VERSION = '1.3'
NAME = 'osm-yolo-crossings'
CREATED_BY = f'{NAME} {VERSION}'
WEBSITE = 'https://github.com/Zaczero/osm-yolo-crossings'
USER_AGENT = f'{NAME}/{VERSION} (+{WEBSITE})'

CHANGESET_ID_PLACEHOLDER = '__CHANGESET_ID_PLACEHOLDER__'

DEFAULT_CHANGESET_TAGS = {
    'comment': 'Import przejść dla pieszych z ortofotomapy',
    'created_by': CREATED_BY,
    'import': 'yes',
    'source': 'aerial imagery',
    'website': WEBSITE,
    # TODO: 'website:import': 'https://wiki.openstreetmap.org/wiki/BDOT10k_buildings_import',
}

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
YOLO_CONFIDENCE = 0.4

ATTRIB_POSITION_EXTEND = 9  # meters
ATTRIB_MODEL_PATH = MODEL_DIR / 'attrib.h5'
ATTRIB_MODEL_RESOLUTION = 224
ATTRIB_PRECISION = 0.998
ATTRIB_CONFIDENCE = 0.995

ATTRIB_NUM_CLASSES = 2
# ATTRIB_PRECISION = (
#     0.995,  # valid
#     0.995,  # signals
# )
# ATTRIB_CONFIDENCES = (
#     0.730,  # valid
#     (0.060, 1.111),  # signals (..., 0.995)
# )

GRID_FILTER_BUILDING_DISTANCE = 1000  # meters
GRID_FILTER_ROAD_INTERPOLATE = 10  # meters

_RANGE = 5  # meters
ADDED_SEARCH_RADIUS = _RANGE + 0.05  # meters
CROSSING_BOX_EXTEND = 15  # meters

# see for picking good values: https://www.openstreetmap.org/node/4464489698
# maximum distance from the center of the box to a road
BOX_VALID_MAX_CENTER_DISTANCE = _RANGE  # meters

# maximum angle between raods before considering the case too complex
BOX_VALID_MAX_ROAD_ANGLE = 40  # degrees

# maximum intersection count for a perpendicular section
BOX_VALID_MAX_ROAD_COUNT = 2

# minimum distance between crossings (circular)
BOX_VALID_MIN_CROSSING_DISTANCE = _RANGE  # meters

# minimum distance between crossings (cone)
BOX_VALID_MIN_CROSSING_DISTANCE_CONE = 15  # meters
BOX_VALID_MIN_CROSSING_DISTANCE_CONE_ANGLE = 30  # degrees

# maximum distance to reuse an existing node
NODE_MERGE_THRESHOLD = 1.5  # meters

# maximum distance to reuse an existing node, if it's also used by a path/footway/...
NODE_MERGE_THRESHOLD_PRIORITY = _RANGE  # meters

DB_PATH = DATA_DIR / 'db.json'
DB = TinyDB(DB_PATH, storage=ORJSONStorage)
DB_GRID = DB.table('grid')

MONGO_URL = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
MONGO = MongoClient(MONGO_URL)
MONGO_DB = MONGO[NAME]
MONGO_ADDED = MONGO_DB['added']

if MONGO_URL != 'IGNORE':
    MONGO_ADDED.create_index([
        ('scorer_version', ASCENDING),
        ('reason', ASCENDING),
        ('position', GEOSPHERE)
    ])
