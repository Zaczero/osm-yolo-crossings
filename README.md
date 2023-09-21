# osm-yolo-crossings

![Python version](https://shields.monicz.dev/github/pipenv/locked/python-version/Zaczero/osm-yolo-crossings)
[![Project license](https://shields.monicz.dev/github/license/Zaczero/osm-yolo-crossings)](https://github.com/Zaczero/osm-yolo-crossings/blob/main/LICENSE)
[![Support my work](https://shields.monicz.dev/badge/%E2%99%A5%EF%B8%8F%20Support%20my%20work-purple)](https://monicz.dev/#support-my-work)
[![GitHub repo stars](https://shields.monicz.dev/github/stars/Zaczero/osm-yolo-crossings?style=social)](https://github.com/Zaczero/osm-yolo-crossings)

🦓 AI-powered OpenStreetMap tool for importing zebra crossings.

<img src="https://github.com/Zaczero/osm-yolo-crossings/raw/main/resources/card.png" alt="Project logo with YOLO text" width="40%">

## 💡 How it works

1. **Query OSM**: Finds populated areas on OpenStreetMap (OSM).
2. **Fetch Imagery**: Downloads [orthophoto imagery](https://www.geoportal.gov.pl/dane/ortofotomapa) for all roads in the query area.
3. **YOLOv8 Model**: Utilizes YOLOv8 for regions of interest detection.
4. **MobileNetV3Large**: Applies binary classification to validate detected regions.
5. **Data Integrity**: Checks against historical OSM data to avoid duplicates.
6. **OSM Import**: Automatically imports new crossings to OSM.

<img src="https://github.com/Zaczero/osm-yolo-crossings/blob/main/resources/diagram-en.png?raw=true" width="80%">

## 🛠️ Local Development

### Prerequisites

Before diving into development, make sure you have installed the [❄️ Nix](https://nixos.org/download) package manager.
Nix ensures seamless dependency management and a reproducible environment.

### ⠿ CUDA (NVIDIA GPU-mode)

Ideal for model development and training.
The application's primary functionality is optimized for CPU-mode.

TensorRT, a proprietary high-performance deep learning inference library by NVIDIA, is used by default. It can be safely disabled by commenting out **cudaPackages.tensorrt** in **cuda-shell.nix**.

```sh
# Install dependencies and packages
nix-shell cuda-shell.nix

# Configure .env file
cp .env.example .env

# Start up the database
make dev-start

# Done, now you can run the application
python main.py
```

### ⊡ CPU-mode

This mode is recommended for typical usage and production runs.

```sh
# Install dependencies and packages
nix-shell

# Configure .env file
cp .env.example .env

# Start up the database
make dev-start

# Done, now you can run the application
python main.py
```

## 📦 Deployment

Here is an example Docker Compose configuration. Note that you will need to build the Docker image first with **nix-build** _(see default.nix for options)_.

```yaml
version: "3"
services:
  db:
    image: mongo
    command: mongod --bind_ip_all

    volumes:
      - ./data/db:/data/db

  app:
    image: osm-yolo-crossings
    restart: unless-stopped

    environment:
      CPU_COUNT: 16
      MAX_TASKS_PER_CHILD: 300
      OSM_USERNAME: CHANGEME
      OSM_PASSWORD: CHANGEME
      MONGO_URL: mongodb://db:27017

    volumes:
      - ./data/app:/app/data
      - ./data/keras:/.keras
```

## 🌟 Special Thanks

Special thanks to [syntex](https://www.openstreetmap.org/user/syntex) for contributing to the dataset.

## ⚓ References

### Community discussion

https://community.openstreetmap.org/t/automatyczne-znakowanie-przejsc-dla-pieszych-oraz-przejazdow-rowerowych/101590/

### Data usage terms

https://www.geoportal.gov.pl/regulamin

https://wiki.openstreetmap.org/wiki/Pl:Geoportal.gov.pl

## Footer

### Contact me

https://monicz.dev/#get-in-touch

### Support my work

https://monicz.dev/#support-my-work

### License

This project is licensed under the GNU Affero General Public License v3.0.

The complete license text can be accessed in the repository at [LICENSE](https://github.com/Zaczero/osm-yolo-crossings/blob/main/LICENSE).
