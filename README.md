# osm-yolo-crossings

![Python version](https://shields.monicz.dev/github/pipenv/locked/python-version/Zaczero/osm-yolo-crossings)
[![Project license](https://shields.monicz.dev/github/license/Zaczero/osm-yolo-crossings)](https://github.com/Zaczero/osm-yolo-crossings/blob/main/LICENSE)
[![Support my work](https://shields.monicz.dev/badge/%E2%99%A5%EF%B8%8F%20Support%20my%20work-purple)](https://monicz.dev/#support-my-work)
[![GitHub repo stars](https://shields.monicz.dev/github/stars/Zaczero/osm-yolo-crossings?style=social)](https://github.com/Zaczero/osm-yolo-crossings)

<img src="https://github.com/Zaczero/osm-yolo-crossings/raw/main/resources/card.png" width="40%">

🦓 OpenStreetMap, AI import tool for zebra crossings

## 💡 How it works

1. Queries OpenStreetMap (OSM) to find populated areas.
2. Retrieves [orthophoto imagery](https://www.geoportal.gov.pl/dane/ortofotomapa) for every road in the area.
3. Utilizes a [YOLOv8 model](https://ultralytics.com/yolov8) to detect regions of interest.
4. Performs binary classification on each region with a MobileNetV3Large model.
5. Checks historical OSM data to identify previously deleted crossings and avoid duplicates.
6. Imports new crossings into OSM.

<img src="https://github.com/Zaczero/osm-yolo-crossings/blob/main/resources/diagram-en.png?raw=true" width="80%">

🌟 Special thanks to [syntex](https://www.openstreetmap.org/user/syntex) for helping out with the dataset.

## Reference

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
