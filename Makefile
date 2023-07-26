.PHONY: update version

IMAGE_NAME=docker.monicz.pl/osm-budynki-orto-import

update:
	docker buildx build -t $(IMAGE_NAME) --push .

version:
	sed -i -r "s|VERSION = '([0-9.]+)'|VERSION = '\1.$$(date +%y%m%d)'|g" config.py
