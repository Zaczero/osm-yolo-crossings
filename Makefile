.PHONY: update version start-dev stop-dev logs-dev

IMAGE_NAME=docker.monicz.pl/osm-budynki-orto-import

update:
	docker buildx build -t $(IMAGE_NAME) --push .

version:
	sed -i -r "s|VERSION = '([0-9.]+)'|VERSION = '\1.$$(date +%y%m%d)'|g" config.py

start-dev:
	docker compose -f docker-compose.dev.yml up -d

stop-dev:
	docker compose -f docker-compose.dev.yml down

logs-dev:
	docker compose -f docker-compose.dev.yml logs -f
