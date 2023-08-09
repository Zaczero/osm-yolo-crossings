.PHONY: update version dev-start dev-stop dev-logs

IMAGE_NAME=docker.monicz.pl/osm-budynki-orto-import

update:
	docker buildx build -t $(IMAGE_NAME) --push .

version:
	sed -i -r "s|VERSION = '([0-9.]+)'|VERSION = '\1.$$(date +%y%m%d)'|g" config.py

dev-start:
	docker compose -f docker-compose.dev.yml up -d

dev-stop:
	docker compose -f docker-compose.dev.yml down

dev-logs:
	docker compose -f docker-compose.dev.yml logs -f
