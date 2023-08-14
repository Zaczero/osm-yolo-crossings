.PHONY: update version dev-start dev-stop dev-logs

update:
	docker push $$(docker load < $$(nix-build --no-out-link) | sed -En 's/Loaded image: (\S+)/\1/p')

version:
	sed -i -r "s|VERSION = '([0-9.]+)'|VERSION = '\1.$$(date +%y%m%d)'|g" config.py

dev-start:
	docker compose -f docker-compose.dev.yml up -d

dev-stop:
	docker compose -f docker-compose.dev.yml down

dev-logs:
	docker compose -f docker-compose.dev.yml logs -f
