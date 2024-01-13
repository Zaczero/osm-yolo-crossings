{ isDevelopment ? true }:

let
  # Currently using nixpkgs-23.11-darwin
  # Get latest hashes from https://status.nixos.org/
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/a1fc22b2efdeba72b0519ac1548ec3c26e7f7b13.tar.gz") { };

  libraries' = with pkgs; [
    # Base libraries
    stdenv.cc.cc.lib
    zlib.out
  ];

  packages' = with pkgs; [
    # Base packages
    python311

    # Scripts
    # -- Misc
    (writeShellScriptBin "make-version" ''
      sed -i -r "s|VERSION = '([0-9.]+)'|VERSION = '\1.$(date +%y%m%d)'|g" config.py
    '')
  ] ++ lib.optionals isDevelopment [
    # Development packages
    poetry
    xorg.libX11

    # Scripts
    # -- Docker (dev)
    (writeShellScriptBin "dev-start" ''
      docker compose -f docker-compose.dev.yml up -d
    '')
    (writeShellScriptBin "dev-stop" ''
      docker compose -f docker-compose.dev.yml down
    '')
    (writeShellScriptBin "dev-logs" ''
      docker compose -f docker-compose.dev.yml logs -f
    '')
    (writeShellScriptBin "dev-clean" ''
      dev-stop
      rm -rf data/db
    '')

    # -- Misc
    (writeShellScriptBin "docker-build-push" ''
      set -e
      # Some data files require elevated permissions
      if [ -d "$PROJECT_DIR/data" ]; then
        image_path=$(sudo nix-build --no-out-link)
      else
        image_path=$(nix-build --no-out-link)
      fi
      docker push $(docker load < "$image_path" | sed -En 's/Loaded image: (\S+)/\1/p')
    '')
  ];

  shell' = with pkgs; ''
    export PROJECT_DIR="$(pwd)"
  '' + lib.optionalString isDevelopment ''
    [ ! -e .venv/bin/python ] && [ -h .venv/bin/python ] && rm -r .venv

    echo "Installing Python dependencies"
    export POETRY_VIRTUALENVS_IN_PROJECT=1
    poetry install --no-root --compile

    echo "Activating Python virtual environment"
    source .venv/bin/activate

    export LD_LIBRARY_PATH="${lib.makeLibraryPath libraries'}"

    # Development environment variables
    if [ -f .env ]; then
      set -o allexport
      source .env set
      +o allexport
    fi
  '' + lib.optionalString (!isDevelopment) ''
    make-version
  '';
in
pkgs.mkShell
{
  libraries = libraries';
  buildInputs = libraries' ++ packages';
  shellHook = shell';
}
