{ pkgs ? import <nixpkgs> { } }:

with pkgs; let
  shell = import ./shell.nix {
    inherit pkgs;
    isDocker = true;
  };

  python-venv = buildEnv {
    name = "python-venv";
    paths = [
      (runCommand "python-venv" { } ''
        mkdir -p $out/lib
        cp -r "${./.venv/lib/python3.11/site-packages}"/* $out/lib
      '')
    ];
  };

  etc-hosts = buildEnv {
    name = "etc-hosts";
    paths = [
      (writeTextDir "etc/hosts" ''
        127.0.0.1 localhost
        ::1 localhost
      '')
    ];
    pathsToLink = [ "/etc" ];
  };
in
dockerTools.buildLayeredImage {
  name = "docker.monicz.pl/osm-yolo-crossings";
  tag = "latest";
  maxLayers = 10;

  contents = shell.buildInputs ++ [ python-venv etc-hosts ];

  extraCommands = ''
    set -e
    mkdir app && cd app
    cp "${./.}"/LICENSE .
    cp "${./.}"/Makefile .
    cp "${./.}"/*.py .
    cp -r "${./.}"/model .
    ${shell.shellHook}
  '';

  config = {
    WorkingDir = "/app";
    Env = [
      "SSL_CERT_FILE=${cacert}/etc/ssl/certs/ca-bundle.crt"
      "LD_LIBRARY_PATH=${lib.makeLibraryPath shell.buildInputs}"
      "PYTHONPATH=${python-venv}/lib"
      "PYTHONUNBUFFERED=1"
      "PYTHONDONTWRITEBYTECODE=1"
    ];
    Volumes = {
      "/app/data" = { };
      "/.keras" = { };
    };
    Entrypoint = [ "python" "main.py" ];
    Cmd = [ ];
  };
}
