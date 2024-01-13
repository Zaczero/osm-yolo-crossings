{ pkgs ? import <nixpkgs> { }, ... }:

let
  shell = import ./shell.nix {
    isDevelopment = false;
  };

  python-venv = pkgs.buildEnv {
    name = "python-venv";
    paths = [
      (pkgs.runCommand "python-venv" { } ''
        mkdir -p $out/lib
        cp -r "${./.venv/lib/python3.11/site-packages}"/* $out/lib
      '')
    ];
  };

  etc-hosts = pkgs.buildEnv {
    name = "etc-hosts";
    paths = [
      (pkgs.writeTextDir "etc/hosts" ''
        127.0.0.1 localhost
        ::1 localhost
      '')
    ];
    pathsToLink = [ "/etc" ];
  };
in
with pkgs; dockerTools.buildLayeredImage {
  name = "docker.monicz.dev/osm-yolo-crossings";
  tag = "latest";
  maxLayers = 10;

  contents = shell.buildInputs ++ [ python-venv etc-hosts ];

  extraCommands = ''
    set -e
    mkdir app && cd app
    cp "${./.}"/*.py .
    cp -r "${./.}"/model .
    export PATH="${lib.makeBinPath shell.buildInputs}:$PATH"
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
