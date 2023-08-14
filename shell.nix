{ pkgs ? import <nixpkgs> { }
, isDocker ? false
}:

with pkgs; let
  commonBuildInputs = [
    busybox
    stdenv.cc.cc.lib
    gnumake
    gnused
    python311
    zlib.out
    util-linux # lscpu
    cacert
  ];

  devBuildInputs = [
    pipenv
    xorg.libX11
  ];

  commonShellHook = ''
  '';

  devShellHook = ''
    export PIPENV_VENV_IN_PROJECT=1
    export PIPENV_VERBOSITY=-1
    [ ! -f .venv/bin/activate ] && pipenv sync --dev
    case $- in *i*) exec pipenv shell --fancy;; esac
  '';

  dockerShellHook = ''
    make version
  '';
in
pkgs.mkShell {
  buildInputs = commonBuildInputs ++ (if isDocker then [ ] else devBuildInputs);
  shellHook = commonShellHook + (if isDocker then dockerShellHook else devShellHook);
}
