{}:

let
  shell = import ./shell.nix { };

  # Currently using nixpkgs-23.11-darwin
  # Get latest hashes from https://status.nixos.org/
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/a1fc22b2efdeba72b0519ac1548ec3c26e7f7b13.tar.gz") { };

  packages' = with pkgs; [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.tensorrt
  ];

  shell' = with pkgs; ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath packages'}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${cudaPackages.cudatoolkit}/lib64:$LD_LIBRARY_PATH"
    export CUDA_DIR="${cudaPackages.cudatoolkit}"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}"
  '';
in
pkgs.mkShell
{
  buildInputs = shell.buildInputs ++ packages';
  shellHook = shell.shellHook + shell';
}
