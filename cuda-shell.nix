{ pkgs ? import <nixpkgs> { } }:

with pkgs; let
  shell = import ./shell.nix { inherit pkgs; };
in
mkShell rec {
  buildInputs = shell.buildInputs ++ [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.tensorrt
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${cudaPackages.cudatoolkit}/lib64:$LD_LIBRARY_PATH"
    export CUDA_DIR="${cudaPackages.cudatoolkit}"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}"
  '' + shell.shellHook;
}
