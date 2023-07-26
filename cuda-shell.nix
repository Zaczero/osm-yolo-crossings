{ pkgs ? import <nixpkgs> { } }:

let
  base = import ./shell.nix { inherit pkgs; };
in
pkgs.mkShell rec {
  buildInputs = with pkgs; base.buildInputs ++ [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.tensorrt
  ];

  shellHook = with pkgs; ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${cudaPackages.cudatoolkit}/lib64:$LD_LIBRARY_PATH"
    export CUDA_DIR="${cudaPackages.cudatoolkit}"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}"
  '' + base.shellHook;
}
