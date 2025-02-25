# default.nix
with import <nixpkgs> {};
stdenv.mkDerivation {
    name = "transformers based nlp";
    allowUnfree = true;
    buildInputs = [ 
        pkg-config 
        gcc 
        python312Packages.torch
        python312Packages.keras
        python312Packages.tf-keras
        python312Packages.distutils
        python312Packages.tensorflow
        python312Packages.streamlit 
        python312Packages.transformers
        cudaPackages.cudatoolkit
    ];
    shellHook = ''
        export NIXPKGS_ALLOW_UNFREE=1
        fish
    '';
}
