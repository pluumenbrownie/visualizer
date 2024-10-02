{
  description = "Flake used for music visualizer, made with Python.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      nixpkgs.overlays = [
        (self: super: {
          python312Packages = super.python312Packages.override {
            overrides = pyself: pysuper: {
              lmfit = pysuper.lmfit.overrideAttrs {doCheck = false;};
            };
          };
        }
        )
      ];

      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          python312Packages.python
          python312Packages.matplotlib
          python312Packages.numpy
          python312Packages.alive-progress
          python312Packages.scipy
          python312Packages.manim
          ffmpeg_7-full
        ];

      };

      doCheck = false;
    };
}
