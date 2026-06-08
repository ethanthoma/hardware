{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devshell.url = "github:numtide/devshell";
  };

  outputs =
    inputs@{ self, ... }:

    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.devshell.flakeModule ];

      systems = [ "x86_64-linux" ];

      perSystem =
        { pkgs, ... }:
        let

          python =
            let
              file = inputs.self + "/.python-version";
              version = if builtins.pathExists file then builtins.readFile file else "3.13";
              major = builtins.substring 0 1 version;
              minor = builtins.substring 2 2 version;
              packageName = "python${major}${minor}";
            in
            pkgs.${packageName} or pkgs.python314;
        in
        {
          devshells.default = {
            packages = [
              pkgs.ruff
              pkgs.yamlfmt
              pkgs.pyrefly
              pkgs.gfortran
              pkgs.openblas
              pkgs.pkg-config
              pkgs.ninja
            ];

            env = [
              {
                name = "UV_PYTHON_DOWNLOADS";
                value = "never";
              }
              {
                name = "UV_PYTHON";
                value = python.interpreter;
              }
              {
                name = "PYTHONPATH";
                unset = true;
              }
              {
                name = "UV_NO_SYNC";
                value = "1";
              }
              {
                name = "PATH";
                eval = "/home/ethanthoma/.local/bin:$PATH";
              }
            ];

            commands = [
              { package = pkgs.uv; }
              { package = pkgs.tokei; }
              { package = pkgs.yosys; }
              { package = pkgs.nextpnr; }
              { package = pkgs.trellis; }
              { package = pkgs.openfpgaloader; }
            ];
          };
        };
    };

  nixConfig = {
    extra-substituters = [ "https://nix-community.cachix.org" ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };
}
