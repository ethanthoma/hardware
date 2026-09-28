{
  description = "bfloat16 MMA unit in Amaranth HDL";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devshell.url = "github:numtide/devshell";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ self, ... }:

    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.devshell.flakeModule ];

      systems = [ "x86_64-linux" ];

      perSystem =
        { pkgs, lib, ... }:
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

          workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

          pdmScmOverlay =
            final: prev:
            lib.genAttrs [ "amaranth" "amaranth-boards" ] (
              name:
              prev.${name}.overrideAttrs (old: {
                nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { pdm-backend = [ ]; };
                env.PDM_BUILD_SCM_VERSION = old.version;
              })
            );

          editableMetadataOverlay = final: prev: {
            hardware = prev.hardware.overrideAttrs (old: {
              src = lib.fileset.toSource {
                root = old.src;
                fileset = lib.fileset.unions [
                  (old.src + "/pyproject.toml")
                  (old.src + "/README.md")
                ];
              };
            });
          };

          pythonSet =
            (pkgs.callPackage inputs.pyproject-nix.build.packages { inherit python; }).overrideScope
              (
                lib.composeManyExtensions [
                  inputs.pyproject-build-systems.overlays.wheel
                  (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
                  pdmScmOverlay
                  (workspace.mkEditablePyprojectOverlay { root = "$PRJ_ROOT"; })
                  editableMetadataOverlay
                ]
              );

          virtualenv = pythonSet.mkVirtualEnv "hardware-dev-env" workspace.deps.all;
        in
        {
          devshells.default = {
            packages = [
              virtualenv
              pkgs.ruff
              pkgs.yamlfmt
              pkgs.pyrefly
            ];

            env = [
              {
                name = "UV_PYTHON_DOWNLOADS";
                value = "never";
              }
              {
                name = "UV_PYTHON";
                value = "${virtualenv}/bin/python";
              }
              {
                name = "PYTHONPATH";
                unset = true;
              }
              {
                name = "UV_PROJECT_ENVIRONMENT";
                value = "${virtualenv}";
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
