{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devshell.url = "github:numtide/devshell";

    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:adisbladis/uv2nix";
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

      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "i686-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem =
        { system, ... }:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfree = true;
            config.cudaSupport = true;
            config.cudaVersion = "12";
            overlays = [ inputs.devshell.overlays.default ];
          };

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
          _module.args.pkgs = pkgs;

          devshells.default = {
            packages = [
              pkgs.ruff
              pkgs.ty
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
              {
                name = "claude";
                package = pkgs.claude-code;
              }
              { package = pkgs.tokei; }
              { package = pkgs.yosys; }
            ];
          };
        };
    };

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };
}
