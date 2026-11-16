# Shared Cargo build setup for everything nix builds from the workspace.
{
  pkgs,
  rust,
  patchedCrates,
}:
let
  lib = pkgs.lib;
in
rec {
  rustPlatform = pkgs.makeRustPlatform {
    cargo = rust;
    rustc = rust;
  };

  # Only Rust sources go in, so other edits don't trigger rebuilds.
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../crates
    ];
  };

  cargoLock = {
    lockFile = ../Cargo.lock;
    outputHashes."rhdl-0.1.0" = "sha256-BV64X4lV93KIvt0TWxJr8D3PfJZCc/B9ukBmtim/bI8=";
  };

  # Same patched crates the devshell links in; see patches/README.md.
  postPatch = lib.concatStrings (
    lib.mapAttrsToList (name: src: ''
      mkdir -p patches
      ln -s ${src} patches/${name}
    '') patchedCrates
  );

  # buildRustPackage args shared by every workspace derivation.
  common = {
    version = "0.1.0";
    inherit src cargoLock postPatch;
    # rhdl shells out to iverilog to validate exported Verilog.
    nativeBuildInputs = [ pkgs.iverilog ];
  };
}
