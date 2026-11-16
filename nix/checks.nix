# `nix flake check` targets.
#
#   nix build .#checks.x86_64-linux.<name>   # run one
{
  pkgs,
  rustBuild,
  kani,
  bitstreams,
}:
let
  lib = pkgs.lib;
  inherit (rustBuild) rustPlatform common;
in
{
  # Unit + exhaustive tests, and rhdl's "kernel compiles to HDL" tests.
  cargo-test = rustPlatform.buildRustPackage (
    common
    // {
      pname = "cargo-test";
      buildPhase = "true";
      checkPhase = ''
        runHook preCheck
        cargo test --workspace --offline --frozen
        runHook postCheck
      '';
      doCheck = true;
      installPhase = "touch $out";
    }
  );

  # Kani proofs. Stubbing stands in for RHDL's trace hooks (see kani_support.rs).
  kani = rustPlatform.buildRustPackage (
    common
    // {
      pname = "kani-proofs";
      nativeBuildInputs = common.nativeBuildInputs ++ [ kani ];
      buildPhase = ''
        runHook preBuild
        export HOME=$TMPDIR
        cargo kani --workspace -Z stubbing
        runHook postBuild
      '';
      doCheck = false;
      installPhase = "touch $out";
    }
  );
}
# Bitstreams build only if every clock meets timing (nextpnr fails otherwise).
// lib.mapAttrs' (name: drv: lib.nameValuePair "bitstream-${name}" drv) bitstreams
