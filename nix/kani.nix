# Kani's prebuilt release bundle, patched to run on NixOS.
#
# Kani normally installs via `cargo kani setup`, which downloads this same
# bundle into ~/.kani plus a rustup nightly. Neither works on NixOS (no
# rustup, no FHS loader), so we patchelf the bundle and point its
# `toolchain/` at the matching nightly from rust-overlay.
{
  pkgs,
  rust-bin,
}:
let
  version = "0.68.0";

  # Must match `rust-toolchain-version` inside the bundle.
  toolchain = rust-bin.nightly."2026-08-21".minimal.override {
    extensions = [
      "rustc-dev"
      "rust-src"
      "llvm-tools"
    ];
  };

  bundle = pkgs.stdenv.mkDerivation {
    pname = "kani-bundle";
    inherit version;

    src = pkgs.fetchurl {
      url = "https://github.com/model-checking/kani/releases/download/kani-${version}/kani-${version}-x86_64-unknown-linux-gnu.tar.gz";
      hash = "sha256-MuK0hNc+3gu/ZKLPCHnEJZQiSX2K7E7mfUSN6a54Q9M=";
    };

    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [
      pkgs.stdenv.cc.cc.lib
      toolchain
    ];

    # kani-compiler finds librustc_driver via $ORIGIN/../toolchain/lib.
    runtimeDependencies = [ toolchain ];
    autoPatchelfIgnoreMissingDeps = [ "librustc_driver-*.so" ];

    dontStrip = true;

    installPhase = ''
      runHook preInstall
      mkdir -p $out
      cp -r . $out/
      ln -s ${toolchain} $out/toolchain
      runHook postInstall
    '';
  };
in
pkgs.symlinkJoin {
  name = "kani-${version}";
  paths = [ ];
  nativeBuildInputs = [ pkgs.makeWrapper ];
  postBuild = ''
    mkdir -p $out/bin
    makeWrapper ${bundle}/bin/kani-driver $out/bin/kani \
      --argv0 kani \
      --prefix PATH : ${bundle}/bin:${toolchain}/bin
    makeWrapper ${bundle}/bin/kani-driver $out/bin/cargo-kani \
      --argv0 cargo-kani \
      --prefix PATH : ${bundle}/bin:${toolchain}/bin
  '';
  passthru = { inherit bundle toolchain; };
}
