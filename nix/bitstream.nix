# Reproducible bitstreams: RHDL top -> Verilog -> yosys -> nextpnr -> ecppack.
#
#   nix build .#blink     # result/top.bit (+ top.v, top.json, top.config, logs)
#   nix build .#verilog   # result/<top>.v for every top in crates/tops
{
  pkgs,
  rust,
  patchedCrates,
}:
let
  lib = pkgs.lib;
  root = ../.;

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rust;
    rustc = rust;
  };

  # Builds the exporter and runs it in the same derivation. RHDL's #[kernel]
  # macro bakes absolute source paths into the binary and re-reads those files
  # at export time, so the export has to happen while the build tree exists.
  verilog = rustPlatform.buildRustPackage {
    pname = "verilog";
    version = "0.1.0";

    # Only Rust sources go in, so Python/doc edits don't rebuild it.
    src = lib.fileset.toSource {
      inherit root;
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

    cargoBuildFlags = [
      "-p"
      "tops"
      "--bin"
      "export-verilog"
    ];
    doCheck = false;

    nativeBuildInputs = [ pkgs.iverilog ]; # rhdl validates exported Verilog with iverilog

    installPhase = ''
      runHook preInstall
      mkdir -p $out
      target/*/release/export-verilog --all $out \
        || target/release/export-verilog --all $out
      runHook postInstall
    '';
  };

  boards = {
    ecp5-5g-evn = {
      lpf = ../boards/ecp5-5g-evn.lpf;
      # LFE5UM5G-85F-8BG381
      nextpnrArgs = [
        "--um5g-85k"
        "--package"
        "CABGA381"
        "--speed"
        "8"
      ];
    };
  };

  mkBitstream =
    {
      top,
      board,
      # Placement depends on the seed; pin it so the bitstream is reproducible.
      seed ? 1,
    }:
    let
      b = boards.${board};
    in
    pkgs.runCommand "bitstream-${top}-${board}"
      {
        nativeBuildInputs = [
          pkgs.yosys
          pkgs.nextpnr
          pkgs.trellis
        ];
      }
      ''
        mkdir -p $out
        cd $out
        cp ${verilog}/${top}.v top.v
        yosys -q -l yosys.log -p "synth_ecp5 -top top -json top.json" top.v
        nextpnr-ecp5 -q -l nextpnr.log \
          ${lib.escapeShellArgs b.nextpnrArgs} \
          --seed ${toString seed} \
          --json top.json --lpf ${b.lpf} --textcfg top.config
        ecppack --compress top.config top.bit
      '';
in
{
  inherit verilog mkBitstream;

  bitstreams = {
    blink = mkBitstream {
      top = "blink";
      board = "ecp5-5g-evn";
    };
  };
}
