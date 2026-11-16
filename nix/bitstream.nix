# Reproducible bitstreams: RHDL top -> Verilog -> yosys -> nextpnr -> ecppack.
#
#   nix build .#hello     # result/top.bit (+ top.v, top.json, top.config, logs)
#   nix build .#verilog   # result/<top>.v for every top in crates/tops
{ pkgs, rustBuild }:
let
  lib = pkgs.lib;

  # Builds the exporter and runs it in the same derivation. RHDL's #[kernel]
  # macro bakes absolute source paths into the binary and re-reads those files
  # at export time, so the export has to happen while the build tree exists.
  verilog = rustBuild.rustPlatform.buildRustPackage (
    rustBuild.common
    // {
      pname = "verilog";
      cargoBuildFlags = [
        "-p"
        "tops"
        "--bin"
        "export-verilog"
      ];
      doCheck = false;
      installPhase = ''
        runHook preInstall
        mkdir -p $out
        $(find target -path '*/release/export-verilog' -type f | head -1) --all $out
        runHook postInstall
      '';
    }
  );

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
        # nextpnr exits non-zero if any clock misses its constraint.
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
    hello = mkBitstream {
      top = "hello";
      board = "ecp5-5g-evn";
    };
  };
}
