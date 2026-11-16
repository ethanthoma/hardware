# Crates that need local fixes, rebuilt from crates.io + a .patch file.
#
# Each entry is symlinked into patches/<name> on devshell entry, where the
# root Cargo.toml's [patch.crates-io] picks it up. See patches/README.md.
{ pkgs }:
{
  backtrace = pkgs.applyPatches {
    name = "backtrace-0.3.77-patched";
    src = pkgs.fetchCrate {
      pname = "backtrace";
      version = "0.3.77";
      hash = "sha256-nGQNyU4wpijUkVk/B4hCLeJa3U9StrtsWRaVdtlWC5Y=";
    };
    patches = [ ../patches/backtrace-kani-unreachable.patch ];
  };
}
