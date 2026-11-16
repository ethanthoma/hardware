# Patched crates

Each `<crate>-*.patch` here is applied by `nix/patched-crates.nix` to the
pristine crates.io source; the devshell symlinks the result to
`patches/<crate>` (gitignored), which the root `Cargo.toml` uses via
`[patch.crates-io]`. Cargo therefore needs the devshell (or direnv) active.

To change a patch: copy the crate source somewhere writable, edit it, and
regenerate with `diff -u`, paths prefixed `a/` and `b/`.

## backtrace 0.3.77 — `backtrace-kani-unreachable.patch`

Pulled in via rhdl-vlog -> miette (fancy) -> backtrace-ext. Under Kani the
crate fails with `E0659: unreachable is ambiguous`: Kani's std shim exports
its own `unreachable!`, which clashes with the `#[macro_use] extern crate std`
glob. Fix: call `::core::unreachable!()` in `src/types.rs`.

Drop this once upstream backtrace or Kani resolves the conflict.
