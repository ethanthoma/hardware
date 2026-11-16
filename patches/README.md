# Vendored patches

## backtrace 0.3.77

Pulled in via rhdl-vlog -> miette (fancy) -> backtrace-ext. Under Kani, the
crate fails with `E0659: unreachable is ambiguous` (Kani's std shim exports
its own `unreachable!`, which clashes with the `#[macro_use] extern crate std`
glob). One-line fix in `src/types.rs`: call `::core::unreachable!()`.

Drop this once upstream backtrace or Kani resolves the conflict.
