//! Kani helpers for proving RHDL kernels.

/// No-op replacements for RHDL's trace-path hooks.
///
/// `#[kernel]` wraps every kernel body in `trace_push_path`/`trace_pop_path`.
/// They're simulation-only bookkeeping (no effect on results) and their
/// thread-local internals crash kani-compiler, so proofs stub them out.
pub fn no_trace_push(_: &'static str) {}
pub fn no_trace_pop() {}

/// Declare a Kani proof harness that may call RHDL kernels.
///
/// ```ignore
/// kernel_proof! {
///     fn my_kernel_is_correct() { ... }
/// }
/// ```
///
/// Expands to `#[kani::proof]` plus the trace-hook stubs. Needs stubbing
/// enabled: run via `cargo verify` (alias for `cargo kani -Z stubbing`).
#[macro_export]
macro_rules! kernel_proof {
    ($(#[$attr:meta])* fn $name:ident() $body:block) => {
        $(#[$attr])*
        #[kani::proof]
        #[kani::stub(rhdl::core::trace_push_path, $crate::kani_support::no_trace_push)]
        #[kani::stub(rhdl::core::trace_pop_path, $crate::kani_support::no_trace_pop)]
        fn $name() $body
    };
}
