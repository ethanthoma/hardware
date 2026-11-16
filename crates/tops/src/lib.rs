//! Board-level top modules: an RHDL design wrapped in a Fixture whose ports
//! match a board's LPF constraint names.

use rhdl::prelude::*;

pub mod hello;

/// A top that can be exported as Verilog for a board.
pub struct Top {
    pub name: &'static str,
    pub build: fn() -> Result<vlog::ModuleList, RHDLError>,
}

pub const TOPS: &[Top] = &[Top {
    name: "hello",
    build: hello::fixture,
}];
