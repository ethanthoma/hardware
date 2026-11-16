//! ECP5-5G EVN blinker: the 8 LEDs show the top byte of a free-running counter.
//!
//! Ports: `clk12` (12 MHz), `led[7:0]` (active-low on this board).

use rhdl::prelude::*;
use rhdl_fpga::core::counter::Counter;

#[derive(Clone, Synchronous, SynchronousDQ, Default)]
#[rhdl(dq_no_prefix)]
pub struct U {
    counter: Counter<32>,
}

impl SynchronousIO for U {
    type I = ();
    type O = b8;
    type Kernel = blink;
}

#[kernel]
pub fn blink(_cr: ClockReset, _i: (), q: Q) -> (b8, D) {
    let mut d = D::dont_care();
    d.counter = true;
    // bits 31..24: the MSB toggles every ~179 s at 12 MHz, the LSB every ~0.7 s.
    let byte: b8 = (q.counter >> 24).resize::<8>();
    (!byte, d)
}

pub fn fixture() -> Result<vlog::ModuleList, RHDLError> {
    type T = Adapter<U, Red>;
    let mut fixture = Fixture::new("top", T::new(U::default()));
    let (i, o) = fixture.io_dont_care();
    bind!(fixture, clk12 -> i.clock_reset.val().clock);
    fixture.constant_input(reset(false), &path!(i.clock_reset.val().reset))?;
    bind!(fixture, led <- o.val());
    fixture.module()
}
