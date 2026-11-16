//! ECP5-5G EVN: send "Hello from RHDL\n" over the UART forever at 115200 baud.
//!
//! Ports: `clk12` (12 MHz), `uart_tx`, `led[7:0]` (active-low).
//! LED7 blinks (~0.7 s) so the design is visibly alive even without a UART
//! listener; LED0 lights while a byte is on the wire.
//!
//! The UART goes to a Raspberry Pi on the 40-pin header (see the board LPF);
//! read it there with `picocom -b 115200 /dev/serial0`.

use periph::uart_tx::{TxIn, UartTx};
use rhdl::prelude::*;
use rhdl_fpga::core::dff::DFF;

/// 12 MHz / 115200 baud = 104.17 -> 104 (0.16% fast; UART tolerates ~2%).
pub const CLOCKS_PER_BIT: u16 = 104;

pub const MSG: [b8; 16] = [
    b8(b'H' as u128),
    b8(b'e' as u128),
    b8(b'l' as u128),
    b8(b'l' as u128),
    b8(b'o' as u128),
    b8(b' ' as u128),
    b8(b'f' as u128),
    b8(b'r' as u128),
    b8(b'o' as u128),
    b8(b'm' as u128),
    b8(b' ' as u128),
    b8(b'R' as u128),
    b8(b'H' as u128),
    b8(b'D' as u128),
    b8(b'L' as u128),
    b8(b'\n' as u128),
];

#[derive(PartialEq, Debug, Digital, Default, Clone, Copy)]
pub struct HelloOut {
    pub uart_tx: bool,
    /// Active-low, as wired on the EVN.
    pub led: b8,
}

#[derive(Clone, Synchronous, SynchronousDQ)]
pub struct Hello {
    tx: UartTx,
    idx: DFF<b4>,
    heartbeat: DFF<b24>,
}

impl Default for Hello {
    fn default() -> Self {
        Self::new(CLOCKS_PER_BIT)
    }
}

impl Hello {
    pub fn new(clocks_per_bit: u16) -> Self {
        Self {
            tx: UartTx::new(clocks_per_bit),
            idx: DFF::new(b4(0)),
            heartbeat: DFF::new(b24(0)),
        }
    }
}

impl SynchronousIO for Hello {
    type I = ();
    type O = HelloOut;
    type Kernel = hello;
}

#[kernel]
pub fn hello(_cr: ClockReset, _i: (), q: HelloQ) -> (HelloOut, HelloD) {
    let mut d = HelloD::dont_care();
    // UartTx latches whatever is offered on a cycle it reports ready.
    d.tx = TxIn {
        valid: true,
        data: MSG[q.idx],
    };
    d.idx = if q.tx.ready { q.idx + b4(1) } else { q.idx };
    d.heartbeat = q.heartbeat + b24(1);

    // bit 7: heartbeat (top counter bit, ~1.4 s period at 12 MHz); bit 0: busy
    let alive: b8 = if (q.heartbeat >> 23).any() { b8(0x80) } else { b8(0) };
    let busy: b8 = if q.tx.ready { b8(0) } else { b8(0x01) };
    let led = !(alive | busy);
    (HelloOut { uart_tx: q.tx.tx, led }, d)
}

pub fn fixture() -> Result<vlog::ModuleList, RHDLError> {
    type T = Adapter<Hello, Red>;
    let mut fixture = Fixture::new("top", T::new(Hello::default()));
    let (i, o) = fixture.io_dont_care();
    bind!(fixture, clk12 -> i.clock_reset.val().clock);
    fixture.constant_input(reset(false), &path!(i.clock_reset.val().reset))?;
    bind!(fixture, uart_tx <- o.val().uart_tx);
    bind!(fixture, led <- o.val().led);
    fixture.module()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sends_the_message_in_a_loop() {
        let cpb = 3;
        let uut = Hello::new(cpb as u16);
        let cycles = 2 * 16 * 10 * cpb + 50;
        let tx: Vec<bool> = uut
            .run(std::iter::repeat_n((), cycles).with_reset(1).clock_pos_edge(100))
            .synchronous_sample()
            .map(|s| s.output.uart_tx)
            .collect();

        // decode 8N1, mid-bit sampling
        let mut bytes = vec![];
        let mut t = 1;
        while t + 10 * cpb < tx.len() {
            if tx[t - 1] && !tx[t] {
                let mid = |bit: usize| t + bit * cpb + cpb / 2;
                bytes.push((0..8).fold(0u8, |b, k| b | ((tx[mid(k + 1)] as u8) << k)));
                t += 10 * cpb;
            } else {
                t += 1;
            }
        }
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("Hello from RHDL\nHello from RHDL\n"), "{text:?}");
    }
}
