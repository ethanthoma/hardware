//! 8N1 UART transmitter.
//!
//! Hand it a byte with `valid` while `ready` is high; it latches the byte that
//! cycle and shifts out start bit, 8 data bits LSB-first, and a stop bit, each
//! held for `clocks_per_bit` clocks. `tx` idles high.

use rhdl::prelude::*;
use rhdl_fpga::core::{constant::Constant, dff::DFF};

#[derive(PartialEq, Debug, Digital, Default, Clone, Copy)]
pub struct TxIn {
    pub valid: bool,
    pub data: b8,
}

#[derive(PartialEq, Debug, Digital, Default, Clone, Copy)]
pub struct TxOut {
    pub tx: bool,
    pub ready: bool,
}

#[derive(Clone, Debug, Synchronous, SynchronousDQ)]
pub struct UartTx {
    clocks_per_bit: Constant<b16>,
    /// Clocks elapsed in the current bit.
    baud: DFF<b16>,
    /// {stop, data[7:0], start}; bit 0 is on the wire.
    shift: DFF<b10>,
    /// Bits left to send, 0 when idle.
    remaining: DFF<b4>,
}

impl UartTx {
    /// `clocks_per_bit` = clock frequency / baud rate, e.g. 12 MHz / 115200 ≈ 104.
    pub fn new(clocks_per_bit: u16) -> Self {
        assert!(clocks_per_bit >= 1);
        Self {
            clocks_per_bit: Constant::new(b16(clocks_per_bit as u128)),
            baud: DFF::new(b16(0)),
            shift: DFF::new(b10(0x3ff)),
            remaining: DFF::new(b4(0)),
        }
    }
}

impl SynchronousIO for UartTx {
    type I = TxIn;
    type O = TxOut;
    type Kernel = uart_tx_kernel;
}

#[kernel]
pub fn uart_tx_kernel(_cr: ClockReset, i: TxIn, q: UartTxQ) -> (TxOut, UartTxD) {
    let busy = q.remaining != b4(0);
    let mut d = UartTxD::dont_care();
    d.baud = q.baud;
    d.shift = q.shift;
    d.remaining = q.remaining;

    if !busy {
        if i.valid {
            // stop bit (1) at bit 9, data at 8..1, start bit (0) at bit 0
            d.shift = (i.data.resize::<10>() << 1) | b10(0x200);
            d.remaining = b4(10);
            d.baud = b16(0);
        }
    } else if q.baud == q.clocks_per_bit - b16(1) {
        d.baud = b16(0);
        d.shift = q.shift >> 1;
        d.remaining = q.remaining - b4(1);
    } else {
        d.baud = q.baud + b16(1);
    }

    let tx = if busy { (q.shift & b10(1)).any() } else { true };
    (TxOut { tx, ready: !busy }, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode 8N1 frames from a per-clock `tx` trace, sampling mid-bit.
    fn decode(tx: &[bool], cpb: usize) -> Vec<u8> {
        let mut out = vec![];
        let mut t = 1;
        while t < tx.len() {
            if tx[t - 1] && !tx[t] {
                let mid = |bit: usize| t + bit * cpb + cpb / 2;
                if mid(9) >= tx.len() {
                    break;
                }
                assert!(!tx[mid(0)], "start bit not held low at t={t}");
                let byte = (0..8).fold(0u8, |b, k| b | ((tx[mid(k + 1)] as u8) << k));
                assert!(tx[mid(9)], "missing stop bit for {byte:#04x}");
                out.push(byte);
                t += 10 * cpb;
            } else {
                t += 1;
            }
        }
        out
    }

    /// Drive one byte per `gap` clocks (valid for a single cycle each).
    fn stimulus(bytes: &[u8], gap: usize, tail: usize) -> Vec<TxIn> {
        let mut v = vec![];
        for &b in bytes {
            v.push(TxIn { valid: true, data: b8(b as u128) });
            v.extend(std::iter::repeat_n(TxIn::default(), gap - 1));
        }
        v.extend(std::iter::repeat_n(TxIn::default(), tail));
        v
    }

    fn run(uut: &UartTx, input: Vec<TxIn>) -> Vec<TxOut> {
        uut.run(input.with_reset(1).clock_pos_edge(100))
            .synchronous_sample()
            .map(|s| s.output)
            .collect()
    }

    #[test]
    fn sends_bytes_8n1() {
        let cpb = 4;
        let bytes = [0xa5, 0x00, 0xff, 0x55, b'H'];
        let uut = UartTx::new(cpb as u16);
        let out = run(&uut, stimulus(&bytes, 10 * cpb + 2, 4 * cpb));
        let tx: Vec<bool> = out.iter().map(|o| o.tx).collect();
        assert_eq!(decode(&tx, cpb), bytes);
    }

    #[test]
    fn ready_low_while_busy_and_idle_high() {
        let cpb = 3;
        let uut = UartTx::new(cpb as u16);
        let out = run(&uut, stimulus(&[0x00], 40, 0));
        // reset cycle, then valid accepted, then exactly 10 bit-times busy
        let busy = out.iter().filter(|o| !o.ready).count();
        assert_eq!(busy, 10 * cpb);
        assert!(out.iter().filter(|o| o.ready).all(|o| o.tx));
    }

    #[test]
    fn ignores_valid_while_busy() {
        let cpb = 2;
        let uut = UartTx::new(cpb as u16);
        // second byte arrives mid-frame and must be dropped
        let mut input = stimulus(&[0x81], 5, 0);
        input.push(TxIn { valid: true, data: b8(0x7e) });
        input.extend(std::iter::repeat_n(TxIn::default(), 40));
        let tx: Vec<bool> = run(&uut, input).iter().map(|o| o.tx).collect();
        assert_eq!(decode(&tx, cpb), [0x81]);
    }

    #[test]
    fn rtl_matches_sim() -> miette::Result<()> {
        let uut = UartTx::new(3);
        let input = stimulus(&[0xa5, 0x3c], 40, 10);
        let tb: SynchronousTestBench<_, _> = uut
            .run(input.with_reset(1).clock_pos_edge(100))
            .collect();
        tb.rtl(&uut, &Default::default())?.run_iverilog()?;
        tb.ntl(&uut, &Default::default())?.run_iverilog()?;
        Ok(())
    }
}
