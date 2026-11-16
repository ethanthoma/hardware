//! usage: export-verilog <top> <out.v>
//!        export-verilog --all <dir>     (writes <dir>/<top>.v for every top)
use miette::{IntoDiagnostic, bail};

fn export(top: &tops::Top, out: &str) -> miette::Result<()> {
    let module = (top.build)()?;
    std::fs::write(out, module.to_string()).into_diagnostic()
}

fn main() -> miette::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_, flag, dir] if flag == "--all" => {
            for top in tops::TOPS {
                export(top, &format!("{dir}/{}.v", top.name))?;
            }
            Ok(())
        }
        [_, name, out] => {
            let Some(top) = tops::TOPS.iter().find(|t| t.name == name) else {
                let names: Vec<_> = tops::TOPS.iter().map(|t| t.name).collect();
                bail!("unknown top {name:?}; available: {names:?}");
            };
            export(top, out)
        }
        _ => Err(miette::miette!("usage: export-verilog <top> <out.v> | --all <dir>")),
    }
}
