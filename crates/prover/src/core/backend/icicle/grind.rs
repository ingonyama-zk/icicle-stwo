use crate::core::{backend::CpuBackend, channel::Channel, proof_of_work::GrindOps};

use super::IcicleBackend;


impl<C: Channel> GrindOps<C> for IcicleBackend {
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        // todo!()
        CpuBackend::grind(channel, pow_bits)
    }
}