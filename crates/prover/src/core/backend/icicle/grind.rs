use super::IcicleBackend;
use crate::core::backend::CpuBackend;
use crate::core::channel::Channel;
use crate::core::proof_of_work::GrindOps;

impl<C: Channel> GrindOps<C> for IcicleBackend {
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        // todo!()
        CpuBackend::grind(channel, pow_bits)
    }
}
