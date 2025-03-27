use super::CpuBackend;
use crate::core::channel::Channel;
use crate::core::proof_of_work::GrindOps;
use crate::{nvtx_timed, nvtx_timed_pop};


impl<C: Channel> GrindOps<C> for CpuBackend {
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        #[cfg(feature = "icicle")]
        nvtx_timed!("fn grind(");

        let mut nonce = 0;
        loop {
            let mut channel = channel.clone();
            channel.mix_u64(nonce);
            if channel.trailing_zeros() >= pow_bits {
                #[cfg(feature = "icicle")]
                nvtx_timed_pop!();
                return nonce;
            }
            nonce += 1;
        }
    }
}
