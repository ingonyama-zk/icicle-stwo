use std::cmp::Reverse;

use icicle_core::tree::{merkle_tree_digests_len, TreeBuilderConfig};
use icicle_core::Matrix;
use icicle_cuda_runtime::memory::HostSlice;
use icicle_hash::blake2s::build_blake2s_mmcs;
use itertools::Itertools;

use super::IcicleBackend;
use crate::core::backend::{BackendForChannel, Col, Column, ColumnOps, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};

impl ColumnOps<Blake2sHash> for IcicleBackend {
    type Column = Vec<Blake2sHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for IcicleBackend {
    const COMMIT_IMPLEMENTED: bool = true;

    fn commit_columns(
        columns: Vec<&Col<Self, BaseField>>,
    ) -> Vec<Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash>> {
        let mut config = TreeBuilderConfig::default();
        config.arity = 2;
        config.digest_elements = 32;
        config.sort_inputs = false;

        nvtx::range_push!("[ICICLE] log_max");
        let log_max = columns
            .iter()
            .sorted_by_key(|c| Reverse(c.len()))
            .next()
            .unwrap()
            .len()
            .ilog2();
        nvtx::range_pop!();
        let mut matrices = vec![];
        nvtx::range_push!("[ICICLE] create matrix");
        for col in columns.into_iter().sorted_by_key(|c| Reverse(c.len())) {
            matrices.push(Matrix::from_slice(col, 4, col.len()));
        }
        nvtx::range_pop!();
        nvtx::range_push!("[ICICLE] merkle_tree_digests_len");
        let digests_len = merkle_tree_digests_len(log_max as u32, 2, 32);
        nvtx::range_pop!();
        let mut digests = vec![0u8; digests_len];
        let digests_slice = HostSlice::from_mut_slice(&mut digests);

        nvtx::range_push!("[ICICLE] build_blake2s_mmcs");
        build_blake2s_mmcs(&matrices, digests_slice, &config).unwrap();
        nvtx::range_pop!();

        let mut digests: &[<Blake2sMerkleHasher as MerkleHasher>::Hash] =
            unsafe { std::mem::transmute(digests.as_mut_slice()) };
        // Transmute digests into stwo format
        let mut layers = vec![];
        let mut offset = 0usize;
        nvtx::range_push!("[ICICLE] convert to CPU layer");
        for log in 0..=log_max {
            let inv_log = log_max - log;
            let number_of_rows = 1 << inv_log;

            let mut layer = vec![];
            layer.extend_from_slice(&digests[offset..offset + number_of_rows]);
            layers.push(layer);

            if log != log_max {
                offset += number_of_rows;
            }
        }

        layers.reverse();
        nvtx::range_pop!();
        layers
    }

    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash> {
        // todo!()
        <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size, prev_layer, columns,
        )
    }
}

impl BackendForChannel<Blake2sMerkleChannel> for IcicleBackend {}
