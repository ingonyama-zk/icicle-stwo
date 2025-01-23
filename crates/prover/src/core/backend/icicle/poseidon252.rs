use crate::core::{backend::{BackendForChannel, Col, CpuBackend}, fields::m31::BaseField, vcs::{ops::{MerkleHasher, MerkleOps}, poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher}}};

use super::IcicleBackend;

impl MerkleOps<Poseidon252MerkleHasher> for IcicleBackend {
    const COMMIT_IMPLEMENTED: bool = false;

    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Poseidon252MerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Poseidon252MerkleHasher as MerkleHasher>::Hash> {
        // todo!()

        <CpuBackend as MerkleOps<Poseidon252MerkleHasher>>::commit_on_layer(
            log_size, prev_layer, columns,
        )
    }
}

impl BackendForChannel<Poseidon252MerkleChannel> for IcicleBackend {}