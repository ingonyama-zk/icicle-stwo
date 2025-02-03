use std::cmp::Reverse;
use std::fmt::Debug;
use std::mem::transmute;
use std::ops::Deref;

use icicle_core::tree::{merkle_tree_digests_len, TreeBuilderConfig};
use icicle_core::vec_ops::{are_bytes_equal, VecOpsConfig};
use icicle_core::Matrix;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_hash::blake2s::blake2s_commit_layer;
use itertools::Itertools;

use super::IcicleBackend;
use crate::core::backend::{BackendForChannel, Col, Column, ColumnOps, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};

impl BackendForChannel<Blake2sMerkleChannel> for IcicleBackend {}

impl ColumnOps<Blake2sHash> for IcicleBackend {
    type Column = DeviceColumnBlake;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

pub struct DeviceColumnBlake {
    pub data: DeviceVec<Blake2sHash>,
    pub length: usize,
}

impl PartialEq for DeviceColumnBlake {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        let cfg = VecOpsConfig::default();
        are_bytes_equal::<Blake2sHash>(self.data.deref(), other.data.deref(), &cfg)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Clone for DeviceColumnBlake {
    fn clone(&self) -> Self {
        let mut data = DeviceVec::cuda_malloc(self.length).unwrap();
        data.copy_from_device(&self.data);
        Self {
            data,
            length: self.length,
        }
    }
}

impl Debug for DeviceColumnBlake {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_cpu();
        f.debug_struct("DeviceColumnBlake")
            .field("data", &data.as_slice())
            .field("length", &self.length)
            .finish()
    }
}

impl DeviceColumnBlake {
    pub fn from_cpu(values: &[Blake2sHash]) -> Self {
        let length = values.len();
        let mut data: DeviceVec<Blake2sHash> = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&values));
        Self { data, length }
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

impl Column<Blake2sHash> for DeviceColumnBlake {
    fn zeros(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        let host_data = vec![Blake2sHash::default(); length];
        data.copy_from_host(HostSlice::from_slice(&host_data));

        Self { data, length }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<Blake2sHash> {
        let mut host_data = Vec::<Blake2sHash>::with_capacity(self.length);
        self.data
            .copy_to_host(HostSlice::from_mut_slice(&mut host_data));
        host_data
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> Blake2sHash {
        let mut host_vec = vec![Blake2sHash::default(); 1];
        unsafe {
            DeviceSlice::from_slice(std::slice::from_raw_parts(self.data.as_ptr().add(index), 1))
                .copy_to_host(HostSlice::from_mut_slice(&mut host_vec))
                .unwrap();
        }
        host_vec[0]
    }

    fn set(&mut self, index: usize, value: Blake2sHash) {
        let host_vec = vec![value; 1];
        unsafe {
            DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(index),
                1,
            ))
            .copy_from_host(HostSlice::from_slice(&host_vec))
            .unwrap();
        }
    }
}

impl FromIterator<Blake2sHash> for DeviceColumnBlake {
    fn from_iter<I: IntoIterator<Item = Blake2sHash>>(iter: I) -> Self {
        let host_data = iter.into_iter().collect_vec();
        let length = host_data.len();
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&host_data))
            .unwrap();

        Self { data, length }
    }
}

impl MerkleOps<Blake2sMerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash> {
        nvtx::range_push!("[ICICLE] Extract prev_layer");
        let prev_layer = match prev_layer {
            Some(layer) => layer,
            // Hacky, since creating a DeviceVec of size 0 seems to not work
            // NOTE: blake2s_commit_layer uses a length of 1 as an indicator that
            // the prev_layer does not exist
            None => unsafe {
                &<Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash> as Column<Blake2sHash>>::uninitialized(1)
            },
        };
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] Create matrices");
        let mut columns_as_matrices = vec![];
        for &col in columns {
            let col_as_slice = col.data[..].as_slice();
            columns_as_matrices.push(Matrix::from_slice(&col_as_slice, 4, col.len()));
        }
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] Cuda malloc digests");
        let digests_bytes = (1 << log_size) * 32;
        let mut d_digests_slice = DeviceVec::cuda_malloc(digests_bytes).unwrap();
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] cuda commit layer");
        blake2s_commit_layer(
            &(unsafe { transmute::<&DeviceVec<Blake2sHash>, &DeviceVec<u8>>(&prev_layer.data) })[..],
            true,
            &columns_as_matrices,
            true,
            columns.len() as u32,
            1 << log_size,
            &mut d_digests_slice[..],
        ).unwrap();
        nvtx::range_pop!();

        DeviceColumnBlake {
            data: unsafe { transmute(d_digests_slice) },
            length: 1 << log_size,
        }
    }
}
