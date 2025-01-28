use std::fmt::Debug;
use std::iter::zip;
use std::mem::transmute;
use std::ops::Deref;
use std::{array, mem};

use bytemuck::allocation::cast_vec;
use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use icicle_core::vec_ops::{are_bytes_equal, stwo_convert, transpose_matrix, VecOpsConfig};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_m31::field::{QuarticExtensionField, ScalarField};
use itertools::{izip, Itertools};
use num_traits::Zero;

use super::IcicleBackend;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::{Column, ColumnOps, CpuBackend};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::secure_column::{SecureColumnByCoords, SECURE_EXTENSION_DEGREE};
use crate::core::fields::{FieldExpOps, FieldOps};

impl FieldOps<BaseField> for IcicleBackend {
    fn batch_inverse(column: &DeviceColumn, dst: &mut DeviceColumn) {
        todo!()
    }
}

impl FieldOps<SecureField> for IcicleBackend {
    fn batch_inverse(column: &DeviceSecureColumn, dst: &mut DeviceSecureColumn) {
        todo!()
    }
}

// A column that is stored on device
pub struct DeviceColumn {
    pub data: DeviceVec<BaseField>,
    /// The number of [`BaseField`]s in the vector.
    pub length: usize,
}

impl PartialEq for DeviceColumn {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        let cfg = VecOpsConfig::default();
        are_bytes_equal::<BaseField>(self.data.deref(), other.data.deref(), &cfg)
    }
    
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Clone for DeviceColumn {
    fn clone(&self) -> Self {
        let mut data = DeviceVec::cuda_malloc(self.length).unwrap();
        data.copy_from_device(&self.data);
        Self{data, length: self.length}
    }
}

impl Debug for DeviceColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_cpu();
        f.debug_struct("DeviceColumn").field("data", &data.as_slice()).field("length", &self.length).finish()
    }
}

impl DeviceColumn {
    pub fn from_cpu(values: &[BaseField]) -> Self {
        let length = values.len();
        let mut data: DeviceVec<BaseField> = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&values));
        Self{data, length}
    }
    
    pub fn len(&self) -> usize {
        self.length
    }
}

impl ColumnOps<BaseField> for IcicleBackend {
    type Column = DeviceColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        todo!()
        // CpuBackend::bit_reverse_column(column)
    }
}

impl Column<BaseField> for DeviceColumn {
    fn zeros(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        let host_data = vec![BaseField::zero(); length];
        data.copy_from_host(HostSlice::from_slice(&host_data));

        Self { data, length }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        let mut host_data = Vec::<BaseField>::with_capacity(self.length);
        self.data
            .copy_to_host(HostSlice::from_mut_slice(&mut host_data));
        host_data
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> BaseField {
        let mut host_vec = vec![BaseField::zero(); 1];
        unsafe {
            DeviceSlice::from_slice(std::slice::from_raw_parts(self.data.as_ptr().add(index), 1))
                .copy_to_host(HostSlice::from_mut_slice(&mut host_vec))
                .unwrap();
        }
        host_vec[0]
    }

    fn set(&mut self, index: usize, value: BaseField) {
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

impl FromIterator<BaseField> for DeviceColumn {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let host_data = iter.into_iter().collect_vec();
        let length = host_data.len();
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&host_data))
            .unwrap();

        Self { data, length }
    }
}

// A efficient structure for storing and operating on a arbitrary number of [`SecureField`] values.
pub struct DeviceCM31Column {
    pub data: DeviceVec<CM31>,
    pub length: usize,
}

impl PartialEq for DeviceCM31Column {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        let cfg = VecOpsConfig::default();
        are_bytes_equal::<CM31>(self.data.deref(), other.data.deref(), &cfg)
    }
    
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Clone for DeviceCM31Column {
    fn clone(&self) -> Self {
        let mut data = DeviceVec::cuda_malloc(self.length).unwrap();
        data.copy_from_device(&self.data);
        Self{data, length: self.length}
    }
}

impl Debug for DeviceCM31Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_cpu();
        f.debug_struct("DeviceColumn").field("data", &data.as_slice()).field("length", &self.length).finish()
    }
}

impl Column<CM31> for DeviceCM31Column {
    fn zeros(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        let host_data = vec![CM31::zero(); length];
        data.copy_from_host(HostSlice::from_slice(&host_data));

        Self { data, length }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<CM31> {
        let mut result = Vec::<CM31>::with_capacity(self.length);
        self.data
            .copy_to_host(HostSlice::from_mut_slice(result.as_mut_slice()));
        result
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> CM31 {
        let mut host_vec = vec![CM31::zero(); 1];
        unsafe {
            DeviceSlice::from_slice(std::slice::from_raw_parts(self.data.as_ptr().add(index), 1))
                .copy_to_host(HostSlice::from_mut_slice(&mut host_vec))
                .unwrap();
        }
        host_vec[0]
    }

    fn set(&mut self, index: usize, value: CM31) {
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

impl FromIterator<CM31> for DeviceCM31Column {
    fn from_iter<I: IntoIterator<Item = CM31>>(iter: I) -> Self {
        let host_data = iter.into_iter().collect_vec();
        let length = host_data.len();
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&host_data))
            .unwrap();

        Self { data, length }
    }
}

/// An efficient structure for storing and operating on a arbitrary number of [`SecureField`]
/// values.
pub struct DeviceSecureColumn {
    pub data: DeviceVec<SecureField>,
    /// The number of [`SecureField`]s in the vector.
    pub length: usize,
}

impl PartialEq for DeviceSecureColumn {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        let cfg = VecOpsConfig::default();
        are_bytes_equal::<QM31>(self.data.deref(), other.data.deref(), &cfg)
    }
    
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Clone for DeviceSecureColumn {
    fn clone(&self) -> Self {
        let mut data = DeviceVec::cuda_malloc(self.length).unwrap();
        data.copy_from_device(&self.data);
        Self{data, length: self.length}
    }
}

impl Debug for DeviceSecureColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_cpu();
        f.debug_struct("DeviceColumn").field("data", &data.as_slice()).field("length", &self.length).finish()
    }
}

impl ColumnOps<SecureField> for IcicleBackend {
    type Column = DeviceSecureColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        todo!()
        // CpuBackend::bit_reverse_column(column)
    }
}

impl Column<SecureField> for DeviceSecureColumn {
    fn zeros(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        let host_data = vec![SecureField::zero(); length];
        data.copy_from_host(HostSlice::from_slice(&host_data));

        Self { data, length }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        let mut result = Vec::<SecureField>::with_capacity(self.length);
        self.data
            .copy_to_host(HostSlice::from_mut_slice(result.as_mut_slice()));
        result
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> SecureField {
        let mut host_vec = vec![SecureField::zero(); 1];
        unsafe {
            DeviceSlice::from_slice(std::slice::from_raw_parts(self.data.as_ptr().add(index), 1))
                .copy_to_host(HostSlice::from_mut_slice(&mut host_vec))
                .unwrap();
        }
        host_vec[0]
    }

    fn set(&mut self, index: usize, value: SecureField) {
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
    
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FromIterator<SecureField> for DeviceSecureColumn {
    fn from_iter<I: IntoIterator<Item = QM31>>(iter: I) -> Self {
        let host_data = iter.into_iter().collect_vec();
        let length = host_data.len();
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&host_data))
            .unwrap();

        Self { data, length }
    }
}
