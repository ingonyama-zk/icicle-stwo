use std::fmt::Debug;
use std::iter::zip;
use std::mem::transmute;
use std::ops::{Deref, DerefMut};
use std::{array, mem};

use bytemuck::allocation::cast_vec;
use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use icicle_core::vec_ops::{
    are_bytes_equal, bit_reverse_inplace, inv_scalars, stwo_convert, transpose_matrix,
    BitReverseConfig, VecOpsConfig,
};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_m31::field::{ComplexExtensionField, QuarticExtensionField, ScalarField};
use itertools::{izip, Itertools};
use num_traits::Zero;

use super::IcicleBackend;
use crate::core::backend::simd::column::SecureColumn;
use crate::core::backend::{Column, ColumnOps, CpuBackend};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::secure_column::{SecureColumnByCoords, SECURE_EXTENSION_DEGREE};
use crate::core::fields::{ExtensionOf, FieldExpOps, FieldOps};

impl FieldOps<BaseField> for IcicleBackend {
    fn batch_inverse(column: &DeviceColumn, dst: &mut DeviceColumn) {
        let cfg = VecOpsConfig::default();
        let column_transmuted: &DeviceSlice<BaseField> = column.data.deref();
        let dst_transmuted: &mut DeviceSlice<BaseField> = dst.data.deref_mut();
        inv_scalars::<ScalarField>(
            unsafe {
                transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(column_transmuted)
            },
            unsafe {
                transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                    dst_transmuted,
                )
            },
            &cfg,
        )
        .unwrap();
    }
}

impl FieldOps<SecureField> for IcicleBackend {
    fn batch_inverse(column: &DeviceSecureColumn, dst: &mut DeviceSecureColumn) {
        let cfg = VecOpsConfig::default();
        let column_transmuted: &DeviceSlice<SecureField> = column.data.deref();
        let dst_transmuted: &mut DeviceSlice<SecureField> = dst.data.deref_mut();
        inv_scalars::<QuarticExtensionField>(
            unsafe {
                transmute::<&DeviceSlice<SecureField>, &DeviceSlice<QuarticExtensionField>>(
                    column_transmuted,
                )
            },
            unsafe {
                transmute::<&mut DeviceSlice<SecureField>, &mut DeviceSlice<QuarticExtensionField>>(
                    dst_transmuted,
                )
            },
            &cfg,
        )
        .unwrap();
    }
}

// A column that is stored on device
pub struct DeviceColumn {
    pub data: DeviceVec<BaseField>,
}

impl PartialEq for DeviceColumn {
    fn eq(&self, other: &Self) -> bool {
        let cfg = VecOpsConfig::default();
        are_bytes_equal::<BaseField>(self.data.deref(), other.data.deref(), &cfg)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Clone for DeviceColumn {
    fn clone(&self) -> Self {
        let mut data = DeviceVec::cuda_malloc(self.data.len()).unwrap();
        data.copy_from_device(&self.data);
        Self { data }
    }
}

impl Debug for DeviceColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_cpu();
        f.debug_struct("DeviceColumn")
            .field("data", &data.as_slice())
            .field("length", &self.data.len())
            .finish()
    }
}

impl DeviceColumn {
    pub fn from_cpu(values: &[BaseField]) -> Self {
        let length = values.len();
        let mut data: DeviceVec<BaseField> = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&values));
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl ColumnOps<BaseField> for IcicleBackend {
    type Column = DeviceColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        let column_transmuted: &mut DeviceSlice<BaseField> = column.data.deref_mut();
        let cfg = BitReverseConfig::default();
        bit_reverse_inplace(
            unsafe {
                transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                    column_transmuted,
                )
            },
            &cfg,
        );
    }
}

impl Column<BaseField> for DeviceColumn {
    fn zeros(length: usize) -> Self {
        let data = DeviceVec::cuda_malloc_zeros(length).unwrap();

        Self { data }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        Self { data }
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        let mut host_data = vec![BaseField::zero(); self.data.len()];
        self.data
            .copy_to_host(HostSlice::from_mut_slice(&mut host_data));
        host_data
    }

    fn len(&self) -> usize {
        self.data.len()
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

        Self { data }
    }
}

impl IntoIterator for DeviceColumn {
    type Item = BaseField;

    type IntoIter = std::vec::IntoIter<BaseField>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_cpu().into_iter()
    }
}

// A efficient structure for storing and operating on a arbitrary number of [`SecureField`] values.
pub struct DeviceCM31Column {
    pub data: DeviceVec<CM31>,
}

impl PartialEq for DeviceCM31Column {
    fn eq(&self, other: &Self) -> bool {
        let cfg = VecOpsConfig::default();
        are_bytes_equal::<CM31>(self.data.deref(), other.data.deref(), &cfg)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Clone for DeviceCM31Column {
    fn clone(&self) -> Self {
        let mut data = DeviceVec::cuda_malloc(self.data.len()).unwrap();
        data.copy_from_device(&self.data);
        Self { data }
    }
}

impl Debug for DeviceCM31Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_cpu();
        f.debug_struct("DeviceColumn")
            .field("data", &data.as_slice())
            .field("length", &self.data.len())
            .finish()
    }
}

impl ColumnOps<CM31> for IcicleBackend {
    type Column = DeviceCM31Column;

    fn bit_reverse_column(column: &mut Self::Column) {
        let column_transmuted: &mut DeviceSlice<CM31> = column.data.deref_mut();
        let cfg = BitReverseConfig::default();
        bit_reverse_inplace(
            unsafe {
                transmute::<&mut DeviceSlice<CM31>, &mut DeviceSlice<ComplexExtensionField>>(
                    column_transmuted,
                )
            },
            &cfg,
        );
    }
}

impl DeviceCM31Column {
    pub fn from_cpu(values: &[CM31]) -> Self {
        let length = values.len();
        let mut data: DeviceVec<CM31> = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&values));
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Column<CM31> for DeviceCM31Column {
    fn zeros(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        let host_data = vec![CM31::zero(); length];
        data.copy_from_host(HostSlice::from_slice(&host_data));

        Self { data }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        Self { data }
    }

    fn to_cpu(&self) -> Vec<CM31> {
        let mut result = Vec::<CM31>::with_capacity(self.data.len());
        self.data
            .copy_to_host(HostSlice::from_mut_slice(result.as_mut_slice()));
        result
    }

    fn len(&self) -> usize {
        self.data.len()
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

        Self { data }
    }
}

impl IntoIterator for DeviceCM31Column {
    type Item = CM31;
    type IntoIter = std::vec::IntoIter<CM31>;

    /// Creates a consuming iterator over the evaluations.
    ///
    /// Evaluations are returned in the same order as elements of the domain.
    fn into_iter(self) -> Self::IntoIter {
        // todo!()
        self.to_cpu().into_iter()
    }
}

/// An efficient structure for storing and operating on a arbitrary number of [`SecureField`]
/// values.
pub struct DeviceSecureColumn {
    pub data: DeviceVec<SecureField>,
}

impl PartialEq for DeviceSecureColumn {
    fn eq(&self, other: &Self) -> bool {
        let cfg = VecOpsConfig::default();
        are_bytes_equal::<QM31>(self.data.deref(), other.data.deref(), &cfg)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Clone for DeviceSecureColumn {
    fn clone(&self) -> Self {
        let mut data = DeviceVec::cuda_malloc(self.data.len()).unwrap();
        data.copy_from_device(&self.data);
        Self { data }
    }
}

impl Debug for DeviceSecureColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_cpu();
        f.debug_struct("DeviceColumn")
            .field("data", &data.as_slice())
            .field("length", &self.data.len())
            .finish()
    }
}

impl ColumnOps<SecureField> for IcicleBackend {
    type Column = DeviceSecureColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        let column_transmuted: &mut DeviceSlice<SecureField> = column.data.deref_mut();
        let cfg = BitReverseConfig::default();
        bit_reverse_inplace(
            unsafe {
                transmute::<&mut DeviceSlice<SecureField>, &mut DeviceSlice<QuarticExtensionField>>(
                    column_transmuted,
                )
            },
            &cfg,
        );
    }
}

impl DeviceSecureColumn {
    pub fn from_cpu(values: &[SecureField]) -> Self {
        let mut data: DeviceVec<SecureField> = DeviceVec::cuda_malloc(values.len()).unwrap();
        data.copy_from_host(HostSlice::from_slice(&values));
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Column<SecureField> for DeviceSecureColumn {
    fn zeros(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        let host_data = vec![SecureField::zero(); length];
        data.copy_from_host(HostSlice::from_slice(&host_data));

        Self { data }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = DeviceVec::cuda_malloc(length).unwrap();

        Self { data }
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        let mut result = Vec::<SecureField>::with_capacity(self.data.len());
        self.data
            .copy_to_host(HostSlice::from_mut_slice(result.as_mut_slice()));
        result
    }

    fn len(&self) -> usize {
        self.data.len()
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

        Self { data }
    }
}

impl IntoIterator for DeviceSecureColumn {
    type Item = QM31;
    type IntoIter = std::vec::IntoIter<QM31>;

    /// Creates a consuming iterator over the evaluations.
    ///
    /// Evaluations are returned in the same order as elements of the domain.
    fn into_iter(self) -> Self::IntoIter {
        // todo!()
        self.to_cpu().into_iter()
    }
}

impl SecureColumnByCoords<IcicleBackend> {
    pub fn packed_len(&self) -> usize {
        self.columns[0].data.len()
    }

    pub fn to_vec(&self) -> Vec<SecureField> {
        // todo!("convert on device");
        izip!(
            self.columns[0].to_cpu(),
            self.columns[1].to_cpu(),
            self.columns[2].to_cpu(),
            self.columns[3].to_cpu(),
        )
        .map(|(a, b, c, d)| SecureField::from_m31_array([a, b, c, d]))
        .collect()
    }

    pub fn icicle_from_cpu(cpu: SecureColumnByCoords<CpuBackend>) -> Self {
        Self {
            columns: cpu
                .columns
                .map(|arg0: Vec<BaseField>| DeviceColumn::from_cpu(&arg0)),
        }
    }
}

impl FromIterator<SecureField> for SecureColumnByCoords<IcicleBackend> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let cpu_col = SecureColumnByCoords::<CpuBackend>::from_iter(iter);
        let columns = cpu_col.columns.map(|col| DeviceColumn::from_cpu(&col));
        SecureColumnByCoords { columns }
    }
}
