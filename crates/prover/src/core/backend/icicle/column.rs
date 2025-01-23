use std::iter::zip;
use std::{array, mem};

use bytemuck::allocation::cast_vec;
use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use itertools::{izip, Itertools};
use num_traits::Zero;

use super::IcicleBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::{SecureColumnByCoords, SECURE_EXTENSION_DEGREE};
use crate::core::fields::{FieldExpOps, FieldOps};

impl FieldOps<SecureField> for IcicleBackend {
    fn batch_inverse(column: &DeviceSecureColumn, dst: &mut DeviceSecureColumn) {
        SecureField::batch_inverse(&column.data, &mut dst.data);
    }
}

impl<T: Debug + Clone + Default> ColumnOps<T> for IcicleBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        // todo!()
        CpuBackend::bit_reverse_column(column)
    }
}

// A column that is stored on device
pub struct DeviceColumn {
    pub data: DeviceVec<BaseField>,
    /// The number of [`BaseField`]s in the vector.
    pub length: usize,
}

impl DeviceColumn {
    /// Extracts a slice containing the entire vector of [`BaseField`]s.
    pub fn as_slice(&self) -> &[BaseField] {
        &cast_slice(&self.data)[..self.length]
    }

    /// Extracts a mutable slice containing the entire vector of [`BaseField`]s.
    pub fn as_mut_slice(&mut self) -> &mut [BaseField] {
        &mut cast_slice_mut(&mut self.data)[..self.length]
    }

    pub fn into_cpu_vec(mut self) -> Vec<BaseField> {
        let capacity = self.data.capacity() * N_LANES;
        let length = self.length;
        let ptr = self.data.as_mut_ptr() as *mut BaseField;
        let res = unsafe { Vec::from_raw_parts(ptr, length, capacity) };
        mem::forget(self);
        res
    }

    pub fn from_cpu(values: Vec<BaseField>) -> Self {
        values.into_iter().collect()
    }

    /// Returns a vector of `BaseColumnMutSlice`s, each mutably owning
    /// `chunk_size` `PackedBaseField`s (i.e, `chuck_size` * `N_LANES` elements).
    pub fn chunks_mut(&mut self, chunk_size: usize) -> Vec<BaseColumnMutSlice<'_>> {
        self.data
            .chunks_mut(chunk_size)
            .map(BaseColumnMutSlice)
            .collect_vec()
    }

    pub fn into_secure_column(self) -> SecureColumn {
        let length = self.len();
        let data = self.data.into_iter().map(PackedSecureField::from).collect();
        SecureColumn { data, length }
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
        let length = iter.len();
        let host_data = iter.collect_vec();
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&host_data))
            .unwrap();

        Self { data, length }
    }
}

// A efficient structure for storing and operating on a arbitrary number of [`SecureField`] values.
#[derive(Clone, Debug, Default)]
pub struct DeviceCM31Column {
    pub data: DeviceVec<CM31>,
    pub length: usize,
}

impl Column<CM31> for DeviceCM31Column {
    fn zeros(length: usize) -> Self {
        Self {
            data: vec![PackedCM31::zeroed(); length.div_ceil(N_LANES)],
            length,
        }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = Vec::with_capacity(length.div_ceil(N_LANES));
        data.set_len(length.div_ceil(N_LANES));
        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<CM31> {
        self.data
            .iter()
            .flat_map(|x| x.to_array())
            .take(self.length)
            .collect()
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> CM31 {
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }

    fn set(&mut self, index: usize, value: CM31) {
        let mut packed = self.data[index / N_LANES].to_array();
        packed[index % N_LANES] = value;
        self.data[index / N_LANES] = PackedCM31::from_array(packed)
    }
}

impl FromIterator<CM31> for DeviceCM31Column {
    fn from_iter<I: IntoIterator<Item = CM31>>(iter: I) -> Self {
        let length = iter.len();
        let host_data = iter.collect_vec();
        let mut data = DeviceVec::cuda_malloc(length).unwrap();
        data.copy_from_host(HostSlice::from_slice(&host_data))
            .unwrap();

        Self { data, length }
    }
}

// /// A mutable slice of a BaseColumn.
// pub struct BaseColumnMutSlice<'a>(pub &'a mut [PackedBaseField]);

// impl<'a> BaseColumnMutSlice<'a> {
//     pub fn at(&self, index: usize) -> BaseField {
//         self.0[index / N_LANES].to_array()[index % N_LANES]
//     }

//     pub fn set(&mut self, index: usize, value: BaseField) {
//         let mut packed = self.0[index / N_LANES].to_array();
//         packed[index % N_LANES] = value;
//         self.0[index / N_LANES] = PackedBaseField::from_array(packed)
//     }
// }

/// An efficient structure for storing and operating on a arbitrary number of [`SecureField`]
/// values.
#[derive(Clone, Debug, Default)]
pub struct DeviceSecureColumn {
    pub data: Vec<SecureField>,
    /// The number of [`SecureField`]s in the vector.
    pub length: usize,
}

impl DeviceSecureColumn {
    // Separates a single column of `PackedSecureField` elements into `SECURE_EXTENSION_DEGREE` many
    // `PackedBaseField` coordinate columns.
    pub fn into_secure_column_by_coords(self) -> SecureColumnByCoords<SimdBackend> {
        if self.len() < N_LANES {
            return self.to_cpu().into_iter().collect();
        }

        let length = self.length;
        let packed_length = self.data.len();
        let mut columns = array::from_fn(|_| Vec::with_capacity(packed_length));

        for v in self.data {
            let packed_coords = v.into_packed_m31s();
            zip(&mut columns, packed_coords).for_each(|(col, packed_coord)| col.push(packed_coord));
        }

        SecureColumnByCoords {
            columns: columns.map(|col| BaseColumn { data: col, length }),
        }
    }
}

impl Column<SecureField> for DeviceSecureColumn {
    fn zeros(length: usize) -> Self {
        Self {
            data: vec![PackedSecureField::zeroed(); length.div_ceil(N_LANES)],
            length,
        }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = Vec::with_capacity(length.div_ceil(N_LANES));
        data.set_len(length.div_ceil(N_LANES));
        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        self.data
            .iter()
            .flat_map(|x| x.to_array())
            .take(self.length)
            .collect()
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> SecureField {
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }

    fn set(&mut self, index: usize, value: SecureField) {
        let mut packed = self.data[index / N_LANES].to_array();
        packed[index % N_LANES] = value;
        self.data[index / N_LANES] = PackedSecureField::from_array(packed)
    }
}

impl FromIterator<SecureField> for DeviceSecureColumn {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut data = (&mut chunks)
            .map(PackedSecureField::from_array)
            .collect_vec();
        let mut length = data.len() * N_LANES;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let mut last = [SecureField::zero(); N_LANES];
                last[..remainder.len()].copy_from_slice(remainder.as_slice());
                data.push(PackedSecureField::from_array(last));
            }
        }

        Self { data, length }
    }
}

impl FromIterator<PackedSecureField> for DeviceSecureColumn {
    fn from_iter<I: IntoIterator<Item = PackedSecureField>>(iter: I) -> Self {
        let data = iter.into_iter().collect_vec();
        let length = data.len() * N_LANES;
        Self { data, length }
    }
}

// /// A mutable slice of a SecureColumnByCoords.
// pub struct SecureColumnByCoordsMutSlice<'a>(pub [BaseColumnMutSlice<'a>;
// SECURE_EXTENSION_DEGREE]);

// impl<'a> SecureColumnByCoordsMutSlice<'a> {
//     /// # Safety
//     ///
//     /// `vec_index` must be a valid index.
//     pub unsafe fn packed_at(&self, vec_index: usize) -> PackedSecureField {
//         PackedQM31([
//             PackedCM31([
//                 *self.0[0].0.get_unchecked(vec_index),
//                 *self.0[1].0.get_unchecked(vec_index),
//             ]),
//             PackedCM31([
//                 *self.0[2].0.get_unchecked(vec_index),
//                 *self.0[3].0.get_unchecked(vec_index),
//             ]),
//         ])
//     }

//     /// # Safety
//     ///
//     /// `vec_index` must be a valid index.
//     pub unsafe fn set_packed(&mut self, vec_index: usize, value: PackedSecureField) {
//         let PackedQM31([PackedCM31([a, b]), PackedCM31([c, d])]) = value;
//         *self.0[0].0.get_unchecked_mut(vec_index) = a;
//         *self.0[1].0.get_unchecked_mut(vec_index) = b;
//         *self.0[2].0.get_unchecked_mut(vec_index) = c;
//         *self.0[3].0.get_unchecked_mut(vec_index) = d;
//     }
// }

impl SecureColumnByCoords<IcicleBackend> {
    pub fn packed_len(&self) -> usize {
        self.columns[0].data.len()
    }

    /// # Safety
    ///
    /// `vec_index` must be a valid index.
    pub unsafe fn packed_at(&self, vec_index: usize) -> PackedSecureField {
        PackedQM31([
            PackedCM31([
                *self.columns[0].data.get_unchecked(vec_index),
                *self.columns[1].data.get_unchecked(vec_index),
            ]),
            PackedCM31([
                *self.columns[2].data.get_unchecked(vec_index),
                *self.columns[3].data.get_unchecked(vec_index),
            ]),
        ])
    }

    /// # Safety
    ///
    /// `vec_index` must be a valid index.
    pub unsafe fn set_packed(&mut self, vec_index: usize, value: PackedSecureField) {
        let PackedQM31([PackedCM31([a, b]), PackedCM31([c, d])]) = value;
        *self.columns[0].data.get_unchecked_mut(vec_index) = a;
        *self.columns[1].data.get_unchecked_mut(vec_index) = b;
        *self.columns[2].data.get_unchecked_mut(vec_index) = c;
        *self.columns[3].data.get_unchecked_mut(vec_index) = d;
    }

    pub fn to_vec(&self) -> Vec<SecureField> {
        izip!(
            self.columns[0].to_cpu(),
            self.columns[1].to_cpu(),
            self.columns[2].to_cpu(),
            self.columns[3].to_cpu(),
        )
        .map(|(a, b, c, d)| SecureField::from_m31_array([a, b, c, d]))
        .collect()
    }

    /// Returns a vector of `SecureColumnByCoordsMutSlice`s, each mutably owning
    /// `SECURE_EXTENSION_DEGREE` slices of `chunk_size` `PackedBaseField`s
    /// (i.e, `chuck_size` * `N_LANES` secure field elements, by coordinates).
    pub fn chunks_mut(&mut self, chunk_size: usize) -> Vec<SecureColumnByCoordsMutSlice<'_>> {
        let [a, b, c, d] = self
            .columns
            .get_many_mut([0, 1, 2, 3])
            .unwrap()
            .map(|x| x.chunks_mut(chunk_size));
        izip!(a, b, c, d)
            .map(|(a, b, c, d)| SecureColumnByCoordsMutSlice([a, b, c, d]))
            .collect_vec()
    }

    pub fn from_cpu(cpu: SecureColumnByCoords<CpuBackend>) -> Self {
        Self {
            columns: cpu.columns.map(BaseColumn::from_cpu),
        }
    }
}

impl FromIterator<SecureField> for SecureColumnByCoords<IcicleBackend> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let cpu_col = SecureColumnByCoords::<CpuBackend>::from_iter(iter);
        let columns = cpu_col.columns.map(|col| col.into_iter().collect());
        SecureColumnByCoords { columns }
    }
}

pub struct SecureColumnByCoordsIter<'a> {
    column: &'a SecureColumnByCoords<IcicleBackend>,
    index: usize,
}
impl Iterator for SecureColumnByCoordsIter<'_> {
    type Item = SecureField;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.column.len() {
            let value = self.column.at(self.index);
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }
}
impl<'a> IntoIterator for &'a SecureColumnByCoords<IcicleBackend> {
    type Item = SecureField;
    type IntoIter = SecureColumnByCoordsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SecureColumnByCoordsIter {
            column: self,
            index: 0,
        }
    }
}

impl FromIterator<SecureField> for SecureColumnByCoords<IcicleBackend> {
    // TODO: just stub - ideally not [m31; 4] layout - and no conversion
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let values = iter.into_iter();
        let (lower_bound, _) = values.size_hint();
        let mut columns = array::from_fn(|_| Vec::with_capacity(lower_bound));

        for value in values {
            let coords = value.to_m31_array();
            zip(&mut columns, coords).for_each(|(col, coord)| col.push(coord));
        }

        SecureColumnByCoords { columns }
    }
}

impl SecureColumnByCoords<IcicleBackend> {
    // TODO(first): Remove.
    pub fn to_vec(&self) -> Vec<SecureField> {
        (0..self.len()).map(|i| self.at(i)).collect()
    }
}

impl SecureColumnByCoords<IcicleBackend> {
    pub fn convert_to_icicle(input: &Self, d_output: &mut DeviceSlice<QuarticExtensionField>) {
        let a: &[u32] = unsafe { transmute(input.columns[0].as_slice()) };
        let b: &[u32] = unsafe { transmute(input.columns[1].as_slice()) };
        let c: &[u32] = unsafe { transmute(input.columns[2].as_slice()) };
        let d: &[u32] = unsafe { transmute(input.columns[3].as_slice()) };

        let a = HostSlice::from_slice(&a);
        let b = HostSlice::from_slice(&b);
        let c = HostSlice::from_slice(&c);
        let d = HostSlice::from_slice(&d);

        let _ = stwo_convert(a, b, c, d, d_output).unwrap();
    }

    pub fn convert_from_icicle(input: &mut Self, d_input: &mut DeviceSlice<ScalarField>) {
        let zero = ScalarField::zero();

        let n = input.columns[0].len();
        let secure_degree = input.columns.len();
        let mut intermediate_host = vec![zero; secure_degree * n];

        let mut result_tr: DeviceVec<ScalarField> =
            DeviceVec::cuda_malloc(secure_degree * n).unwrap();

        transpose_matrix(
            d_input,
            secure_degree as u32,
            n as u32,
            &mut result_tr[..],
            &DeviceContext::default(),
            true,
            false,
        )
        .unwrap();

        let res_host = HostSlice::from_mut_slice(&mut intermediate_host[..]);
        result_tr.copy_to_host(res_host).unwrap();

        let res: Vec<BaseField> = unsafe { transmute(intermediate_host) };

        // Assign the sub-slices to the column
        for i in 0..secure_degree {
            let start = i * n;
            let end = start + n;

            input.columns[i].truncate(0);
            input.columns[i].extend_from_slice(&res[start..end]);
        }
    }

    pub fn convert_from_icicle_q31(
        // TODO: remove as convert_from_icicle should perform same on device via transpose just on
        output: &mut SecureColumnByCoords<IcicleBackend>,
        d_input: &mut DeviceSlice<QuarticExtensionField>,
    ) {
        Self::convert_from_icicle(output, unsafe { transmute(d_input) });
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::DeviceColumn;
    use crate::core::backend::icicle::column::DeviceSecureColumn;
    use crate::core::backend::Column;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;

    #[test]
    fn base_field_vec_from_iter_works() {
        let values: [BaseField; 30] = array::from_fn(BaseField::from);

        let res = values.into_iter().collect::<DeviceColumn>();

        assert_eq!(res.to_cpu(), values);
    }

    #[test]
    fn secure_field_vec_from_iter_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values: [SecureField; 30] = rng.gen();

        let res = values.into_iter().collect::<DeviceSecureColumn>();

        assert_eq!(res.to_cpu(), values);
    }

    #[test]
    fn test_base_column_chunks_mut() {
        let values: [BaseField; N_LANES * 7] = array::from_fn(BaseField::from);
        let mut col = values.into_iter().collect::<BaseColumn>();

        const CHUNK_SIZE: usize = 2;
        let mut chunks = col.chunks_mut(CHUNK_SIZE);
        chunks[2].set(19, BaseField::from(1234));
        chunks[3].set(1, BaseField::from(5678));

        assert_eq!(col.at(2 * CHUNK_SIZE * N_LANES + 19), BaseField::from(1234));
        assert_eq!(col.at(3 * CHUNK_SIZE * N_LANES + 1), BaseField::from(5678));
    }

    #[test]
    fn test_secure_column_by_coords_chunks_mut() {
        const COL_PACKED_SIZE: usize = 16;
        let a: [BaseField; N_LANES * COL_PACKED_SIZE] = array::from_fn(BaseField::from);
        let b: [BaseField; N_LANES * COL_PACKED_SIZE] = array::from_fn(BaseField::from);
        let c: [BaseField; N_LANES * COL_PACKED_SIZE] = array::from_fn(BaseField::from);
        let d: [BaseField; N_LANES * COL_PACKED_SIZE] = array::from_fn(BaseField::from);
        let mut col = SecureColumnByCoords {
            columns: [a, b, c, d].map(|values| values.into_iter().collect::<BaseColumn>()),
        };

        let mut rng = SmallRng::seed_from_u64(0);
        let rand0 = PackedQM31::from_array(rng.gen());
        let rand1 = PackedQM31::from_array(rng.gen());

        const CHUNK_SIZE: usize = 4;
        let mut chunks = col.chunks_mut(CHUNK_SIZE);
        unsafe {
            chunks[2].set_packed(3, rand0);
            chunks[3].set_packed(1, rand1);

            assert_eq!(
                col.packed_at(2 * CHUNK_SIZE + 3).to_array(),
                rand0.to_array()
            );
            assert_eq!(
                col.packed_at(3 * CHUNK_SIZE + 1).to_array(),
                rand1.to_array()
            );
        }
    }
}
