use std::mem::transmute;

use super::IcicleBackend;
use crate::core::air::accumulation::AccumulationOps;
use crate::core::fields::m31::BaseField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::qm31::SecureField;
use crate::core::backend::CpuBackend;
use icicle_core::vec_ops::accumulate_scalars_stwo;
use icicle_cuda_runtime::memory::{HostSlice, DeviceVec, DeviceSlice, HostOrDeviceSlice};
use icicle_m31::field::ScalarField;

impl AccumulationOps for IcicleBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        let transmuted_cols = &column.columns.iter().map(|coord_col| {
            unsafe { transmute::<*const BaseField, *const ScalarField>(coord_col.data.as_ptr()) }
        })
        .collect::<Vec<*const ScalarField>>();
        
        let transmuted_others = &other.columns.iter().map(|coord_col| {
            unsafe { transmute::<*const BaseField, *const ScalarField>(coord_col.data.as_ptr()) }
        })
        .collect::<Vec<*const ScalarField>>();

        accumulate_scalars_stwo(
            HostSlice::from_slice(&transmuted_cols),
            HostSlice::from_slice(&transmuted_others), 
            column.len() as u32
        ).unwrap();
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        CpuBackend::generate_secure_powers(felt, n_powers)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::backend::icicle::column::DeviceColumn;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::{air::accumulation::AccumulationOps, backend::icicle::IcicleBackend, fields::{qm31::SecureField, secure_column::SecureColumnByCoords}};
    use num_traits::{Zero, One};

    #[cfg(feature = "icicle")]
    #[test]
    fn test_accumulate() {
        let a_h = vec![SecureField::zero(); 8];
        let mut column = SecureColumnByCoords::from_iter(a_h.clone());
        let mut cpu_column = SecureColumnByCoords::from_iter(a_h);

        column.columns.iter().for_each(|coord_col: &DeviceColumn| {
            let cpu_vec: Vec<BaseField> = <DeviceColumn as Column<BaseField>>::to_cpu(coord_col);
        });
        
        let other_h = vec![SecureField::one(); 8];
        let mut other = SecureColumnByCoords::from_iter(other_h.clone());
        let mut cpu_other = SecureColumnByCoords::from_iter(other_h);
    
        <CpuBackend as AccumulationOps>::accumulate(&mut cpu_column, &mut cpu_other);

        <IcicleBackend as AccumulationOps>::accumulate(&mut column, &mut other);

        assert_eq!(column.to_vec(), cpu_column.to_vec());
    }
}
