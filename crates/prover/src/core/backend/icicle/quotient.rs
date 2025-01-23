use std::mem::transmute;

use icicle_core::field::{Field, Field as IcicleField};
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use icicle_m31::field::{QuarticExtensionField, ScalarCfg};
use icicle_m31::quotient;

use super::IcicleBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation};
use crate::core::poly::BitReversedOrder;

impl QuotientOps for IcicleBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // unsafe {
        //     transmute(CpuBackend::accumulate_quotients(
        //         domain,
        //         unsafe { transmute(columns) },
        //         random_coeff,
        //         sample_batches,
        //         log_blowup_factor,
        //     ))
        // }

        let total_columns_size = columns
            .iter()
            .fold(0, |acc, column| acc + column.values.len());
        let mut icicle_device_columns = DeviceVec::cuda_malloc(total_columns_size).unwrap();
        let mut start = 0;
        nvtx::range_push!("[ICICLE] columns to device");
        columns.iter().for_each(|column| {
            let end = start + column.values.len();
            let device_slice = &mut icicle_device_columns[start..end];
            let transmuted: Vec<IcicleField<1, ScalarCfg>> =
                unsafe { transmute(column.values.clone()) };
            device_slice.copy_from_host(&HostSlice::from_slice(&transmuted));
            start += column.values.len();
        });
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] column sample batch");
        let icicle_sample_batches = sample_batches
            .into_iter()
            .map(|sample| {
                let (columns, values) = sample
                    .columns_and_values
                    .iter()
                    .map(|(index, value)| {
                        ((*index) as u32, unsafe {
                            transmute::<QM31, QuarticExtensionField>(*value)
                        })
                    })
                    .unzip();

                quotient::ColumnSampleBatch {
                    point: unsafe { transmute(sample.point) },
                    columns,
                    values,
                }
            })
            .collect_vec();
        nvtx::range_pop!();

        let mut icicle_result_raw = vec![QuarticExtensionField::zero(); domain.size()];
        let icicle_result = HostSlice::from_mut_slice(icicle_result_raw.as_mut_slice());
        let cfg = quotient::QuotientConfig::default();

        nvtx::range_push!("[ICICLE] accumulate_quotients_wrapped");
        quotient::accumulate_quotients_wrapped(
            domain.half_coset.initial_index.0 as u32,
            domain.half_coset.step_size.0 as u32,
            domain.log_size() as u32,
            &icicle_device_columns[..],
            unsafe { transmute(random_coeff) },
            &icicle_sample_batches,
            icicle_result,
            &cfg,
        );
        nvtx::range_pop!();
        // TODO: make it on cuda side
        nvtx::range_push!("[ICICLE] res to SecureEvaluation");
        let mut result = unsafe { SecureColumnByCoords::uninitialized(domain.size()) };
        (0..domain.size()).for_each(|i| result.set(i, unsafe { transmute(icicle_result_raw[i]) }));
        let ret = SecureEvaluation::new(domain, result);
        nvtx::range_pop!();

        ret
    }
}
