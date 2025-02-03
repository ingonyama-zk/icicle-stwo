use std::mem::transmute;
use std::ops::DerefMut;

use icicle_core::field::{Field, Field as IcicleField};
use icicle_core::ntt::FieldImpl;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_cuda_runtime::stream::CudaStream;
use icicle_m31::field::{QuarticExtensionField, ScalarCfg, ScalarField};
use icicle_m31::quotient::{self, to_internal_column_batch, QuotientConfig};

use super::IcicleBackend;
use crate::core::backend::icicle::column::DeviceColumn;
use crate::core::backend::{Column, CpuBackend};
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

        // todo!("support device columns");

        let total_columns_size = columns
            .iter()
            .fold(0, |acc, column| acc + column.values.len());
        let mut ptr_columns: Vec<*const ScalarField> = Vec::with_capacity(columns.len());
        let mut start = 0;
        nvtx::range_push!("[ICICLE] columns to device");
        columns.iter().for_each(|column| {
            ptr_columns.push(unsafe { transmute(column.values.data.as_ptr()) });
        });
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] column sample batch");
        let icicle_sample_batches: Vec<_> = sample_batches
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
            .collect();
        let icicle_internal_sample_batches = to_internal_column_batch(&icicle_sample_batches);
        nvtx::range_pop!();

        let icicle_columns = HostSlice::from_slice(&ptr_columns);

        nvtx::range_push!("[ICICLE] allocate accumualtion results");
        let stream1 = CudaStream::create().unwrap();
        let mut icicle_device_result1 = unsafe { DeviceColumn::uninitialized_async(domain.size(), &stream1) };
        let stream2 = CudaStream::create().unwrap();
        let mut icicle_device_result2 = unsafe { DeviceColumn::uninitialized_async(domain.size(), &stream2) };
        let stream3 = CudaStream::create().unwrap();
        let mut icicle_device_result3 = unsafe { DeviceColumn::uninitialized_async(domain.size(), &stream3) };
        let stream4 = CudaStream::create().unwrap();
        let mut icicle_device_result4 = unsafe { DeviceColumn::uninitialized_async(domain.size(), &stream4) };

        stream1.synchronize().unwrap();
        stream2.synchronize().unwrap();
        stream3.synchronize().unwrap();
        stream4.synchronize().unwrap();
        
        stream1.destroy().unwrap();
        stream2.destroy().unwrap();
        stream3.destroy().unwrap();
        stream4.destroy().unwrap();

        let icicle_device_result_transmuted1: &mut DeviceSlice<BaseField> =
            icicle_device_result1.data.deref_mut();
        let icicle_device_result_transmuted2: &mut DeviceSlice<BaseField> =
            icicle_device_result2.data.deref_mut();
        let icicle_device_result_transmuted3: &mut DeviceSlice<BaseField> =
            icicle_device_result3.data.deref_mut();
        let icicle_device_result_transmuted4: &mut DeviceSlice<BaseField> =
            icicle_device_result4.data.deref_mut();
        nvtx::range_pop!();

        let mut cfg = QuotientConfig::default();

        nvtx::range_push!("[ICICLE] accumulate_quotients_wrapped");
        quotient::accumulate_quotients_wrapped(
            domain.log_size() as u32,
            icicle_columns,
            unsafe { transmute(random_coeff) },
            &icicle_internal_sample_batches,
            unsafe {
                transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                    icicle_device_result_transmuted1,
                )
            },
            unsafe {
                transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                    icicle_device_result_transmuted2,
                )
            },
            unsafe {
                transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                    icicle_device_result_transmuted3,
                )
            },
            unsafe {
                transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                    icicle_device_result_transmuted4,
                )
            },
            &cfg,
        );
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] res to SecureEvaluation");
        let res_vec = [
            icicle_device_result1,
            icicle_device_result2,
            icicle_device_result3,
            icicle_device_result4,
        ];
        let result = SecureColumnByCoords { columns: res_vec };
        nvtx::range_pop!();

        SecureEvaluation::new(domain, result)
    }
}

#[cfg(test)]

pub(crate) mod tests {
    use crate::core::backend::cpu::CpuCirclePoly;
    use crate::core::backend::icicle::circle::IcicleCirclePoly;
    use crate::core::backend::icicle::column::DeviceColumn;
    use crate::core::backend::icicle::IcicleBackend;
    use crate::core::backend::CpuBackend;
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::{m31, qm31};

    #[test]
    fn test_icicle_quotients() {
        const LOG_SIZE: u32 = 19;
        const LOG_BLOWUP_FACTOR: u32 = 1;
        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + 1).circle_domain();
        let eval = polynomial.evaluate(eval_domain);

        let point = SECURE_FIELD_CIRCLE_GEN;
        let value = polynomial.eval_at_point(point);
        let coeff = qm31!(1, 2, 3, 4);
        let quot_eval_cpu = CpuBackend::accumulate_quotients(
            eval_domain,
            &[&eval],
            coeff,
            &[ColumnSampleBatch {
                point,
                columns_and_values: vec![(0, value)],
            }],
            LOG_BLOWUP_FACTOR,
        )
        .to_vec();

        let eval_icicle = CircleEvaluation::new(eval_domain, DeviceColumn::from_cpu(&eval));
        let quot_eval_icicle = IcicleBackend::accumulate_quotients(
            eval_domain,
            &[&eval_icicle],
            coeff,
            &[ColumnSampleBatch {
                point,
                columns_and_values: vec![(0, value)],
            }],
            LOG_BLOWUP_FACTOR,
        )
        .to_vec();
        assert_eq!(quot_eval_cpu, quot_eval_icicle);
    }
}
