use std::mem::transmute;
use std::ops::{Deref, DerefMut};

use icicle_core::ntt::FieldImpl;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostSlice};
use icicle_m31::field::{QuarticExtensionField, ScalarField};
use icicle_m31::fri::{self, fold_circle_into_line, fold_circle_into_line_new, FriConfig};

use super::IcicleBackend;
use crate::core::backend::icicle::column::{self, DeviceColumn};
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::FieldExpOps;
use crate::core::fri::{FriOps, CIRCLE_TO_LINE_FOLD_STEP, FOLD_STEP};
use crate::core::poly::circle::SecureEvaluation;
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;

impl FriOps for IcicleBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        use crate::core::backend::icicle::IcicleBackend;
        let length = eval.values.len(); // TODO: same as n

        assert!(length >= 2, "Evaluation too small");

        let domain = eval.domain();

        let dom_vals_len = length / 2;

        nvtx::range_push!("[ICICLE] domain evals convert + move");
        let eval1: &DeviceSlice<BaseField> = eval.values.columns[0].data.deref();
        let eval2: &DeviceSlice<BaseField> = eval.values.columns[1].data.deref();
        let eval3: &DeviceSlice<BaseField> = eval.values.columns[2].data.deref();
        let eval4: &DeviceSlice<BaseField> = eval.values.columns[3].data.deref();
        let eval_slice1 =
            unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(eval1) };
        let eval_slice2 =
            unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(eval2) };
        let eval_slice3 =
            unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(eval3) };
        let eval_slice4 =
            unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(eval4) };
        nvtx::range_pop!();
        let mut d_folded_eval =
            DeviceVec::<QuarticExtensionField>::cuda_malloc(dom_vals_len).unwrap();

        let cfg = FriConfig::default();
        let icicle_alpha = unsafe { transmute(alpha) };

        let mut icicle_device_result1 = unsafe { DeviceColumn::uninitialized(domain.size() >> 1) };
        let mut icicle_device_result2 = unsafe { DeviceColumn::uninitialized(domain.size() >> 1) };
        let mut icicle_device_result3 = unsafe { DeviceColumn::uninitialized(domain.size() >> 1) };
        let mut icicle_device_result4 = unsafe { DeviceColumn::uninitialized(domain.size() >> 1) };

        let icicle_device_result_transmuted1: &mut DeviceSlice<BaseField> =
            icicle_device_result1.data.deref_mut();
        let icicle_device_result_transmuted2: &mut DeviceSlice<BaseField> =
            icicle_device_result2.data.deref_mut();
        let icicle_device_result_transmuted3: &mut DeviceSlice<BaseField> =
            icicle_device_result3.data.deref_mut();
        let icicle_device_result_transmuted4: &mut DeviceSlice<BaseField> =
            icicle_device_result4.data.deref_mut();

        let folded_eval1: &mut DeviceSlice<ScalarField> = unsafe {
            transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                icicle_device_result_transmuted1,
            )
        };
        let folded_eval2: &mut DeviceSlice<ScalarField> = unsafe {
            transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                icicle_device_result_transmuted2,
            )
        };
        let folded_eval3: &mut DeviceSlice<ScalarField> = unsafe {
            transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                icicle_device_result_transmuted3,
            )
        };
        let folded_eval4: &mut DeviceSlice<ScalarField> = unsafe {
            transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                icicle_device_result_transmuted4,
            )
        };

        nvtx::range_push!("[ICICLE] fold_line");
        let _ = fri::fold_line_new(
            eval_slice1,
            eval_slice2,
            eval_slice3,
            eval_slice4,
            domain.coset().initial_index.0 as u64,
            domain.log_size(),
            folded_eval1,
            folded_eval2,
            folded_eval3,
            folded_eval4,
            icicle_alpha,
            &cfg,
        )
        .unwrap();
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] convert to SecureColumnByCoords");
        let folded_values = SecureColumnByCoords {
            columns: [
                icicle_device_result1,
                icicle_device_result2,
                icicle_device_result3,
                icicle_device_result4,
            ],
        };

        let line_eval = LineEvaluation::new(domain.double(), folded_values);
        nvtx::range_pop!();

        line_eval
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        assert_eq!(src.len() >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

        let domain = src.domain;
        let length = src.values.len();

        nvtx::range_push!("[ICICLE] d_evals_icicle");
        let eval_vec1 = src.columns[0].data.deref();
        let eval_vec2 = src.columns[1].data.deref();
        let eval_vec3 = src.columns[2].data.deref();
        let eval_vec4 = src.columns[3].data.deref();
        let eval1 =
            unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(eval_vec1) };
        let eval2 =
            unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(eval_vec2) };
        let eval3 =
            unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(eval_vec3) };
        let eval4 =
            unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(eval_vec4) };
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] d_folded_evals");
        let mut iter = dst.values.columns.iter_mut();
        let icicle_device_result_transmuted1: &mut DeviceSlice<BaseField> =
            iter.next().unwrap().data.deref_mut();
        let icicle_device_result_transmuted2: &mut DeviceSlice<BaseField> =
            iter.next().unwrap().data.deref_mut();
        let icicle_device_result_transmuted3: &mut DeviceSlice<BaseField> =
            iter.next().unwrap().data.deref_mut();
        let icicle_device_result_transmuted4: &mut DeviceSlice<BaseField> =
            iter.next().unwrap().data.deref_mut();

        let folded_eval1: &mut DeviceSlice<ScalarField> = unsafe {
            transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                icicle_device_result_transmuted1,
            )
        };
        let folded_eval2: &mut DeviceSlice<ScalarField> = unsafe {
            transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                icicle_device_result_transmuted2,
            )
        };
        let folded_eval3: &mut DeviceSlice<ScalarField> = unsafe {
            transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                icicle_device_result_transmuted3,
            )
        };
        let folded_eval4: &mut DeviceSlice<ScalarField> = unsafe {
            transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(
                icicle_device_result_transmuted4,
            )
        };
        nvtx::range_pop!();

        let cfg = FriConfig::default();
        let icicle_alpha = unsafe { transmute(alpha) };

        nvtx::range_push!("[ICICLE] fold circle");
        let _ = fold_circle_into_line_new(
            eval1,
            eval2,
            eval3,
            eval4,
            domain.half_coset.initial_index.0 as u64,
            domain.half_coset.log_size,
            folded_eval1,
            folded_eval2,
            folded_eval3,
            folded_eval4,
            icicle_alpha,
            &cfg,
        )
        .unwrap();
        nvtx::range_pop!();
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        todo!()
        //unsafe { transmute(CpuBackend::decompose(unsafe { transmute(eval) })) }
    }
}

#[cfg(test)]

pub(crate) mod tests {
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::icicle::column::DeviceColumn;
    use crate::core::backend::icicle::IcicleBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;
    use crate::core::fri::FriOps;
    use crate::core::poly::circle::{CanonicCoset, PolyOps, SecureEvaluation};
    use crate::core::poly::line::{LineDomain, LineEvaluation};
    use crate::qm31;

    #[test]
    fn test_icicle_fold_line() {
        let mut is_correct = true;
        for log_size in 1..20 {
            let mut rng = SmallRng::seed_from_u64(0);
            let values = (0..1 << log_size).map(|_| rng.gen()).collect_vec();
            let alpha = qm31!(1, 3, 5, 7);
            let domain = LineDomain::new(CanonicCoset::new(log_size + 1).half_coset());

            let secure_column: SecureColumnByCoords<_> = values.iter().copied().collect();
            let line_evaluation: LineEvaluation<CpuBackend> =
                LineEvaluation::new(domain, secure_column);
            let cpu_fold = CpuBackend::fold_line(
                &line_evaluation,
                alpha,
                &CpuBackend::precompute_twiddles(domain.coset()),
            );

            let columns: [DeviceColumn; 4] = line_evaluation
                .values
                .columns
                .map(|arg| DeviceColumn::from_cpu(&arg));

            let line_evaluation = LineEvaluation::new(domain, SecureColumnByCoords { columns });
            let dummy_twiddles = IcicleBackend::precompute_twiddles(domain.coset());
            let icicle_fold = IcicleBackend::fold_line(&line_evaluation, alpha, &dummy_twiddles);

            if icicle_fold.values.to_vec() != cpu_fold.values.to_vec() {
                println!("failed to fold log2: {}", log_size);
                is_correct = false;
            }
        }
        assert!(is_correct);
    }

    #[test]
    fn test_icicle_fold_circle_into_line() {
        let mut is_correct = true;
        for log_size in 1..20 {
            let values: Vec<SecureField> = (0..(1 << log_size))
                .map(|i| qm31!(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
                .collect();
            let alpha = qm31!(1, 3, 5, 7);
            let circle_domain = CanonicCoset::new(log_size).circle_domain();
            let line_domain = LineDomain::new(circle_domain.half_coset);
            let mut cpu_fold = LineEvaluation::new(
                line_domain,
                SecureColumnByCoords::zeros(1 << (log_size - 1)),
            );
            let cpu_src = SecureEvaluation::new(circle_domain, values.iter().copied().collect());
            CpuBackend::fold_circle_into_line(
                &mut cpu_fold,
                &cpu_src,
                alpha,
                &CpuBackend::precompute_twiddles(line_domain.coset()),
            );

            let columns: [DeviceColumn; 4] = cpu_src
                .values
                .columns
                .map(|arg| DeviceColumn::from_cpu(&arg));

            let icicle_src = SecureEvaluation::new(circle_domain, SecureColumnByCoords { columns });
            let dummy_twiddles = IcicleBackend::precompute_twiddles(line_domain.coset());
            let mut icicle_fold = LineEvaluation::new(
                line_domain,
                SecureColumnByCoords::zeros(1 << (log_size - 1)),
            );

            IcicleBackend::fold_circle_into_line(
                &mut icicle_fold,
                &icicle_src,
                alpha,
                &dummy_twiddles,
            );

            assert_eq!(cpu_fold.values.to_vec(), icicle_fold.values.to_vec());

            if cpu_fold.values.to_vec() != icicle_fold.values.to_vec() {
                println!("failed to fold log2: {}", log_size);
                is_correct = false;
            }
        }
        assert!(is_correct);
    }
}
