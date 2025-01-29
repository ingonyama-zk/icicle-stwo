use std::mem::transmute;

use icicle_core::ntt::FieldImpl;
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use icicle_m31::field::{QuarticExtensionField, ScalarField};
use icicle_m31::fri::{self, fold_circle_into_line, FriConfig};

use super::IcicleBackend;
use crate::core::backend::CpuBackend;
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
        todo!()
        // use crate::core::backend::icicle::IcicleBackend;
        // let length = eval.values.len(); // TODO: same as n

        // let n = eval.len();
        // assert!(n >= 2, "Evaluation too small");

        // let domain = eval.domain();

        // let dom_vals_len = length / 2;

        // let mut domain_vals = Vec::new();
        // let line_domain_log_size = domain.log_size();
        // nvtx::range_push!("[ICICLE] calc domain values");
        // for i in 0..dom_vals_len {
        //     // TODO: on-device batch
        //     // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
        //     domain_vals.push(ScalarField::from_u32(
        //         domain
        //             .at(bit_reverse_index(i << FOLD_STEP, line_domain_log_size))
        //             .inverse()
        //             .0,
        //     ));
        // }
        // nvtx::range_pop!();

        // nvtx::range_push!("[ICICLE] domain values to device");
        // let domain_icicle_host = HostSlice::from_slice(domain_vals.as_slice());
        // let mut d_domain_icicle = DeviceVec::<ScalarField>::cuda_malloc(dom_vals_len).unwrap();
        // d_domain_icicle.copy_from_host(domain_icicle_host).unwrap();
        // nvtx::range_pop!();

        // nvtx::range_push!("[ICICLE] domain evals convert + move");
        // let mut d_evals_icicle = DeviceVec::<QuarticExtensionField>::cuda_malloc(length).unwrap();
        // SecureColumnByCoords::<IcicleBackend>::convert_to_icicle(
        //     unsafe { transmute(&eval.values) },
        //     &mut d_evals_icicle,
        // );
        // nvtx::range_pop!();
        // let mut d_folded_eval =
        //     DeviceVec::<QuarticExtensionField>::cuda_malloc(dom_vals_len).unwrap();

        // let cfg = FriConfig::default();
        // let icicle_alpha = unsafe { transmute(alpha) };
        // nvtx::range_push!("[ICICLE] fold_line");
        // let _ = fri::fold_line(
        //     &d_evals_icicle[..],
        //     &d_domain_icicle[..],
        //     &mut d_folded_eval[..],
        //     icicle_alpha,
        //     &cfg,
        // )
        // .unwrap();
        // nvtx::range_pop!();

        // nvtx::range_push!("[ICICLE] convert to SecureColumnByCoords");
        // let mut folded_values = unsafe { SecureColumnByCoords::uninitialized(dom_vals_len) };
        // SecureColumnByCoords::<IcicleBackend>::convert_from_icicle_q31(
        //     &mut folded_values,
        //     &mut d_folded_eval[..],
        // );

        // let line_eval = LineEvaluation::new(domain.double(), folded_values);
        // nvtx::range_pop!();

        // line_eval
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        todo!()
    //     assert_eq!(src.len() >> CIRCLE_TO_LINE_FOLD_STEP, dst.len());

    //     let domain = src.domain;
    //     let length = src.values.len();

    //     let dom_vals_len = length / 2;
    //     let _domain_log_size = domain.log_size();

    //     let mut domain_rev = Vec::new();
    //     nvtx::range_push!("[ICICLE] domain_rev");
    //     for i in 0..dom_vals_len {
    //         // TODO: on-device batch
    //         // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
    //         let p = domain.at(bit_reverse_index(
    //             i << CIRCLE_TO_LINE_FOLD_STEP,
    //             domain.log_size(),
    //         ));
    //         let p = p.y.inverse();
    //         domain_rev.push(p);
    //     }
    //     nvtx::range_pop!();

    //     nvtx::range_push!("[ICICLE] domain_vals");
    //     let domain_vals = (0..dom_vals_len)
    //         .map(|i| {
    //             let p = domain_rev[i];
    //             ScalarField::from_u32(p.0)
    //         })
    //         .collect::<Vec<_>>();
    //     nvtx::range_pop!();

    //     nvtx::range_push!("[ICICLE] domain to device");
    //     let domain_icicle_host = HostSlice::from_slice(domain_vals.as_slice());
    //     let mut d_domain_icicle = DeviceVec::<ScalarField>::cuda_malloc(dom_vals_len).unwrap();
    //     d_domain_icicle.copy_from_host(domain_icicle_host).unwrap();
    //     nvtx::range_pop!();

    //     nvtx::range_push!("[ICICLE] d_evals_icicle");
    //     let mut d_evals_icicle = DeviceVec::<QuarticExtensionField>::cuda_malloc(length).unwrap();
    //     SecureColumnByCoords::convert_to_icicle(&src.values, &mut d_evals_icicle);
    //     nvtx::range_pop!();

    //     nvtx::range_push!("[ICICLE] d_folded_evals");
    //     let mut d_folded_eval =
    //         DeviceVec::<QuarticExtensionField>::cuda_malloc(dom_vals_len).unwrap();
    //     SecureColumnByCoords::convert_to_icicle(&dst.values, &mut d_folded_eval);
    //     nvtx::range_pop!();

    //     let mut folded_eval_raw = vec![QuarticExtensionField::zero(); dom_vals_len];
    //     let folded_eval = HostSlice::from_mut_slice(folded_eval_raw.as_mut_slice());

    //     let cfg = FriConfig::default();
    //     let icicle_alpha = unsafe { transmute(alpha) };

    //     nvtx::range_push!("[ICICLE] fold circle");
    //     let _ = fold_circle_into_line(
    //         &d_evals_icicle[..],
    //         &d_domain_icicle[..],
    //         &mut d_folded_eval[..],
    //         icicle_alpha,
    //         &cfg,
    //     )
    //     .unwrap();
    //     nvtx::range_pop!();

    //     d_folded_eval.copy_to_host(folded_eval).unwrap();

    //     nvtx::range_push!("[ICICLE] convert to SecureColumnByCoords");
    //     SecureColumnByCoords::convert_from_icicle_q31(&mut dst.values, &mut d_folded_eval[..]);
    //     nvtx::range_pop!();
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // todo!()
        unsafe { transmute(CpuBackend::decompose(unsafe { transmute(eval) })) }
    }
}
