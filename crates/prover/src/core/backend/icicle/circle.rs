use std::mem::transmute;

use icicle_core::ntt::{FieldImpl, NTTConfig, Ordering};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostSlice, HostOrDeviceSlice};
use icicle_m31::dcct::{self, get_dcct_root_of_unity, initialize_dcct_domain};
use icicle_m31::field::ScalarField;
use itertools::Itertools;

use super::IcicleBackend;
use crate::core::backend::icicle::column::DeviceColumn;
use crate::core::backend::{Col, Column, CpuBackend};
use crate::core::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
};
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

pub(crate) type IcicleCirclePoly = CirclePoly<IcicleBackend>;
pub(crate) type IcicleCircleEvaluation<F, EvalOrder> =
    CircleEvaluation<IcicleBackend, F, EvalOrder>;
// type CpuMle<F> = Mle<CpuBackend, F>;

impl PolyOps for IcicleBackend {
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        todo!("device bit reverse");
        // unsafe { transmute(CpuBackend::new_canonical_ordered(coset, values)) }
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        if eval.domain.log_size() <= 3 || eval.domain.log_size() == 7 {
            let cpu_eval = CpuCircleEvaluation::new(eval.domain, eval.values.to_cpu());

            let cpu_circle_poly = CpuBackend::interpolate(
                cpu_eval,
                unsafe { transmute(itwiddles) },
            );

            let icicle_coeffs = DeviceColumn::from_cpu(cpu_circle_poly.coeffs.as_slice());

            return IcicleCirclePoly::new(icicle_coeffs);
        }

        nvtx::range_push!("[ICICLE] get_dcct_root_of_unity");
        let rou = get_dcct_root_of_unity(eval.domain.size() as _);
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] initialize_dcct_domain");
        initialize_dcct_domain(eval.domain.log_size(), rou, &DeviceContext::default()).unwrap();
        nvtx::range_pop!();
        let eval_values = unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(&eval.values.data[..]) };

        let mut coeffs = unsafe { DeviceColumn::uninitialized(eval_values.len()) };
        let mut coeffs_data = unsafe { transmute::<&mut DeviceSlice<BaseField>, &mut DeviceSlice<ScalarField>>(&mut coeffs.data[..]) };

        let mut cfg = NTTConfig::default();
        cfg.ordering = Ordering::kMN;
        nvtx::range_push!("[ICICLE] interpolate");
        dcct::interpolate(
            eval_values,
            &cfg,
            coeffs_data,
        )
        .unwrap();
        nvtx::range_pop!();

        CirclePoly::new(coeffs)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        // todo!()
        // unsafe { CpuBackend::eval_at_point(transmute(poly), point) }
        if poly.log_size() == 0 {
            return poly.coeffs.to_cpu()[0].into();
        }
        // TODO: to gpu after correctness fix
        nvtx::range_push!("[ICICLE] create mappings");
        let mut mappings = vec![point.y];
        let mut x = point.x;
        for _ in 1..poly.log_size() {
            mappings.push(x);
            x = CirclePoint::double_x(x);
        }
        mappings.reverse();
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] fold");
        let folded = crate::core::backend::icicle::utils::fold::<BaseField, SecureField>(&poly.coeffs, &mappings);
        nvtx::range_pop!();
        folded
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        assert!(log_size >= poly.log_size());
        let count_zeros_to_extend = 1 << (log_size-1) - poly.coeffs.len() as u32;
        let coeffs = DeviceVec::cuda_malloc_extend_with_zeros(&poly.coeffs.data, count_zeros_to_extend).unwrap();
        CirclePoly::new(DeviceColumn {data: coeffs})
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        if domain.log_size() <= 3 || domain.log_size() == 7 {
            let cpu_poly = CpuCirclePoly::new(poly.coeffs.to_cpu());

            let cpu_circle_eval = CpuBackend::evaluate(
                &cpu_poly,
                domain,
                unsafe { transmute(twiddles) },
            );

            let icicle_eval_values = DeviceColumn::from_cpu(cpu_circle_eval.values.as_slice());

            return IcicleCircleEvaluation::new(cpu_circle_eval.domain, icicle_eval_values);
        }

        let values = poly.extend(domain.log_size()).coeffs;
        nvtx::range_push!("[ICICLE] get_dcct_root_of_unity");
        let rou = get_dcct_root_of_unity(domain.size() as _);
        nvtx::range_pop!();
        nvtx::range_push!("[ICICLE] initialize_dcct_domain");
        initialize_dcct_domain(domain.log_size(), rou, &DeviceContext::default()).unwrap();
        nvtx::range_pop!();

        // let mut evaluations = vec![ScalarField::zero(); values.len()];
        let mut evaluations = DeviceColumn { data: DeviceVec::cuda_malloc(values.len()).unwrap()};

        let mut cfg = NTTConfig::default();
        cfg.ordering = Ordering::kNM;
        nvtx::range_push!("[ICICLE] evaluate");
        dcct::evaluate(
            unsafe { transmute::<&DeviceSlice<M31>, &DeviceSlice<_>>(&values.data[..])},
            &cfg,
            unsafe { transmute::<&mut DeviceSlice<M31>, &mut DeviceSlice<_>>(&mut evaluations.data[..])},
        )
        .unwrap();
        nvtx::range_pop!();

        IcicleCircleEvaluation::<BaseField, BitReversedOrder>::new(domain, evaluations)
    }

    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CirclePoly<Self>> {
        columns
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect()

        // let mut result = Vec::new();
        // let values: Vec<Vec<_>> = columns.into_iter().map(|eval| eval.values).collect();
        // let domain_size = values[0].len();
        // let domain_size_log2 = (domain_size as f64).log2() as u32;
        // let batch_size = values.len();
        // let ctx = DeviceContext::default();
        // let rou = get_dcct_root_of_unity(domain_size as _);
        // initialize_dcct_domain(domain_size_log2, rou, &ctx).unwrap();
        // assuming this is always evenly-sized batch m x n
        //
        // let mut result_tr: DeviceVec<ScalarField> =
        // DeviceVec::cuda_malloc(domain_size * batch_size).unwrap();
        // let mut evaluations_batch = vec![ScalarField::zero(); domain_size * batch_size];
        //
        // let mut res_host = HostSlice::from_mut_slice(&mut evaluations_batch[..]);
        // result_tr.copy_to_host(res_host).unwrap();
        //
        // non-contiguous memory on host
        // let evals: Vec<Vec<ScalarField>> = unsafe { transmute(values) };
        //
        // contiguous memory on device
        // result_tr
        // .copy_from_host_slice_vec_async(&evals, &ctx.stream)
        // .unwrap();
        //
        // ctx.stream.synchronize().unwrap();
        //
        // let mut cfg = NTTConfig::default();
        // cfg.batch_size = batch_size as _;
        // cfg.ordering = Ordering::kNM;
        // evaluate(&result_tr[..], &cfg, res_host).unwrap();
        // for i in 0..batch_size {
        // result.push(CirclePoly::new(unsafe {
        // transmute(res_host.as_slice()[i * domain_size..(i + 1) * domain_size].to_vec())
        // }));
        // }
        //
        // result
    }

    fn evaluate_polynomials(
        polynomials: &ColumnVec<CirclePoly<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>> {
        // TODO: it's variable size batch after all :(
        polynomials
            .iter()
            .map(|poly| {
                poly.evaluate_with_twiddles(
                    CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain(),
                    twiddles,
                )
            })
            .collect_vec()
        // let mut result = Vec::new();
        // let domain =
        // CanonicCoset::new(polynomials[0].log_size() + log_blowup_factor).circle_domain();
        // let rou = get_dcct_root_of_unity(domain.size() as _);
        // let domain_size = 1 << domain.log_size();
        // let batch_size = polynomials.len();
        // let ctx = DeviceContext::default();
        // initialize_dcct_domain(domain.log_size(), rou, &ctx).unwrap();
        // assuming this is always evenly-sized batch m x n
        //
        // let mut result_tr: DeviceVec<ScalarField> =
        // DeviceVec::cuda_malloc(domain_size * batch_size).unwrap();
        // let mut evaluations_batch = vec![ScalarField::zero(); domain_size * batch_size];
        //
        // let mut res_host = HostSlice::from_mut_slice(&mut evaluations_batch[..]);
        // result_tr.copy_to_host(res_host).unwrap();
        //
        // non-contiguous memory on host
        // let vals_extended = polynomials
        // .iter()
        // .map(|poly| poly.extend(domain.log_size()).coeffs)
        // .collect_vec();
        // let evals: Vec<Vec<ScalarField>> = unsafe { transmute(vals_extended) };
        //
        // contiguous memory on device
        // result_tr
        // .copy_from_host_slice_vec_async(&evals, &ctx.stream)
        // .unwrap();
        //
        // ctx.stream.synchronize().unwrap();
        //
        // let mut cfg = NTTConfig::default();
        // cfg.batch_size = batch_size as _;
        // cfg.ordering = Ordering::kNM;
        // evaluate(&result_tr[..], &cfg, res_host).unwrap();
        // for i in 0..batch_size {
        // result.push(IcicleCircleEvaluation::<BaseField, BitReversedOrder>::new(
        // domain,
        // unsafe {
        // transmute(res_host.as_slice()[i * domain_size..(i + 1) * domain_size].to_vec())
        // },
        // ));
        // }
        //
        // result
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        // todo!()
        unsafe { transmute(CpuBackend::precompute_twiddles(coset)) }
    }
}
