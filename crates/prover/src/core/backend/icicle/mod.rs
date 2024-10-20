// IcicleBackend amalgamation
// TODO: move to separate files
use core::fmt::Debug;
use std::ffi::c_void;
use std::mem::{size_of_val, transmute};

use icicle_core::vec_ops::{accumulate_scalars, VecOpsConfig};
use icicle_m31::dcct::{evaluate, get_dcct_root_of_unity, initialize_dcct_domain, interpolate};
use serde::{Deserialize, Serialize};
use twiddles::TwiddleTree;

use super::{
    Backend, BackendForChannel, BaseField, Col, ColumnOps, CpuBackend, PolyOps, QuotientOps,
};
use crate::core::air::accumulation::AccumulationOps;
use crate::core::channel::Channel;
use crate::core::circle::{self, CirclePoint, Coset};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::{Field, FieldOps};
use crate::core::fri::FriOps;
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::lookups::mle::MleOps;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, SecureEvaluation,
};
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::{twiddles, BitReversedOrder};
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::vcs::poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher};
#[derive(Copy, Clone, Debug, Deserialize, Serialize, Default)]
pub struct IcicleBackend;

impl Backend for IcicleBackend {}

// stwo/crates/prover/src/core/backend/cpu/lookups/gkr.rs
impl GkrOps for IcicleBackend {
    fn gen_eq_evals(
        y: &[crate::core::fields::qm31::SecureField],
        v: crate::core::fields::qm31::SecureField,
    ) -> crate::core::lookups::mle::Mle<Self, crate::core::fields::qm31::SecureField> {
        todo!()
    }

    fn next_layer(
        layer: &crate::core::lookups::gkr_prover::Layer<Self>,
    ) -> crate::core::lookups::gkr_prover::Layer<Self> {
        todo!()
    }

    fn sum_as_poly_in_first_variable(
        h: &crate::core::lookups::gkr_prover::GkrMultivariatePolyOracle<'_, Self>,
        claim: crate::core::fields::qm31::SecureField,
    ) -> crate::core::lookups::utils::UnivariatePoly<crate::core::fields::qm31::SecureField> {
        todo!()
    }
}

impl MleOps<BaseField> for IcicleBackend {
    fn fix_first_variable(
        mle: crate::core::lookups::mle::Mle<Self, BaseField>,
        assignment: crate::core::fields::qm31::SecureField,
    ) -> crate::core::lookups::mle::Mle<Self, crate::core::fields::qm31::SecureField>
    where
        Self: MleOps<crate::core::fields::qm31::SecureField>,
    {
        todo!()
    }
}
impl MleOps<SecureField> for IcicleBackend {
    fn fix_first_variable(
        mle: crate::core::lookups::mle::Mle<Self, SecureField>,
        assignment: crate::core::fields::qm31::SecureField,
    ) -> crate::core::lookups::mle::Mle<Self, crate::core::fields::qm31::SecureField>
    where
        Self: MleOps<crate::core::fields::qm31::SecureField>,
    {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/accumulation.rs
impl AccumulationOps for IcicleBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        use icicle_core::vec_ops::{accumulate_scalars, VecOpsConfig};
        use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice};

        let cfg = VecOpsConfig::default();

        unsafe {
            let limbs_count: usize = size_of_val(&column.columns[0]) / 4;
            use std::slice;

            use icicle_core::traits::FieldImpl;
            use icicle_core::vec_ops::VecOps;
            use icicle_cuda_runtime::device::get_device_from_pointer;
            use icicle_cuda_runtime::memory::{DeviceSlice, HostSlice};
            use icicle_m31::field::{QuarticExtensionField, ScalarField};

            let mut a_ptr = column as *mut _ as *mut c_void;
            let mut d_a_slice;
            let n = column.columns[0].len();
            let secure_degree = column.columns.len();

            let column: &mut SecureColumnByCoords<IcicleBackend> = transmute(column);
            let other = transmute(other);

            let is_a_on_host = get_device_from_pointer(a_ptr).unwrap() == 18446744073709551614;
            let mut col_a;
            if is_a_on_host {
                col_a = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
                d_a_slice = &mut col_a[..];
                SecureColumnByCoords::convert_to_icicle(column, d_a_slice);
            } else {
                let mut v_ptr = a_ptr as *mut QuarticExtensionField;
                let rr = unsafe { slice::from_raw_parts_mut(v_ptr, n) };
                d_a_slice = DeviceSlice::from_mut_slice(rr);
            }
            let b_ptr = other as *const _ as *const c_void;
            let mut d_b_slice;
            let mut col_b;
            if get_device_from_pointer(b_ptr).unwrap() == 18446744073709551614 {
                col_b = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
                d_b_slice = &mut col_b[..];
                SecureColumnByCoords::convert_to_icicle(other, d_b_slice);
            } else {
                let mut v_ptr = b_ptr as *mut QuarticExtensionField;
                let rr = unsafe { slice::from_raw_parts_mut(v_ptr, n) };
                d_b_slice = DeviceSlice::from_mut_slice(rr);
            }

            accumulate_scalars(d_a_slice, d_b_slice, &cfg);

            let mut v_ptr = d_a_slice.as_mut_ptr() as *mut _;
            let d_slice = unsafe { slice::from_raw_parts_mut(v_ptr, secure_degree * n) };
            let d_a_slice = DeviceSlice::from_mut_slice(d_slice);
            SecureColumnByCoords::convert_from_icicle(column, d_a_slice);
        }
    }

    // fn confirm(column: &mut SecureColumnByCoords<Self>) {
    //     column.convert_from_icicle(); // TODO: won't be necessary here on each call, only send
    // back                                   // to stwo core
    // }
}

// stwo/crates/prover/src/core/backend/cpu/blake2s.rs
impl MerkleOps<Blake2sMerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash> {
        // todo!()
        <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size, prev_layer, columns,
        )
    }
}

// stwo/crates/prover/src/core/backend/cpu/circle.rs

type IcicleCirclePoly = CirclePoly<IcicleBackend>;
type IcicleCircleEvaluation<F, EvalOrder> = CircleEvaluation<IcicleBackend, F, EvalOrder>;
// type CpuMle<F> = Mle<CpuBackend, F>;

impl PolyOps for IcicleBackend {
    type Twiddles = Vec<BaseField>;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // todo!()
        unsafe { transmute(CpuBackend::new_canonical_ordered(coset, values)) }
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        // todo!()
        if eval.domain.log_size() <= 3 || eval.domain.log_size() == 7 {
            // TODO: as property .is_dcct_available etc...
            return unsafe {
                transmute(CpuBackend::interpolate(
                    transmute(eval),
                    transmute(itwiddles),
                ))
            };
        }

        let values = eval.values;
        let rou = get_dcct_root_of_unity(eval.domain.size() as _);
        println!("ROU {:?}", rou);
        initialize_dcct_domain(eval.domain.log_size(), rou, &DeviceContext::default()).unwrap();
        println!("initialied DCCT succesfully");

        let mut evaluations = vec![ScalarField::zero(); values.len()];
        let values: Vec<ScalarField> = unsafe { transmute(values) };
        let mut cfg = NTTConfig::default();
        cfg.ordering = Ordering::kMN;
        interpolate(
            HostSlice::from_slice(&values),
            &cfg,
            HostSlice::from_mut_slice(&mut evaluations),
        )
        .unwrap();
        let values: Vec<BaseField> = unsafe { transmute(evaluations) };

        CirclePoly::new(values)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        unsafe { CpuBackend::eval_at_point(transmute(poly), point) }
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        // todo!()
        unsafe { transmute(CpuBackend::extend(transmute(poly), log_size)) }
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // todo!()
        if domain.log_size() <= 3 || domain.log_size() == 7 {
            return unsafe {
                transmute(CpuBackend::evaluate(
                    transmute(poly),
                    domain,
                    transmute(twiddles),
                ))
            };
        }

        let values = poly.extend(domain.log_size()).coeffs;
        // assert!(domain.half_coset.is_doubling_of(twiddles.root_coset));

        // assert_eq!(1 << domain.log_size(), values.len() as u32);

        let rou = get_dcct_root_of_unity(domain.size() as _);
        initialize_dcct_domain(domain.log_size(), rou, &DeviceContext::default()).unwrap();

        let mut evaluations = vec![ScalarField::zero(); values.len()];
        let values: Vec<ScalarField> = unsafe { transmute(values) };
        let mut cfg = NTTConfig::default();
        cfg.ordering = Ordering::kNM;
        evaluate(
            HostSlice::from_slice(&values),
            &cfg,
            HostSlice::from_mut_slice(&mut evaluations),
        )
        .unwrap();
        unsafe {
            transmute(IcicleCircleEvaluation::<BaseField, BitReversedOrder>::new(
                domain,
                transmute(evaluations),
            ))
        }
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        // todo!()
        unsafe { transmute(CpuBackend::precompute_twiddles(coset)) }
    }
}

// stwo/crates/prover/src/core/backend/cpu/fri.rs
impl FriOps for IcicleBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        todo!()
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        todo!()
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        todo!()
    }
}

// stwo/crates/prover/src/core/backend/cpu/grind.rs
impl<C: Channel> GrindOps<C> for IcicleBackend {
    fn grind(channel: &C, pow_bits: u32) -> u64 {
        // todo!()
        CpuBackend::grind(channel, pow_bits)
    }
}

// stwo/crates/prover/src/core/backend/cpu/mod.rs
// impl Backend for IcicleBackend {}

impl BackendForChannel<Blake2sMerkleChannel> for IcicleBackend {}
impl BackendForChannel<Poseidon252MerkleChannel> for IcicleBackend {}
impl<T: Debug + Clone + Default> ColumnOps<T> for IcicleBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        // todo!()
        CpuBackend::bit_reverse_column(column)
    }
}
impl<F: Field> FieldOps<F> for IcicleBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        // todo!()
        CpuBackend::batch_inverse(column, dst)
    }
}

// stwo/crates/prover/src/core/backend/cpu/quotients.rs
impl QuotientOps for IcicleBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        todo!()
    }
}

// stwo/crates/prover/src/core/vcs/poseidon252_merkle.rs
impl MerkleOps<Poseidon252MerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Poseidon252MerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Poseidon252MerkleHasher as MerkleHasher>::Hash> {
        // todo!()

        <CpuBackend as MerkleOps<Poseidon252MerkleHasher>>::commit_on_layer(
            log_size, prev_layer, columns,
        )
    }
}

use std::ptr::{self, slice_from_raw_parts, slice_from_raw_parts_mut};

use icicle_core::ntt::{FieldImpl, NTTConfig, Ordering};
use icicle_core::vec_ops::{stwo_convert, transpose_matrix};
use icicle_cuda_runtime::device::get_device_from_pointer;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_cuda_runtime::stream::CudaStream;
use icicle_m31::field::{QuarticExtensionField, ScalarField};

impl SecureColumnByCoords<IcicleBackend> {
    pub fn convert_to_icicle(input: &Self, d_output: &mut DeviceSlice<QuarticExtensionField>) {
        let zero = ScalarField::zero();

        let n = input.columns[0].len();
        let secure_degree = input.columns.len();

        let cfg = VecOpsConfig::default();

        let a: &[u32] = unsafe { transmute(input.columns[0].as_slice()) };
        let b: &[u32] = unsafe { transmute(input.columns[1].as_slice()) };
        let c: &[u32] = unsafe { transmute(input.columns[2].as_slice()) };
        let d: &[u32] = unsafe { transmute(input.columns[3].as_slice()) };

        let a = HostSlice::from_slice(&a);
        let b = HostSlice::from_slice(&b);
        let c = HostSlice::from_slice(&c);
        let d = HostSlice::from_slice(&d);

        stwo_convert(a, b, c, d, d_output);
    }

    pub fn convert_from_icicle(input: &mut Self, d_input: &mut DeviceSlice<ScalarField>) {
        let zero = ScalarField::zero();

        let n = input.columns[0].len();
        let secure_degree = input.columns.len();
        let mut intermediate_host = vec![zero; secure_degree * n];

        let cfg = VecOpsConfig::default();

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

        let mut res_host = HostSlice::from_mut_slice(&mut intermediate_host[..]);
        result_tr.copy_to_host(res_host).unwrap();

        use crate::core::fields::m31::M31;

        let res: Vec<M31> = unsafe { transmute(intermediate_host) };

        // Assign the sub-slices to the column
        for i in 0..secure_degree {
            let start = i * n;
            let end = start + n;

            input.columns[i].truncate(0);
            input.columns[i].extend_from_slice(&res[start..end]);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;
    use num_traits::One;

    use crate::core::backend::cpu::CpuCirclePoly;
    use crate::core::backend::icicle::{IcicleBackend, IcicleCircleEvaluation, IcicleCirclePoly};
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::ExtensionOf;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};

    impl<F: ExtensionOf<BaseField>, EvalOrder> IntoIterator
        for CircleEvaluation<IcicleBackend, F, EvalOrder>
    {
        type Item = F;
        type IntoIter = std::vec::IntoIter<F>;

        /// Creates a consuming iterator over the evaluations.
        ///
        /// Evaluations are returned in the same order as elements of the domain.
        fn into_iter(self) -> Self::IntoIter {
            self.values.into_iter()
        }
    }

    #[test]
    fn test_icicle_eval_at_point_with_4_coeffs() {
        // Represents the polynomial `1 + 2y + 3x + 4xy`.
        // Note coefficients are passed in bit reversed order.
        let poly = IcicleCirclePoly::new([1, 3, 2, 4].map(BaseField::from).to_vec());
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(
            eval,
            poly.coeffs[0] + poly.coeffs[1] * y + poly.coeffs[2] * x + poly.coeffs[3] * x * y
        );
    }

    #[test]
    fn test_icicle_eval_at_point_with_2_coeffs() {
        // Represents the polynomial `1 + 2y`.
        let poly = IcicleCirclePoly::new(vec![BaseField::from(1), BaseField::from(2)]);
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(eval, poly.coeffs[0] + poly.coeffs[1] * y);
    }

    #[test]
    fn test_icicle_eval_at_point_with_1_coeff() {
        // Represents the polynomial `1`.
        let poly = IcicleCirclePoly::new(vec![BaseField::one()]);
        let x = BaseField::from(5).into();
        let y = BaseField::from(8).into();

        let eval = poly.eval_at_point(CirclePoint { x, y });

        assert_eq!(eval, SecureField::one());
    }

    #[test]
    #[ignore = "log=1?"]
    fn test_icicle_evaluate_2_coeffs() {
        let domain = CanonicCoset::new(1).circle_domain();
        let poly = IcicleCirclePoly::new((1..=2).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (p, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(p.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    #[ignore = "log=2?"]

    fn test_icicle_evaluate_4_coeffs() {
        let domain = CanonicCoset::new(2).circle_domain();
        let poly = IcicleCirclePoly::new((1..=4).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (x, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(x.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_icicle_evaluate_16_coeffs() {
        let domain = CanonicCoset::new(4).circle_domain();
        let poly = IcicleCirclePoly::new((1..=16).map(BaseField::from).collect());

        let evaluation = poly.clone().evaluate(domain).bit_reverse();

        for (i, (x, eval)) in zip(domain, evaluation).enumerate() {
            let eval: SecureField = eval.into();
            assert_eq!(eval, poly.eval_at_point(x.into_ef()), "mismatch at i={i}");
        }
    }

    #[test]
    fn test_icicle_interpolate_2_evals() {
        let poly = IcicleCirclePoly::new(vec![BaseField::one(), BaseField::from(2)]);
        let domain = CanonicCoset::new(1).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn test_icicle_interpolate_4_evals() {
        let poly = IcicleCirclePoly::new((1..=4).map(BaseField::from).collect());
        let domain = CanonicCoset::new(2).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn test_icicle_interpolate_8_evals() {
        let poly = IcicleCirclePoly::new((1..=8).map(BaseField::from).collect());
        let domain = CanonicCoset::new(3).circle_domain();
        let evals = poly.clone().evaluate(domain);

        let interpolated_poly = evals.interpolate();

        assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    }

    #[test]
    fn test_icicle_interpolate_and_eval() {
        for log in (4..6).chain(8..25) {
            let domain = CanonicCoset::new(log).circle_domain();
            assert_eq!(domain.log_size(), log);
            let evaluation = IcicleCircleEvaluation::new(
                domain,
                (0..1 << log).map(BaseField::from_u32_unchecked).collect(),
            );
            let poly = evaluation.clone().interpolate();
            let evaluation2 = poly.evaluate(domain);
            assert_eq!(evaluation.values, evaluation2.values);
        }
    }

    // use crate::qm31;
    // use crate::core::backend::icicle::column::BaseColumn;
    // use crate::core::poly::circle::CircleEvaluation;
    // use crate::core::poly::BitReversedOrder;
    // use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    // use crate::core::pcs::quotients::ColumnSampleBatch;
    // use crate::core::backend::CpuBackend;
    // use crate::core::pcs::quotients::QuotientOps;
    //
    // #[test]
    // fn test_icicle_accumulate_quotients() {
    // const LOG_SIZE: u32 = 8;
    // const LOG_BLOWUP_FACTOR: u32 = 1;
    // let small_domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    // let domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
    // let e0: BaseColumn = (0..small_domain.size()).map(BaseField::from).collect();
    // let e1: BaseColumn = (0..small_domain.size())
    // .map(|i| BaseField::from(2 * i))
    // .collect();
    // let polys = vec![
    // CircleEvaluation::<IcicleBackend, BaseField, BitReversedOrder>::new(small_domain, e0)
    // .interpolate(),
    // CircleEvaluation::<IcicleBackend, BaseField, BitReversedOrder>::new(small_domain, e1)
    // .interpolate(),
    // ];
    // let columns = vec![polys[0].evaluate(domain), polys[1].evaluate(domain)];
    // let random_coeff = qm31!(1, 2, 3, 4);
    // let a = polys[0].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
    // let b = polys[1].eval_at_point(SECURE_FIELD_CIRCLE_GEN);
    // let samples = vec![ColumnSampleBatch {
    // point: SECURE_FIELD_CIRCLE_GEN,
    // columns_and_values: vec![(0, a), (1, b)],
    // }];
    // let cpu_columns = columns
    // .iter()
    // .map(|c| CircleEvaluation::new(c.domain, c.values.to_cpu()))
    // .collect_vec();
    // let cpu_result = CpuBackend::accumulate_quotients(
    // domain,
    // &cpu_columns.iter().collect_vec(),
    // random_coeff,
    // &samples,
    // LOG_BLOWUP_FACTOR,
    // )
    // .values
    // .to_vec();
    //
    // let res = IcicleBackend::accumulate_quotients(
    // domain,
    // &columns.iter().collect_vec(),
    // random_coeff,
    // &samples,
    // LOG_BLOWUP_FACTOR,
    // )
    // .values
    // .to_vec();
    //
    // assert_eq!(res, cpu_result);
    // }
}
