// IcicleBackend amalgamation
// TODO: move to separate files
use core::fmt::Debug;
use std::array;
use std::cmp::Reverse;
use std::ffi::c_void;
use std::iter::zip;
use std::mem::{size_of_val, transmute};

use icicle_core::field::Field as IcicleField;
use icicle_core::tree::{merkle_tree_digests_len, TreeBuilderConfig};
use icicle_core::vec_ops::{accumulate_scalars, VecOpsConfig};
use icicle_core::Matrix;
use icicle_hash::blake2s::blake2s_commit_layer;
use icicle_m31::dcct::{evaluate, get_dcct_root_of_unity, initialize_dcct_domain, interpolate};
use icicle_m31::field::ScalarCfg;
use icicle_m31::fri::{self, fold_circle_into_line, fold_circle_into_line_new, FriConfig};
use icicle_m31::quotient;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use twiddles::TwiddleTree;

use super::{
    Backend, BackendForChannel, BaseField, Col, ColumnOps, CpuBackend, PolyOps, QuotientOps,
};
use crate::core::air::accumulation::AccumulationOps;
use crate::core::channel::Channel;
use crate::core::circle::{self, CirclePoint, Coset};
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::{Field, FieldExpOps, FieldOps};
use crate::core::fri::{FriOps, CIRCLE_TO_LINE_FOLD_STEP, FOLD_STEP};
use crate::core::lookups::gkr_prover::GkrOps;
use crate::core::lookups::mle::MleOps;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::core::poly::circle::{
    CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, SecureEvaluation,
};
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::{twiddles, BitReversedOrder};
use crate::core::proof_of_work::GrindOps;
use crate::core::utils::bit_reverse_index;
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::vcs::poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher};
use crate::core::ColumnVec;
#[derive(Copy, Clone, Debug, Deserialize, Serialize, Default)]
pub struct IcicleBackend;

pub mod utils;

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
                nvtx::range_push!("[ICICLE] convert + move");
                col_a = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
                d_a_slice = &mut col_a[..];
                SecureColumnByCoords::convert_to_icicle(column, d_a_slice);
                nvtx::range_pop!();
            } else {
                let mut v_ptr = a_ptr as *mut QuarticExtensionField;
                let rr = unsafe { slice::from_raw_parts_mut(v_ptr, n) };
                d_a_slice = DeviceSlice::from_mut_slice(rr);
            }
            let b_ptr = other as *const _ as *const c_void;
            let mut d_b_slice;
            let mut col_b;
            if get_device_from_pointer(b_ptr).unwrap() == 18446744073709551614 {
                nvtx::range_push!("[ICICLE] convert + move");
                col_b = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
                d_b_slice = &mut col_b[..];
                SecureColumnByCoords::convert_to_icicle(other, d_b_slice);
                nvtx::range_pop!();
            } else {
                let mut v_ptr = b_ptr as *mut QuarticExtensionField;
                let rr = unsafe { slice::from_raw_parts_mut(v_ptr, n) };
                d_b_slice = DeviceSlice::from_mut_slice(rr);
            }
            nvtx::range_push!("[ICICLE] accum scalars");
            accumulate_scalars(d_a_slice, d_b_slice, &cfg);
            nvtx::range_pop!();

            nvtx::range_push!("[ICICLE] convert + move to SecureColumnByCoords");
            let mut v_ptr = d_a_slice.as_mut_ptr() as *mut _;
            let d_slice = unsafe { slice::from_raw_parts_mut(v_ptr, secure_degree * n) };
            let d_a_slice = DeviceSlice::from_mut_slice(d_slice);
            SecureColumnByCoords::convert_from_icicle(column, d_a_slice);
            nvtx::range_pop!();
        }
    }
    
    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        //todo!()
        CpuBackend::generate_secure_powers(felt, n_powers)
    }
}

// stwo/crates/prover/src/core/backend/cpu/blake2s.rs
impl MerkleOps<Blake2sMerkleHasher> for IcicleBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, <Blake2sMerkleHasher as MerkleHasher>::Hash> {
        let prev_layer = match prev_layer {
            Some(layer) => unsafe{ transmute(layer) },
            None => &Vec::<u8>::with_capacity(0),
        };

        let mut columns_as_matrices = vec![];
        for &col in columns {
            columns_as_matrices.push(Matrix::from_slice(col, 4, col.len()));
        }

        let digest_bytes = (1 << log_size) * 32;
        let mut d_digests_slice = DeviceVec::cuda_malloc(digest_bytes).unwrap();

        blake2s_commit_layer(
            HostSlice::from_slice(prev_layer),
            false,
            &columns_as_matrices,
            false,
            columns.len() as u32,
            1 << log_size,
            &mut d_digests_slice[..],
        ).unwrap();

        let mut digests = vec![0u8; digest_bytes];
        let mut digests_slice = HostSlice::from_mut_slice(&mut digests);

        d_digests_slice
            .copy_to_host(&mut digests_slice)
            .unwrap();

        unsafe { std::mem::transmute(digests) }
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
        nvtx::range_push!("[ICICLE] get_dcct_root_of_unity");
        let rou = get_dcct_root_of_unity(eval.domain.size() as _);
        nvtx::range_pop!();
        nvtx::range_push!("[ICICLE] initialize_dcct_domain");
        initialize_dcct_domain(eval.domain.log_size(), rou, &DeviceContext::default()).unwrap();
        nvtx::range_pop!();

        let mut evaluations = vec![ScalarField::zero(); values.len()];
        let values: Vec<ScalarField> = unsafe { transmute(values) };
        let mut cfg = NTTConfig::default();
        cfg.ordering = Ordering::kMN;
        nvtx::range_push!("[ICICLE] interpolate");
        interpolate(
            HostSlice::from_slice(&values),
            &cfg,
            HostSlice::from_mut_slice(&mut evaluations),
        )
        .unwrap();
        nvtx::range_pop!();
        let values: Vec<BaseField> = unsafe { transmute(evaluations) };

        CirclePoly::new(values)
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        // todo!()
        // unsafe { CpuBackend::eval_at_point(transmute(poly), point) }
        if poly.log_size() == 0 {
            return poly.coeffs[0].into();
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
        let folded = crate::core::backend::icicle::utils::fold(&poly.coeffs, &mappings);
        nvtx::range_pop!();
        folded
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
        nvtx::range_push!("[ICICLE] get_dcct_root_of_unity");
        let rou = get_dcct_root_of_unity(domain.size() as _);
        nvtx::range_pop!();
        nvtx::range_push!("[ICICLE] initialize_dcct_domain");
        initialize_dcct_domain(domain.log_size(), rou, &DeviceContext::default()).unwrap();
        nvtx::range_pop!();

        let mut evaluations = vec![ScalarField::zero(); values.len()];
        let values: Vec<ScalarField> = unsafe { transmute(values) };
        let mut cfg = NTTConfig::default();
        cfg.ordering = Ordering::kNM;
        nvtx::range_push!("[ICICLE] evaluate");
        evaluate(
            HostSlice::from_slice(&values),
            &cfg,
            HostSlice::from_mut_slice(&mut evaluations),
        )
        .unwrap();
        nvtx::range_pop!();
        unsafe {
            transmute(IcicleCircleEvaluation::<BaseField, BitReversedOrder>::new(
                domain,
                transmute(evaluations),
            ))
        }
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

// stwo/crates/prover/src/core/backend/cpu/fri.rs
impl FriOps for IcicleBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        use crate::core::backend::icicle::IcicleBackend;
        let length = eval.values.len(); // TODO: same as n

        let n = eval.len();
        assert!(n >= 2, "Evaluation too small");

        let domain = eval.domain();

        let dom_vals_len = length / 2;

        let line_domain_log_size = domain.log_size();
        
        nvtx::range_push!("[ICICLE] domain evals convert + move");
        let mut d_evals_icicle = DeviceVec::<QuarticExtensionField>::cuda_malloc(length).unwrap();
        SecureColumnByCoords::<IcicleBackend>::convert_to_icicle(
            unsafe { transmute(&eval.values) },
            &mut d_evals_icicle,
        );
        nvtx::range_pop!();
        let mut d_folded_eval =
            DeviceVec::<QuarticExtensionField>::cuda_malloc(dom_vals_len).unwrap();

        let cfg = FriConfig::default();
        let icicle_alpha = unsafe { transmute(alpha) };
        nvtx::range_push!("[ICICLE] fold_line");
        let _ = fri::fold_line_new(
            &d_evals_icicle[..],
            // &d_domain_icicle[..],
            domain.coset().initial_index.0 as _,
            domain.coset().log_size,
            &mut d_folded_eval[..],
            icicle_alpha,
            &cfg,
        )
        .unwrap();
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] convert to SecureColumnByCoords");
        let mut folded_values = unsafe { SecureColumnByCoords::uninitialized(dom_vals_len) };
        SecureColumnByCoords::<IcicleBackend>::convert_from_icicle_q31(
            &mut folded_values,
            &mut d_folded_eval[..],
        );

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

        let dom_vals_len = length / 2;
        let domain_log_size = domain.log_size();

        let mut d_evals_icicle = DeviceVec::<QuarticExtensionField>::cuda_malloc(length).unwrap();
        SecureColumnByCoords::convert_to_icicle(&src.values, &mut d_evals_icicle);

        nvtx::range_push!("[ICICLE] d_folded_evals");
        let mut d_folded_eval =
            DeviceVec::<QuarticExtensionField>::cuda_malloc(dom_vals_len).unwrap();
        SecureColumnByCoords::convert_to_icicle(&dst.values, &mut d_folded_eval);
        nvtx::range_pop!();

        let cfg = FriConfig::default();
        let icicle_alpha = unsafe { transmute(alpha) };

        nvtx::range_push!("[ICICLE] fold circle");
        let _ = fold_circle_into_line_new(
            &d_evals_icicle[..],
            domain.half_coset.initial_index.0 as _,
            domain.half_coset.log_size,
            &mut d_folded_eval[..],
            icicle_alpha,
            &cfg,
        )
        .unwrap();
        nvtx::range_pop!();

        nvtx::range_push!("[ICICLE] convert to SecureColumnByCoords");
        SecureColumnByCoords::convert_from_icicle_q31(&mut dst.values, &mut d_folded_eval[..]);
        nvtx::range_pop!();
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // todo!()
        unsafe { transmute(CpuBackend::decompose(unsafe { transmute(eval) })) }
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
    use std::collections::BTreeMap;
    use std::iter::zip;

    use itertools::Itertools;
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::cpu::CpuCirclePoly;
    use crate::core::backend::icicle::{IcicleBackend, IcicleCircleEvaluation, IcicleCirclePoly};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;
    use crate::core::fields::ExtensionOf;
    use crate::core::fri::FriOps;
    use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use crate::core::poly::circle::{
        CanonicCoset, CircleEvaluation, CirclePoly, PolyOps, SecureEvaluation,
    };
    use crate::core::poly::line::{LineDomain, LineEvaluation};
    use crate::core::poly::twiddles::TwiddleTree;
    use crate::core::vcs::prover::MerkleProver;
    use crate::core::vcs::verifier::MerkleVerifier;
    use crate::{m31, qm31};

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

    use std::ptr::null_mut;

    use num_traits::Zero;
    #[cfg(feature = "parallel")]
    use rayon::iter::IntoParallelIterator;

    use crate::core::backend::{ColumnOps, CpuBackend};
    use crate::core::circle::{CirclePointIndex, Coset};
    use crate::core::fields::Field;
    use crate::core::fri::{
        fold_circle_into_line, fold_line, CirclePolyDegreeBound, FriConfig,
        CIRCLE_TO_LINE_FOLD_STEP,
    };
    use crate::core::poly::line::LinePoly;
    use crate::core::poly::{BitReversedOrder, NaturalOrder};
    use crate::core::queries::Queries;
    use crate::core::test_utils::test_channel;
    use crate::core::utils::bit_reverse_index;
    use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};

    /// Default blowup factor used for tests.
    const LOG_BLOWUP_FACTOR: u32 = 2;

    #[test]
    fn tetst_icicle_blake2s_merkle_tree() {
        const N_COLS: usize = 10;
        const N_QUERIES: usize = 3;
        let log_size_range = 3..5;

        let mut rng = SmallRng::seed_from_u64(0);
        let log_sizes = (0..N_COLS)
            .map(|_| rng.gen_range(log_size_range.clone()))
            .collect_vec();
        let cols = log_sizes
            .iter()
            .map(|&log_size| {
                (0..(1 << log_size))
                    .map(|_| BaseField::from(rng.gen_range(0..(1 << 30))))
                    .collect_vec()
            })
            .collect_vec();

        let merkle =
            MerkleProver::<CpuBackend, Blake2sMerkleHasher>::commit(cols.iter().collect_vec());

        let icicle_merkle =
            MerkleProver::<IcicleBackend, Blake2sMerkleHasher>::commit(cols.iter().collect_vec());

        for (layer, icicle_layer) in merkle.layers.iter().zip(icicle_merkle.layers.iter()) {
            for (h1, h2) in layer.iter().zip(icicle_layer.iter()) {
                assert_eq!(h1, h2);
            }
        }

        let mut queries = BTreeMap::<u32, Vec<usize>>::new();
        for log_size in log_size_range.rev() {
            let layer_queries = (0..N_QUERIES)
                .map(|_| rng.gen_range(0..(1 << log_size)))
                .sorted()
                .dedup()
                .collect_vec();
            queries.insert(log_size, layer_queries);
        }

        let (values, decommitment) = merkle.decommit(&queries, cols.iter().collect_vec());

        let verifier = MerkleVerifier::new(merkle.root(), log_sizes);

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_icicle_fold_line_works() {
        const DEGREE: usize = 8;
        // Coefficients are bit-reversed.
        let even_coeffs: [SecureField; DEGREE / 2] = [1, 2, 1, 3]
            .map(BaseField::from_u32_unchecked)
            .map(SecureField::from);
        let odd_coeffs: [SecureField; DEGREE / 2] = [3, 5, 4, 1]
            .map(BaseField::from_u32_unchecked)
            .map(SecureField::from);
        let poly = LinePoly::new([even_coeffs, odd_coeffs].concat());
        let even_poly = LinePoly::new(even_coeffs.to_vec());
        let odd_poly = LinePoly::new(odd_coeffs.to_vec());
        let alpha = BaseField::from_u32_unchecked(19283).into();
        let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2()));
        let drp_domain = domain.double();
        let mut values = domain
            .iter()
            .map(|p| poly.eval_at_point(p.into()))
            .collect();
        IcicleBackend::bit_reverse_column(&mut values);
        let evals = LineEvaluation::new(domain, values.into_iter().collect());

        let dummy_domain = CanonicCoset::new(2);

        let dummy_twiddles = IcicleBackend::precompute_twiddles(dummy_domain.half_coset());
        let drp_evals = IcicleBackend::fold_line(&evals, alpha, &dummy_twiddles);
        let mut drp_evals = drp_evals.values.into_iter().collect_vec();
        IcicleBackend::bit_reverse_column(&mut drp_evals);

        assert_eq!(drp_evals.len(), DEGREE / 2);
        for (i, (&drp_eval, x)) in zip(&drp_evals, drp_domain).enumerate() {
            let f_e: SecureField = even_poly.eval_at_point(x.into());
            let f_o: SecureField = odd_poly.eval_at_point(x.into());
            assert_eq!(drp_eval, (f_e + alpha * f_o).double(), "mismatch at {i}");
        }
    }

    #[test]
    fn test_icicle_fold_line() {
        let mut is_correct = true;
        for log_size in 1..24 {
            let mut rng = SmallRng::seed_from_u64(0);
            let values = (0..1 << log_size).map(|_| rng.gen()).collect_vec();
            let alpha = qm31!(1, 3, 5, 7);
            let domain = LineDomain::new(CanonicCoset::new(log_size + 1).half_coset());

            let secure_column: SecureColumnByCoords<_> = values.iter().copied().collect();
            let line_evaluation = LineEvaluation::new(domain, secure_column);
            let cpu_fold = CpuBackend::fold_line(
                &line_evaluation,
                alpha,
                &CpuBackend::precompute_twiddles(domain.coset()),
            );

            let line_evaluation = LineEvaluation::new(domain, values.into_iter().collect());
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
            let mut icicle_fold = LineEvaluation::new(
                line_domain,
                SecureColumnByCoords::zeros(1 << (log_size - 1)),
            );
            IcicleBackend::fold_circle_into_line(
                &mut icicle_fold,
                &SecureEvaluation::new(circle_domain, values.iter().copied().collect()),
                alpha,
                &IcicleBackend::precompute_twiddles(line_domain.coset()),
            );

            let mut simd_fold = LineEvaluation::new(
                line_domain,
                SecureColumnByCoords::zeros(1 << (log_size - 1)),
            );
            SimdBackend::fold_circle_into_line(
                &mut simd_fold,
                &SecureEvaluation::new(circle_domain, values.iter().copied().collect()),
                alpha,
                &SimdBackend::precompute_twiddles(line_domain.coset()),
            );

            if icicle_fold.values.to_vec() != simd_fold.values.to_vec() {
                println!("failed to fold log2: {}", log_size);
                is_correct = false;
            }
        }
        assert!(is_correct);
    }
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
        let polynomial_icicle =
            IcicleCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
        let eval_icicle = polynomial_icicle.evaluate(eval_domain);
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
