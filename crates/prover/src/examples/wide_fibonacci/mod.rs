use itertools::Itertools;

use crate::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};
use crate::core::backend::simd::m31::PackedBaseField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

pub type WideFibonacciComponent<const N: usize> = FrameworkComponent<WideFibonacciEval<N>>;

pub struct FibInput {
    a: PackedBaseField,
    b: PackedBaseField,
}

/// A component that enforces the Fibonacci sequence.
/// Each row contains a seperate Fibonacci sequence of length `N`.
#[derive(Clone)]
pub struct WideFibonacciEval<const N: usize> {
    pub log_n_rows: u32,
}
impl<const N: usize> FrameworkEval for WideFibonacciEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square()));
            a = b;
            b = c;
        }
        eval
    }
}

pub fn generate_trace<const N: usize>(
    log_size: u32,
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let mut trace = (0..N)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.data[vec_index] = b;
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};

    use super::WideFibonacciEval;
    use crate::constraint_framework::{
        assert_constraints, AssertEvaluator, FrameworkEval, TraceLocationAllocator,
    };
    use crate::core::air::Component;
    use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::Column;
    use crate::core::channel::Blake2sChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::core::channel::Poseidon252Channel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
    use crate::core::poly::BitReversedOrder;
    use crate::core::prover::{prove, verify};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
    use crate::core::ColumnVec;
    use crate::examples::wide_fibonacci::{generate_trace, FibInput, WideFibonacciComponent};

    const FIB_SEQUENCE_LENGTH: usize = 100;

    fn generate_test_trace(
        log_n_instances: u32,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        if log_n_instances < LOG_N_LANES {
            let n_instances = 1 << log_n_instances;
            let inputs = vec![FibInput {
                a: PackedBaseField::from_array(std::array::from_fn(|j| {
                    if j < n_instances {
                        BaseField::one()
                    } else {
                        BaseField::zero()
                    }
                })),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    if j < n_instances {
                        BaseField::from_u32_unchecked((j) as u32)
                    } else {
                        BaseField::zero()
                    }
                })),
            }];
            return generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs);
        }
        let inputs = (0..(1 << (log_n_instances - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec();
        generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs)
    }

    fn fibonacci_constraint_evaluator<const N: u32>(eval: AssertEvaluator<'_>) {
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { log_n_rows: N }.evaluate(eval);
    }

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_INSTANCES: u32 = 6;
        let traces = TreeVec::new(vec![vec![], generate_test_trace(LOG_N_INSTANCES)]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            (SecureField::zero(), None),
        );
    }

    #[test]
    #[should_panic]
    fn test_wide_fibonacci_constraints_fails() {
        const LOG_N_INSTANCES: u32 = 6;

        let mut trace = generate_test_trace(LOG_N_INSTANCES);
        // Modify the trace such that a constraint fail.
        trace[17].values.set(2, BaseField::one());
        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            (SecureField::zero(), None),
        );
    }

    #[test_log::test]
    fn test_wide_fib_prove_with_blake_simd() {
        use crate::examples::utils::get_env_var;

        let min_log = get_env_var("MIN_FIB_LOG", 2u32);
        let max_log = get_env_var("MAX_FIB_LOG", 25u32);

        nvtx::name_thread!("stark_prover");

        for log_n_instances in min_log..=max_log {
            let config = PcsConfig::default();
            // Precompute twiddles.
            nvtx::range_push!("Precompute twiddles");
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );
            nvtx::range_pop!();

            // Setup protocol.
            nvtx::range_push!("Create CommitmentSchemeProver");
            let prover_channel = &mut Blake2sChannel::default();
            let mut commitment_scheme =
                CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);
            nvtx::range_pop!();

            // Preprocessed trace
            nvtx::range_push!("Tree builder");
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals([]);
            tree_builder.commit(prover_channel);
            nvtx::range_pop!();

            // Trace.
            nvtx::range_push!("Generate trace");
            let trace = generate_test_trace(log_n_instances);
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);
            nvtx::range_pop!();

            // Prove constraints.
            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                (SecureField::zero(), None),
            );

            let start = std::time::Instant::now();
            let proof = prove::<SimdBackend, Blake2sMerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();
            println!(
                "SIMD proving for 2^{:?} took {:?} ms",
                log_n_instances,
                start.elapsed().as_millis()
            );

            // Verify.
            let verifier_channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test]
    #[cfg(feature = "icicle")]
    fn test_wide_fib_prove_with_blake_icicle() {
        use std::mem::transmute;

        use icicle_cuda_runtime::memory::HostSlice;

        use crate::constraint_framework::PREPROCESSED_TRACE_IDX;
        use crate::core::backend::icicle::column::DeviceColumn;
        use crate::core::backend::icicle::IcicleBackend;
        // use crate::core::backend::CpuBackend;
        use crate::core::fields::m31::M31;
        use crate::examples::utils::get_env_var;
        type TheBackend = IcicleBackend;
        // type TheBackend = CpuBackend;

        let min_log = get_env_var("MIN_FIB_LOG", 5u32);
        let max_log = get_env_var("MAX_FIB_LOG", 23u32);

        nvtx::name_thread!("stark_prover");

        for log_n_instances in min_log..=max_log {
            let config = PcsConfig::default();
            // Precompute twiddles.
            nvtx::range_push!("Precompute twiddles");
            let twiddles = TheBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );
            nvtx::range_pop!();

            // Setup protocol.
            nvtx::range_push!("Create CommitmentSchemeProver");
            let prover_channel = &mut Blake2sChannel::default();
            let mut commitment_scheme =
                CommitmentSchemeProver::<TheBackend, Blake2sMerkleChannel>::new(config, &twiddles);
            nvtx::range_pop!();

            // Preprocessed trace
            nvtx::range_push!("Tree builder");
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals([]);
            tree_builder.commit(prover_channel);
            nvtx::range_pop!();
            use icicle_cuda_runtime::memory::DeviceVec;
            // Trace.
            nvtx::range_push!("Generate trace");
            type IcicleCircleEvaluation = CircleEvaluation<TheBackend, M31, BitReversedOrder>;
            let trace: Vec<CircleEvaluation<TheBackend, M31, BitReversedOrder>> =
                generate_test_trace(log_n_instances)
                    .iter()
                    .map(|c| {
                        let mut values = DeviceVec::cuda_malloc(c.values.len()).unwrap();
                        values
                            .copy_from_host(HostSlice::from_slice(&c.values.to_cpu()))
                            .unwrap();
                        IcicleCircleEvaluation::new(c.domain, DeviceColumn { data: values })
                    })
                    .collect_vec();

            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);
            nvtx::range_pop!();

            // Prove constraints.
            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                (SecureField::zero(), None),
            );

            icicle_m31::fri::precompute_fri_twiddles(log_n_instances).unwrap();

            let trace_wip = commitment_scheme.trace();

            // nvtx::range_push!("component_evals");
            let mut component_evals = trace_wip.evals.sub_tree(&component.trace_locations());
            component_evals[PREPROCESSED_TRACE_IDX] = component
                .preproccessed_column_indices()
                .iter()
                .map(|idx| &trace_wip.evals[PREPROCESSED_TRACE_IDX][*idx])
                .collect();
            // nvtx::range_pop!();
            let b: Vec<BaseField> = component_evals
                .to_vec()
                .iter()
                .map(|c| c.iter().map(|v| v.values.to_cpu()))
                .flatten()
                .flatten()
                .collect::<Vec<_>>();

            let mut h = DeviceVec::cuda_malloc(b.len()).unwrap();

            h.copy_from_host(HostSlice::from_slice(unsafe {
                transmute(b.as_slice())
            })).unwrap();

            icicle_m31::fri::preload_trace(&h[..]).unwrap();

            icicle_m31::fri::precompute_fri_twiddles(log_n_instances).unwrap();

            let start = std::time::Instant::now();
            let proof = prove::<TheBackend, Blake2sMerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();
            println!(
                "proving for 2^{:?} took {:?} ms",
                log_n_instances,
                start.elapsed().as_millis()
            );

            // Verify.
            let verifier_channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap_or_else(
                |err| {
                    println!("verify failed for {} with: {}", log_n_instances, err);
                },
            );
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wide_fib_prove_with_poseidon() {
        const LOG_N_INSTANCES: u32 = 6;
        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Poseidon252Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

        // TODO(ilya): remove the following once preproccessed columns are not mandatory.
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace = generate_test_trace(LOG_N_INSTANCES);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Prove constraints.
        let component = WideFibonacciComponent::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                log_n_rows: LOG_N_INSTANCES,
            },
            (SecureField::zero(), None),
        );
        let proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
            &[&component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Poseidon252Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }
}
