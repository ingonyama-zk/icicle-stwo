pub mod accumulation;
pub mod blake2s;
pub mod circle;
pub mod column;
pub mod fri;
pub mod grind;
pub mod lookups;
pub mod poseidon252;
pub mod quotient;
pub mod utils;

use serde::{Deserialize, Serialize};

use super::{Backend, BackendForChannel};

#[derive(Copy, Clone, Debug, Deserialize, Serialize, Default)]
pub struct IcicleBackend;

impl Backend for IcicleBackend {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::iter::zip;

    use itertools::Itertools;
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::backend::cpu::CpuCirclePoly;
    use crate::core::backend::icicle::circle::{IcicleCircleEvaluation, IcicleCirclePoly};
    use crate::core::backend::icicle::IcicleBackend;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;
    use crate::core::fields::{ExtensionOf, FieldOps};
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
        for CircleEvaluation<IcicleBackend, F, EvalOrder> where IcicleBackend: FieldOps<F>
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

    // #[test]
    // fn test_icicle_eval_at_point_with_4_coeffs() {
    //     // Represents the polynomial `1 + 2y + 3x + 4xy`.
    //     // Note coefficients are passed in bit reversed order.
    //     let poly = IcicleCirclePoly::new([1, 3, 2, 4].map(BaseField::from).to_vec());
    //     let x = BaseField::from(5).into();
    //     let y = BaseField::from(8).into();

    //     let eval = poly.eval_at_point(CirclePoint { x, y });

    //     assert_eq!(
    //         eval,
    //         poly.coeffs[0] + poly.coeffs[1] * y + poly.coeffs[2] * x + poly.coeffs[3] * x * y
    //     );
    // }

    // #[test]
    // fn test_icicle_eval_at_point_with_2_coeffs() {
    //     // Represents the polynomial `1 + 2y`.
    //     let poly = IcicleCirclePoly::new(vec![BaseField::from(1), BaseField::from(2)]);
    //     let x = BaseField::from(5).into();
    //     let y = BaseField::from(8).into();

    //     let eval = poly.eval_at_point(CirclePoint { x, y });

    //     assert_eq!(eval, poly.coeffs[0] + poly.coeffs[1] * y);
    // }

    // #[test]
    // fn test_icicle_eval_at_point_with_1_coeff() {
    //     // Represents the polynomial `1`.
    //     let poly = IcicleCirclePoly::new(vec![BaseField::one()]);
    //     let x = BaseField::from(5).into();
    //     let y = BaseField::from(8).into();

    //     let eval = poly.eval_at_point(CirclePoint { x, y });

    //     assert_eq!(eval, SecureField::one());
    // }

    // #[test]
    // fn test_icicle_evaluate_2_coeffs() {
    //     let domain = CanonicCoset::new(1).circle_domain();
    //     let poly = IcicleCirclePoly::new((1..=2).map(BaseField::from).collect());

    //     let evaluation = poly.clone().evaluate(domain).bit_reverse();

    //     for (i, (p, eval)) in zip(domain, evaluation).enumerate() {
    //         let eval: SecureField = eval.into();
    //         assert_eq!(eval, poly.eval_at_point(p.into_ef()), "mismatch at i={i}");
    //     }
    // }

    // #[test]
    // fn test_icicle_evaluate_4_coeffs() {
    //     let domain = CanonicCoset::new(2).circle_domain();
    //     let poly = IcicleCirclePoly::new((1..=4).map(BaseField::from).collect());

    //     let evaluation = poly.clone().evaluate(domain).bit_reverse();

    //     for (i, (x, eval)) in zip(domain, evaluation).enumerate() {
    //         let eval: SecureField = eval.into();
    //         assert_eq!(eval, poly.eval_at_point(x.into_ef()), "mismatch at i={i}");
    //     }
    // }

    // #[test]
    // fn test_icicle_evaluate_16_coeffs() {
    //     let domain = CanonicCoset::new(4).circle_domain();
    //     let poly = IcicleCirclePoly::new((1..=16).map(BaseField::from).collect());

    //     let evaluation = poly.clone().evaluate(domain).bit_reverse();

    //     for (i, (x, eval)) in zip(domain, evaluation).enumerate() {
    //         let eval: SecureField = eval.into();
    //         assert_eq!(eval, poly.eval_at_point(x.into_ef()), "mismatch at i={i}");
    //     }
    // }

    // #[test]
    // fn test_icicle_interpolate_2_evals() {
    //     let poly = IcicleCirclePoly::new(vec![BaseField::one(), BaseField::from(2)]);
    //     let domain = CanonicCoset::new(1).circle_domain();
    //     let evals = poly.clone().evaluate(domain);

    //     let interpolated_poly = evals.interpolate();

    //     assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    // }

    // #[test]
    // fn test_icicle_interpolate_4_evals() {
    //     let poly = IcicleCirclePoly::new((1..=4).map(BaseField::from).collect());
    //     let domain = CanonicCoset::new(2).circle_domain();
    //     let evals = poly.clone().evaluate(domain);

    //     let interpolated_poly = evals.interpolate();

    //     assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    // }

    // #[test]
    // fn test_icicle_interpolate_8_evals() {
    //     let poly = IcicleCirclePoly::new((1..=8).map(BaseField::from).collect());
    //     let domain = CanonicCoset::new(3).circle_domain();
    //     let evals = poly.clone().evaluate(domain);

    //     let interpolated_poly = evals.interpolate();

    //     assert_eq!(interpolated_poly.coeffs, poly.coeffs);
    // }

    // #[test]
    // fn test_icicle_interpolate_and_eval() {
    //     for log in (4..6).chain(8..25) {
    //         let domain = CanonicCoset::new(log).circle_domain();
    //         assert_eq!(domain.log_size(), log);
    //         let evaluation = IcicleCircleEvaluation::new(
    //             domain,
    //             (0..1 << log).map(BaseField::from_u32_unchecked).collect(),
    //         );
    //         let poly = evaluation.clone().interpolate();
    //         let evaluation2 = poly.evaluate(domain);
    //         assert_eq!(evaluation.values, evaluation2.values);
    //     }
    // }

    // use std::ptr::null_mut;

    // use num_traits::Zero;
    // #[cfg(feature = "parallel")]
    // use rayon::iter::IntoParallelIterator;

    // use crate::core::backend::{ColumnOps, CpuBackend};
    // use crate::core::circle::{CirclePointIndex, Coset};
    // use crate::core::fields::Field;
    // use crate::core::fri::{
    //     fold_circle_into_line, fold_line, CirclePolyDegreeBound, FriConfig,
    //     CIRCLE_TO_LINE_FOLD_STEP,
    // };
    // use crate::core::poly::line::LinePoly;
    // use crate::core::poly::{BitReversedOrder, NaturalOrder};
    // use crate::core::queries::Queries;
    // use crate::core::test_utils::test_channel;
    // use crate::core::utils::bit_reverse_index;
    // use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};

    // /// Default blowup factor used for tests.
    // const LOG_BLOWUP_FACTOR: u32 = 2;

    // #[test]
    // fn tetst_icicle_blake2s_merkle_tree() {
    //     const N_COLS: usize = 10;
    //     const N_QUERIES: usize = 3;
    //     let log_size_range = 3..5;

    //     let mut rng = SmallRng::seed_from_u64(0);
    //     let log_sizes = (0..N_COLS)
    //         .map(|_| rng.gen_range(log_size_range.clone()))
    //         .collect_vec();
    //     let cols = log_sizes
    //         .iter()
    //         .map(|&log_size| {
    //             (0..(1 << log_size))
    //                 .map(|_| BaseField::from(rng.gen_range(0..(1 << 30))))
    //                 .collect_vec()
    //         })
    //         .collect_vec();

    //     let merkle =
    //         MerkleProver::<CpuBackend, Blake2sMerkleHasher>::commit(cols.iter().collect_vec());

    //     let icicle_merkle =
    //         MerkleProver::<IcicleBackend, Blake2sMerkleHasher>::commit(cols.iter().collect_vec());

    //     for (layer, icicle_layer) in merkle.layers.iter().zip(icicle_merkle.layers.iter()) {
    //         for (h1, h2) in layer.iter().zip(icicle_layer.iter()) {
    //             assert_eq!(h1, h2);
    //         }
    //     }

    //     let mut queries = BTreeMap::<u32, Vec<usize>>::new();
    //     for log_size in log_size_range.rev() {
    //         let layer_queries = (0..N_QUERIES)
    //             .map(|_| rng.gen_range(0..(1 << log_size)))
    //             .sorted()
    //             .dedup()
    //             .collect_vec();
    //         queries.insert(log_size, layer_queries);
    //     }

    //     let (values, decommitment) = merkle.decommit(&queries, cols.iter().collect_vec());

    //     let verifier = MerkleVerifier {
    //         root: merkle.root(),
    //         column_log_sizes: log_sizes,
    //     };

    //     verifier.verify(&queries, values, decommitment).unwrap();
    // }

    // #[test]
    // fn test_icicle_fold_line_works() {
    //     const DEGREE: usize = 8;
    //     // Coefficients are bit-reversed.
    //     let even_coeffs: [SecureField; DEGREE / 2] = [1, 2, 1, 3]
    //         .map(BaseField::from_u32_unchecked)
    //         .map(SecureField::from);
    //     let odd_coeffs: [SecureField; DEGREE / 2] = [3, 5, 4, 1]
    //         .map(BaseField::from_u32_unchecked)
    //         .map(SecureField::from);
    //     let poly = LinePoly::new([even_coeffs, odd_coeffs].concat());
    //     let even_poly = LinePoly::new(even_coeffs.to_vec());
    //     let odd_poly = LinePoly::new(odd_coeffs.to_vec());
    //     let alpha = BaseField::from_u32_unchecked(19283).into();
    //     let domain = LineDomain::new(Coset::half_odds(DEGREE.ilog2()));
    //     let drp_domain = domain.double();
    //     let mut values = domain
    //         .iter()
    //         .map(|p| poly.eval_at_point(p.into()))
    //         .collect();
    //     IcicleBackend::bit_reverse_column(&mut values);
    //     let evals = LineEvaluation::new(domain, values.into_iter().collect());

    //     let dummy_domain = CanonicCoset::new(2);

    //     let dummy_twiddles = IcicleBackend::precompute_twiddles(dummy_domain.half_coset());
    //     let drp_evals = IcicleBackend::fold_line(&evals, alpha, &dummy_twiddles);
    //     let mut drp_evals = drp_evals.values.into_iter().collect_vec();
    //     IcicleBackend::bit_reverse_column(&mut drp_evals);

    //     assert_eq!(drp_evals.len(), DEGREE / 2);
    //     for (i, (&drp_eval, x)) in zip(&drp_evals, drp_domain).enumerate() {
    //         let f_e: SecureField = even_poly.eval_at_point(x.into());
    //         let f_o: SecureField = odd_poly.eval_at_point(x.into());
    //         assert_eq!(drp_eval, (f_e + alpha * f_o).double(), "mismatch at {i}");
    //     }
    // }

    // #[test]
    // fn test_icicle_fold_line() {
    //     let mut is_correct = true;
    //     for log_size in 1..24 {
    //         let mut rng = SmallRng::seed_from_u64(0);
    //         let values = (0..1 << log_size).map(|_| rng.gen()).collect_vec();
    //         let alpha = qm31!(1, 3, 5, 7);
    //         let domain = LineDomain::new(CanonicCoset::new(log_size + 1).half_coset());

    //         let secure_column: SecureColumnByCoords<_> = values.iter().copied().collect();
    //         let line_evaluation = LineEvaluation::new(domain, secure_column);
    //         let cpu_fold = CpuBackend::fold_line(
    //             &line_evaluation,
    //             alpha,
    //             &CpuBackend::precompute_twiddles(domain.coset()),
    //         );

    //         let line_evaluation = LineEvaluation::new(domain, values.into_iter().collect());
    //         let dummy_twiddles = IcicleBackend::precompute_twiddles(domain.coset());
    //         let icicle_fold = IcicleBackend::fold_line(&line_evaluation, alpha, &dummy_twiddles);

    //         if icicle_fold.values.to_vec() != cpu_fold.values.to_vec() {
    //             println!("failed to fold log2: {}", log_size);
    //             is_correct = false;
    //         }
    //     }
    //     assert!(is_correct);
    // }

    // #[test]
    // fn test_icicle_fold_circle_into_line() {
    //     let mut is_correct = true;
    //     for log_size in 1..20 {
    //         let values: Vec<SecureField> = (0..(1 << log_size))
    //             .map(|i| qm31!(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
    //             .collect();
    //         let alpha = qm31!(1, 3, 5, 7);
    //         let circle_domain = CanonicCoset::new(log_size).circle_domain();
    //         let line_domain = LineDomain::new(circle_domain.half_coset);
    //         let mut icicle_fold = LineEvaluation::new(
    //             line_domain,
    //             SecureColumnByCoords::zeros(1 << (log_size - 1)),
    //         );
    //         IcicleBackend::fold_circle_into_line(
    //             &mut icicle_fold,
    //             &SecureEvaluation::new(circle_domain, values.iter().copied().collect()),
    //             alpha,
    //             &IcicleBackend::precompute_twiddles(line_domain.coset()),
    //         );

    //         let mut simd_fold = LineEvaluation::new(
    //             line_domain,
    //             SecureColumnByCoords::zeros(1 << (log_size - 1)),
    //         );
    //         SimdBackend::fold_circle_into_line(
    //             &mut simd_fold,
    //             &SecureEvaluation::new(circle_domain, values.iter().copied().collect()),
    //             alpha,
    //             &SimdBackend::precompute_twiddles(line_domain.coset()),
    //         );

    //         if icicle_fold.values.to_vec() != simd_fold.values.to_vec() {
    //             println!("failed to fold log2: {}", log_size);
    //             is_correct = false;
    //         }
    //     }
    //     assert!(is_correct);
    // }
    // #[test]
    // fn test_icicle_quotients() {
    //     const LOG_SIZE: u32 = 19;
    //     const LOG_BLOWUP_FACTOR: u32 = 1;
    //     let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
    //     let eval_domain = CanonicCoset::new(LOG_SIZE + 1).circle_domain();
    //     let eval = polynomial.evaluate(eval_domain);

    //     let point = SECURE_FIELD_CIRCLE_GEN;
    //     let value = polynomial.eval_at_point(point);
    //     let coeff = qm31!(1, 2, 3, 4);
    //     let quot_eval_cpu = CpuBackend::accumulate_quotients(
    //         eval_domain,
    //         &[&eval],
    //         coeff,
    //         &[ColumnSampleBatch {
    //             point,
    //             columns_and_values: vec![(0, value)],
    //         }],
    //         LOG_BLOWUP_FACTOR,
    //     )
    //     .to_vec();
    //     let polynomial_icicle =
    //         IcicleCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
    //     let eval_icicle = polynomial_icicle.evaluate(eval_domain);
    //     let quot_eval_icicle = IcicleBackend::accumulate_quotients(
    //         eval_domain,
    //         &[&eval_icicle],
    //         coeff,
    //         &[ColumnSampleBatch {
    //             point,
    //             columns_and_values: vec![(0, value)],
    //         }],
    //         LOG_BLOWUP_FACTOR,
    //     )
    //     .to_vec();
    //     assert_eq!(quot_eval_cpu, quot_eval_icicle);
    // }
}
