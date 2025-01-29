use std::env;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
// #[cfg(feature = "icicle")]
// use stwo_prover::core::backend::icicle::IcicleBackend;
use stwo_prover::core::backend::{Backend, ColumnOps, CpuBackend};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::fri::FriOps;
use stwo_prover::core::poly::circle::{CanonicCoset, PolyOps, SecureEvaluation};
use stwo_prover::core::poly::line::{LineDomain, LineEvaluation};
use stwo_prover::core::poly::twiddles::TwiddleTree;

const ALPHA: SecureField = SecureField::from_u32_unchecked(2213980, 2213981, 2213982, 2213983);

fn get_min_max_log_size() -> (u32, u32) {
    const MIN_LOG2: u32 = 1; // min length = 2 ^ MIN_LOG2
    const MAX_LOG2: u32 = 25; // max length = 2 ^ MAX_LOG2

    let min_log2 = env::var("MIN_LOG2")
        .unwrap_or_else(|_| MIN_LOG2.to_string())
        .parse::<u32>()
        .unwrap_or(MIN_LOG2);
    let max_log2 = env::var("MAX_LOG2")
        .unwrap_or_else(|_| MAX_LOG2.to_string())
        .parse::<u32>()
        .unwrap_or(MAX_LOG2);

    assert!(min_log2 >= MIN_LOG2);
    assert!(min_log2 < max_log2);

    (min_log2, max_log2)
}

fn folding_benchmark(c: &mut Criterion) {
    let (min_log2, max_log2) = get_min_max_log_size();
    for log_size in min_log2..=max_log2 {
        let coset = CanonicCoset::new(log_size + 1);
        let line_domain = LineDomain::new(coset.half_coset());
        let twiddles = CpuBackend::precompute_twiddles(line_domain.coset()); // TODO: actually twiddles are not used

        // fold line
        fn bench_fold_line<B: Backend + FriOps + ColumnOps<BaseField, Column = Vec<BaseField>>>(
            line_domain: LineDomain,
            c: &mut Criterion,
            log_size: u32,
            twiddles: &TwiddleTree<CpuBackend>, // TODO: precompute on IcicleBackend?
            backend_descr: &str,
        ) {
            let evals = LineEvaluation::new(
                line_domain,
                SecureColumnByCoords {
                    columns: std::array::from_fn(|i| {
                        vec![BaseField::from_u32_unchecked(i as u32); 1 << log_size]
                    }),
                },
            );

            c.bench_function(
                &format!("{} fold_line log2 = {}", backend_descr, log_size),
                |b| {
                    b.iter(|| {
                        black_box(B::fold_line(black_box(&evals), black_box(ALPHA), unsafe {
                            std::mem::transmute(&twiddles)
                        }));
                    })
                },
            );
        }

        if max_log2 < 15 {
            // TODO: no need to bench long slow runs
            bench_fold_line::<CpuBackend>(line_domain, c, log_size, &twiddles, "cpu");

            // #[cfg(feature = "icicle")]
            // bench_fold_line::<IcicleBackend>(line_domain, c, log_size, &twiddles, "icicle");
        }

        // fold circle
        fn bench_fold_circle<B: Backend + FriOps + ColumnOps<BaseField>>(
            log_size: u32,
            c: &mut Criterion,
            twiddles: &TwiddleTree<CpuBackend>,
            backend_descr: &str,
        ) where
            SecureColumnByCoords<B>: FromIterator<SecureField>,
        {
            let values: Vec<SecureField> = (0..(1 << log_size))
                .map(|i| SecureField::from_u32_unchecked(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
                .collect();
            let circle_domain = CanonicCoset::new(log_size).circle_domain();
            let circle_eval =
                SecureEvaluation::new(circle_domain, values.iter().copied().collect());

            let line_domain = LineDomain::new(circle_domain.half_coset);
            let mut dst = LineEvaluation::new(
                line_domain,
                SecureColumnByCoords::zeros(1 << (log_size - 1)),
            );

            c.bench_function(
                &format!(
                    "{} fold_circle_into_line log2 = {}",
                    backend_descr, log_size
                ),
                |b| {
                    b.iter(|| {
                        black_box(B::fold_circle_into_line(
                            black_box(&mut dst),
                            black_box(&circle_eval),
                            black_box(ALPHA),
                            unsafe { std::mem::transmute(&twiddles) },
                        ));
                    })
                },
            );
        }

        if max_log2 < 15 {
            // TODO: no need to bench long slow runs
            bench_fold_circle::<CpuBackend>(log_size, c, &twiddles, "cpu");

            // #[cfg(feature = "icicle")]
            // bench_fold_circle::<IcicleBackend>(log_size, c, &twiddles, "icicle");
        }
    }
}

// fn icicle_raw_folding_benchmark(c: &mut Criterion) {
//     black_box(&c);
//     #[cfg(feature = "icicle")]
//     {
//         use std::mem::transmute;

//         use icicle_core::ntt::FieldImpl;
//         use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
//         use icicle_m31::field::{QuarticExtensionField, ScalarField};
//         use icicle_m31::fri::{self, fold_circle_into_line, FriConfig};
//         use stwo_prover::core::fields::FieldExpOps;
//         use stwo_prover::core::fri::{CIRCLE_TO_LINE_FOLD_STEP, FOLD_STEP};
//         use stwo_prover::core::poly::BitReversedOrder;
//         use stwo_prover::core::utils::bit_reverse_index;

//         let (min_log2, max_log2) = get_min_max_log_size();
//         for log_size in min_log2..=max_log2 {
//             let coset = CanonicCoset::new(log_size + 1);
//             let line_domain = LineDomain::new(coset.half_coset());
//             let backend_descr: &str = "icicle raw";

//             // line
//             let evals = LineEvaluation::<IcicleBackend>::new(
//                 line_domain,
//                 SecureColumnByCoords {
//                     columns: std::array::from_fn(
//                         |i| std::vec![BaseField::from_u32_unchecked(i as u32); 1 << log_size],
//                     ),
//                 },
//             );

//             let n = evals.len();
//             assert!(n >= 2, "Evaluation too small");

//             let dom_vals_len = n / 2;

//             let mut domain_vals = Vec::new();
//             let line_domain_log_size = line_domain.log_size();
//             for i in 0..dom_vals_len {
//                 // TODO: on-device batch
//                 // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
//                 let x = line_domain.at(bit_reverse_index(i << FOLD_STEP, line_domain_log_size));
//                 let x = x.inverse();
//                 domain_vals.push(ScalarField::from_u32(x.0));
//             }

//             let domain_icicle_host = HostSlice::from_slice(domain_vals.as_slice());
//             let mut d_domain_icicle =
// DeviceVec::<ScalarField>::cuda_malloc(dom_vals_len).unwrap();             
// d_domain_icicle.copy_from_host(domain_icicle_host).unwrap();

//             let mut d_evals_icicle = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
//             SecureColumnByCoords::<IcicleBackend>::convert_to_icicle(
//                 unsafe { transmute(&evals.values) },
//                 &mut d_evals_icicle,
//             );
//             let mut d_folded_eval =
//                 DeviceVec::<QuarticExtensionField>::cuda_malloc(dom_vals_len).unwrap();

//             let cfg = FriConfig::default();
//             let icicle_alpha = unsafe { transmute(ALPHA) };

//             println!(
//                 "fold line: d_evals_icicle len: {:?} d_folded_eval len: {:?}",
//                 n, dom_vals_len
//             );

//             c.bench_function(
//                 &std::format!("{} fold_line log2 = {}", backend_descr, log_size),
//                 |b| {
//                     b.iter(|| {
//                         black_box(
//                             fri::fold_line(
//                                 black_box(&d_evals_icicle[..]),
//                                 black_box(&d_domain_icicle[..]),
//                                 black_box(&mut d_folded_eval[..]),
//                                 black_box(icicle_alpha),
//                                 black_box(&cfg),
//                             )
//                             .unwrap(),
//                         );
//                     })
//                 },
//             );

//             // circle
//             let values: Vec<SecureField> = (0..(1 << log_size))
//                 .map(|i| SecureField::from_u32_unchecked(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
//                 .collect();
//             let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);
//             let circle_domain = CanonicCoset::new(log_size).circle_domain();
//             let circle_evals: SecureEvaluation<IcicleBackend, BitReversedOrder> =
//                 SecureEvaluation::new(circle_domain, values.iter().copied().collect());

//             let dst_len = 1 << (log_size - 1); // n / 2

//             assert_eq!(circle_evals.len() >> CIRCLE_TO_LINE_FOLD_STEP, dst_len);

//             // let circle_domain = circle_evals.domain;
//             let length = circle_evals.values.len();

//             let dom_vals_len = length / 2;

//             let mut domain_rev = Vec::new();
//             for i in 0..dom_vals_len {
//                 // TODO: on-device batch
//                 // TODO(andrew): Inefficient. Update when domain twiddles get stored in a buffer.
//                 let p = circle_domain.at(bit_reverse_index(
//                     i << CIRCLE_TO_LINE_FOLD_STEP,
//                     circle_domain.log_size(),
//                 ));
//                 let p = p.y.inverse();
//                 domain_rev.push(p);
//             }

//             let domain_vals = (0..dom_vals_len)
//                 .map(|i| {
//                     let p = domain_rev[i];
//                     ScalarField::from_u32(p.0)
//                 })
//                 .collect::<Vec<_>>();

//             let domain_icicle_host = HostSlice::from_slice(domain_vals.as_slice());
//             let mut d_domain_icicle =
// DeviceVec::<ScalarField>::cuda_malloc(dom_vals_len).unwrap();             
// d_domain_icicle.copy_from_host(domain_icicle_host).unwrap();

//             let mut d_evals_icicle =
//                 DeviceVec::<QuarticExtensionField>::cuda_malloc(length).unwrap();
//             SecureColumnByCoords::convert_to_icicle(&circle_evals.values, &mut d_evals_icicle);
//             let mut d_folded_eval =
//                 DeviceVec::<QuarticExtensionField>::cuda_malloc(dom_vals_len).unwrap();

//             println!(
//                 "fold circle: d_evals_icicle len: {:?} d_folded_eval len: {:?}, length  {},
// circle_domain.log_size(): {} {}",                 d_evals_icicle.len(),
//                 d_folded_eval.len(),
//                 length,
//                 circle_domain.log_size(),
//                 circle_domain.size(),
//             );

//             let cfg = FriConfig::default();
//             let icicle_alpha = unsafe { transmute(alpha) };

//             c.bench_function(
//                 &std::format!(
//                     "{} fold_circle_into_line log2 = {}",
//                     backend_descr,
//                     log_size
//                 ),
//                 |b| {
//                     b.iter(|| {
//                         black_box(
//                             fold_circle_into_line(
//                                 black_box(&d_evals_icicle[..]),
//                                 black_box(&d_domain_icicle[..]),
//                                 black_box(&mut d_folded_eval[..]),
//                                 black_box(icicle_alpha),
//                                 black_box(&cfg),
//                             )
//                             .unwrap(),
//                         );
//                     })
//                 },
//             );
//         }
//     }
// }

criterion_group!(benches, folding_benchmark);
criterion_main!(benches);
