#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use stwo_prover::core::air::accumulation::AccumulationOps;
use stwo_prover::core::backend::cpu::CpuBackend;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;

const LOG_SIZE: usize = 28;
const SIZE: usize = 1 << LOG_SIZE;

pub fn cpu_accumulate(_c: &mut Criterion) {
    // #[cfg(feature = "icicle")]
    // {
    //     use icicle_core::vec_ops::{accumulate_scalars, VecOpsConfig};
    //     use icicle_cuda_runtime::memory::DeviceVec;

    //     let cfg = VecOpsConfig::default();

    //     use icicle_m31::field::QuarticExtensionField;
    //     let data = SecureColumnByCoords::<CpuBackend> {
    //         columns: std::array::from_fn(|i| vec![BaseField::from_u32_unchecked(i as u32);
    // SIZE]),     };

    //     let data2 = data.clone();
    //     let bench_id = format!("icicle accumulate SecureColumn 2^{LOG_SIZE}");

    //     let d_a_slice;
    //     let n = data.columns[0].len();

    //     let mut col_a;
    //     col_a = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
    //     d_a_slice = &mut col_a[..];
    //     SecureColumnByCoords::convert_to_icicle(&data, d_a_slice);

    //     let d_b_slice;
    //     let mut col_b = DeviceVec::<QuarticExtensionField>::cuda_malloc(n).unwrap();
    //     d_b_slice = &mut col_b[..];
    //     SecureColumnByCoords::convert_to_icicle(&data2, d_b_slice);

    //     c.bench_function(&bench_id, |b| {
    //         b.iter(|| accumulate_scalars(d_a_slice, d_b_slice, &cfg).unwrap());
    //     });
    // }

    #[cfg(not(feature = "icicle"))]
    {
        let mut data = SecureColumnByCoords::<CpuBackend> {
            columns: std::array::from_fn(|i| vec![BaseField::from_u32_unchecked(i as u32); SIZE]),
        };
        let data2 = data.clone();
        let bench_id = format!("cpu accumulate SecureColumn 2^{LOG_SIZE}");
        _c.bench_function(&bench_id, |b| {
            b.iter(|| CpuBackend::accumulate(&mut data, &data2));
        });
    }
}

pub fn simd_accumulate(c: &mut Criterion) {
    let cpu_col = SecureColumnByCoords::<CpuBackend> {
        columns: std::array::from_fn(|i| vec![BaseField::from_u32_unchecked(i as u32); SIZE]),
    };

    let columns = cpu_col.columns.map(|col| col.into_iter().collect());
    let data = SecureColumnByCoords::<SimdBackend> { columns };

    let mut data2 = data.clone();
    let bench_id = format!("simd accumulate SecureColumn 2^{LOG_SIZE}");
    c.bench_function(&bench_id, |b| {
        b.iter_batched(
            || data.clone(),
            |mut data| SimdBackend::accumulate(&mut data, &mut data2),
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_accumulate, cpu_accumulate);
criterion_main!(bit_rev);
