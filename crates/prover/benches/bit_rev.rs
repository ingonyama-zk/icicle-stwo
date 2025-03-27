#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use stwo_prover::core::fields::m31::BaseField;

const LOG_SIZE: usize = 28;
const SIZE: usize = 1 << LOG_SIZE;

// pub fn cpu_bit_rev(c: &mut Criterion) {
//     use stwo_prover::core::backend::cpu::bit_reverse;
//     // TODO(andrew): Consider using same size for all.

//     let mut data = (0..SIZE).map(BaseField::from).collect_vec();

//     use std::slice;

//     use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
//     use icicle_m31::field::ScalarField;
//     let rr = unsafe { slice::from_raw_parts_mut(data.as_mut_ptr() as *mut ScalarField, SIZE) };
//     let v = HostSlice::from_mut_slice(rr);
//     let mut data = DeviceVec::cuda_malloc(SIZE).unwrap();
//     data.copy_from_host(v).unwrap();
//     let rr = unsafe { slice::from_raw_parts_mut(data.as_mut_ptr() as *mut ScalarField, SIZE) };

//     let bench_id = &format!("cpu bit_rev 2^{LOG_SIZE}");
//     c.bench_function(bench_id, |b| {
//         #[cfg(not(feature = "icicle"))]
//         b.iter_batched(
//             || data.clone(),
//             |mut data| bit_reverse(&mut data),
//             BatchSize::LargeInput,
//         );

//         #[cfg(feature = "icicle")]
//         {
//             b.iter(|| bit_reverse(rr));
//         }
//     });
// }

pub fn simd_bit_rev(c: &mut Criterion) {
    use stwo_prover::core::backend::simd::bit_reverse::bit_reverse_m31;
    use stwo_prover::core::backend::simd::column::BaseColumn;
    let data = (0..SIZE).map(BaseField::from).collect::<BaseColumn>();

    let bench_id = &format!("simd bit_rev 2^{LOG_SIZE}");
    c.bench_function(bench_id, |b| {
        b.iter_batched(
            || data.data.clone(),
            |mut data| bit_reverse_m31(&mut data),
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    name = bit_rev;
    config = Criterion::default().sample_size(10);
    targets = simd_bit_rev);
criterion_main!(bit_rev);
