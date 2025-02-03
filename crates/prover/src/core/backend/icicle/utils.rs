use std::mem::transmute;

use icicle_core::ntt::FieldImpl;
use icicle_core::vec_ops::{fold_scalars, VecOps, VecOpsConfig};
use icicle_cuda_runtime::memory::{DeviceSlice, HostSlice, DeviceVec};
use icicle_m31::field::{ComplexExtensionField, QuarticExtensionField, ScalarField};

use crate::core::backend::Col;
use crate::core::backend::icicle::IcicleBackend;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::QM31;
use crate::core::fields::{ExtensionOf, Field};

use super::column::DeviceColumn;

pub static mut D_INTERIM_RESULTS: *mut DeviceVec<QuarticExtensionField> = std::ptr::null_mut();

macro_rules! select_result_type {
    (1) => {
        ScalarField
    };
    (2) => {
        ComplexExtensionField
    };
    (4) => {
        QuarticExtensionField
    };
    ($other:expr) => {
        compile_error!("Unsupported limbs count")
    };
}

/// Folds values recursively in `O(n)` by a hierarchical application of folding factors.
///
/// i.e. folding `n = 8` values with `folding_factors = [x, y, z]`:
///
/// ```text
///               n2=n1+x*n2
///           /               \
///     n1=n3+y*n4          n2=n5+y*n6
///      /      \            /      \
/// n3=a+z*b  n4=c+z*d  n5=e+z*f  n6=g+z*h
///   /  \      /  \      /  \      /  \
///  a    b    c    d    e    f    g    h
/// ```
///
/// # Panics
///
/// Panics if the number of values is not a power of two or if an incorrect number of of folding
/// factors is provided.
// TODO(Andrew): Can be made to run >10x faster by unrolling lower layers of recursion
pub fn fold<'a, F: Field, E: ExtensionOf<F> + Sized>(
    values: &'a DeviceColumn,
    folding_factors: &'a [E],
) -> E {
    assert!(values.len().is_power_of_two());

    let a = unsafe { transmute::<&DeviceSlice<BaseField>, &DeviceSlice<ScalarField>>(&values.data) };
    let b = HostSlice::from_slice(unsafe { transmute(folding_factors) });
    let mut result = vec![QuarticExtensionField::zero()];
    let res = HostSlice::from_mut_slice(&mut result);

    let cfg = VecOpsConfig::default();

    // TODO: generic macro for selecting appropriate result type
    // let limbs_count: usize = std::mem::size_of::<E>() / 4;
    // type EE = select_result_type!(limbs_count);
    // let limbs_count: usize = std::mem::size_of::<F>() / 4;
    // type FF = select_result_type!(limbs_count);

    unsafe {
        // fold_scalars::<QuarticExtensionField, ScalarField>(a, b, res, &cfg).unwrap();
        fold_scalars::<QuarticExtensionField, ScalarField>(a, b, res, &cfg, &mut (*D_INTERIM_RESULTS)[..]).unwrap();
    }

    // println!("result: {:?}", result);

    unsafe {
        let vec: Vec<E> = transmute(result);
        if let Some(first) = vec.first() {
            *first
        } else {
            panic!("Fold result empty.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    #[test]
    fn test_fold_works() {
        // Example input: power-of-two values and appropriate folding factors
        let values = vec![
            M31(1),
            M31(2),
            M31(3),
            M31(4),
            M31(5),
            M31(6),
            M31(7),
            M31(8),
        ];
        let folding_factors = vec![
            QM31::from_u32_unchecked(2, 0, 0, 0),
            QM31::from_u32_unchecked(3, 0, 0, 0),
            QM31::from_u32_unchecked(4, 0, 0, 0),
        ];

        let values_device = DeviceColumn::from_cpu(&values);
        let result = fold::<M31, QM31>(&values_device, &folding_factors);

        let expected = QM31::from_u32_unchecked(358, 0, 0, 0);
        assert_eq!(result, expected, "Result for simple folding is incorrect");

        // Set the desired length for folding_factors
        let folding_factors_length = 20; // Example length
        let values_length = 1 << folding_factors_length; // 2^folding_factors_length

        // Initialize the `values` vector
        let mut values: Vec<M31> = Vec::with_capacity(values_length);
        #[cfg(feature = "parallel")]
        use rayon::iter::IntoParallelIterator;
        #[cfg(feature = "parallel")]
        use rayon::prelude::*;

        #[cfg(feature = "parallel")]
        let values: Vec<M31> = (1..=values_length)
            .into_par_iter()
            .map(|i| M31(i as u32))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let values: Vec<M31> = (1..=values_length)
            .into_iter()
            .map(|i| M31(i as u32))
            .collect();

        let values_device = DeviceColumn::from_cpu(&values);
        // Initialize the `folding_factors` vector
        let mut folding_factors = Vec::with_capacity(folding_factors_length);
        for i in 2..(2 + folding_factors_length) {
            folding_factors.push(QM31::from_u32_unchecked(i as u32, 0, 0, 0));
        }
        let time = std::time::Instant::now();
        let result = fold::<M31, QM31>(&values_device, &folding_factors);
        let elapsed = time.elapsed();
        println!(
            "Elapsed time for 2^{}: {:?}",
            folding_factors_length, elapsed
        );

        let expected = QM31::from_u32_unchecked(223550878, 0, 0, 0);
        assert_eq!(result, expected, "Result for large folding is incorrect");
    }
}
