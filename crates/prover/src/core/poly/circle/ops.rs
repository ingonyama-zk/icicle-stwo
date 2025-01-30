use itertools::Itertools;

use super::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
use crate::core::backend::Col;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldOps;
use crate::core::poly::twiddles::TwiddleTree;
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

/// Operations on BaseField polynomials.
pub trait PolyOps: FieldOps<BaseField> + Sized {
    // TODO(alont): Use a column instead of this type.
    /// The type for precomputed twiddles.
    type Twiddles;

    /// Creates a [CircleEvaluation] from values ordered according to [CanonicCoset].
    /// Used by the [`CircleEvaluation::new_canonical_ordered()`] function.
    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;

    /// Computes a minimal [CirclePoly] that evaluates to the same values as this evaluation.
    /// Used by the [`CircleEvaluation::interpolate()`] function.
    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        itwiddles: &TwiddleTree<Self>,
    ) -> CirclePoly<Self>;

    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CirclePoly<Self>> {
        columns
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect()
    }

    /// Evaluates the polynomial at a single point.
    /// Used by the [`CirclePoly::eval_at_point()`] function.
    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField;

    /// Extends the polynomial to a larger degree bound.
    /// Used by the [`CirclePoly::extend()`] function.
    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self>;

    /// Evaluates the polynomial at all points in the domain.
    /// Used by the [`CirclePoly::evaluate()`] function.
    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder>;

    fn evaluate_polynomials(
        polynomials: &ColumnVec<CirclePoly<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>> {
        polynomials
            .iter()
            .map(|poly| {
                poly.evaluate_with_twiddles(
                    CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain(),
                    twiddles,
                )
            })
            .collect_vec()
    }

    /// Precomputes twiddles for a given coset.
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self>;
}

#[cfg(test)]
mod tests {
    use crate::core::backend::CpuBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::{CanonicCoset, PolyOps};

    #[cfg(feature = "icicle")]
    #[test]
    fn test_evaluate_polynomials() {
        use itertools::Itertools;

        use crate::core::backend::icicle::column::DeviceColumn;
        use crate::core::backend::icicle::IcicleBackend;
        use crate::core::backend::Column;
        use crate::core::ColumnVec;

        let log_size = 9;
        let log2_cols = 8;
        let log_blowup_factor = 2;

        let size = 1 << log_size;
        let n_cols = 1 << log2_cols;

        let cpu_vals = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();
        let gpu_vals = DeviceColumn::from_cpu(&cpu_vals);

        let trace_coset = CanonicCoset::new(log_size);
        let cpu_evals = CpuBackend::new_canonical_ordered(trace_coset, cpu_vals);
        let gpu_evals = IcicleBackend::new_canonical_ordered(trace_coset, gpu_vals);

        let interpolation_coset = CanonicCoset::new(log_size + log_blowup_factor);
        let cpu_twiddles = CpuBackend::precompute_twiddles(interpolation_coset.half_coset());
        let gpu_twiddles = IcicleBackend::precompute_twiddles(interpolation_coset.half_coset());

        let cpu_poly = CpuBackend::interpolate(cpu_evals, &cpu_twiddles);
        let gpu_poly = IcicleBackend::interpolate(gpu_evals, &gpu_twiddles);

        let mut cpu_cols = ColumnVec::from((0..n_cols).map(|_| cpu_poly.clone()).collect_vec());
        let mut gpu_cols = ColumnVec::from((0..n_cols).map(|_| gpu_poly.clone()).collect_vec());

        let result =
            IcicleBackend::evaluate_polynomials(&mut gpu_cols, log_blowup_factor, &gpu_twiddles);
        let expected_result =
            CpuBackend::evaluate_polynomials(&mut cpu_cols, log_blowup_factor, &cpu_twiddles);

        let expected_vals = expected_result
            .iter()
            .map(|eval| eval.clone().values)
            .collect_vec();
        let icicle_vals = result
            .iter()
            .map(|eval| eval.clone().values.to_cpu())
            .collect_vec();

        assert_eq!(icicle_vals, expected_vals);
    }
}
