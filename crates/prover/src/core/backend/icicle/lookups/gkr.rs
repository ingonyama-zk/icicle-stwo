use crate::core::{backend::icicle::IcicleBackend, lookups::gkr_prover::GkrOps};


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