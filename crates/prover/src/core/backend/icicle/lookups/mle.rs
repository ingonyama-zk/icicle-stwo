use crate::core::backend::icicle::IcicleBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::mle::MleOps;

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
