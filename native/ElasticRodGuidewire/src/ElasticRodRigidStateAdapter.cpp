#define SOFA_CORE_MAPPING_CPP
#include <sofa/core/Mapping.inl>

#include <ElasticRodGuidewire/ElasticRodRigidStateAdapter.h>
#include <ElasticRodGuidewire/ElasticRodGuidewireModel.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/logging/Messaging.h>

namespace elastic_rod_guidewire
{

ElasticRodRigidStateAdapter::ElasticRodRigidStateAdapter() = default;

void ElasticRodRigidStateAdapter::init()
{
    Inherit::init();
    auto* from = this->getFromModel();
    if (from == nullptr)
    {
        msg_error() << "ElasticRodRigidStateAdapter requires a Vec6d input state.";
        return;
    }
    auto* fromContext = from->getContext();
    m_rodModel = fromContext != nullptr
        ? fromContext->template get<ElasticRodGuidewireModel>(sofa::core::objectmodel::BaseContext::Local)
        : nullptr;
    if (m_rodModel == nullptr)
        msg_error() << "ElasticRodRigidStateAdapter could not find ElasticRodGuidewireModel next to the input state.";
}

void ElasticRodRigidStateAdapter::apply(const sofa::core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in)
{
    SOFA_UNUSED(mparams);
    OutVecCoord& rigid = *out.beginEdit();
    const InVecCoord& q = in.getValue();
    if (m_rodModel != nullptr)
        m_rodModel->buildRigidState(q, rigid);
    out.endEdit();
}

void ElasticRodRigidStateAdapter::applyJ(const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in)
{
    SOFA_UNUSED(mparams);
    OutVecDeriv& rigidVel = *out.beginEdit();
    const InVecDeriv& v = in.getValue();
    const InVecCoord* q = nullptr;
    if (auto* from = this->getFromModel())
        q = &from->read(sofa::core::vec_id::read_access::position)->getValue();
    if (m_rodModel != nullptr && q != nullptr)
        m_rodModel->buildRigidVelocity(*q, v, rigidVel);
    out.endEdit();
}

void ElasticRodRigidStateAdapter::applyJT(const sofa::core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(out);
    SOFA_UNUSED(in);
}

void ElasticRodRigidStateAdapter::applyJT(const sofa::core::ConstraintParams* cparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in)
{
    SOFA_UNUSED(cparams);
    SOFA_UNUSED(out);
    SOFA_UNUSED(in);
}

void ElasticRodRigidStateAdapter::computeAccFromMapping(const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& accOut, const InDataVecDeriv& vIn, const InDataVecDeriv& accIn)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(vIn);
    OutVecDeriv& rigidAcc = *accOut.beginEdit();
    const InVecDeriv& acc = accIn.getValue();
    const InVecCoord* q = nullptr;
    if (auto* from = this->getFromModel())
        q = &from->read(sofa::core::vec_id::read_access::position)->getValue();
    if (m_rodModel != nullptr && q != nullptr)
        m_rodModel->buildRigidVelocity(*q, acc, rigidAcc);
    accOut.endEdit();
}

int ElasticRodRigidStateAdapterClass = sofa::core::RegisterObject(
    "Compatibility adapter from the native SI Vec6d rod state to display/controller Rigid3d state."
).add<ElasticRodRigidStateAdapter>();

template class sofa::core::Mapping<CarrierTypes, sofa::defaulttype::Rigid3Types>;

} // namespace elastic_rod_guidewire
