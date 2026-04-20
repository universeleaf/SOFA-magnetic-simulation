#define SOFA_CORE_MAPPING_CPP
#include <sofa/core/Mapping.inl>

#include <ElasticRodGuidewire/ElasticRodCollisionMapping.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>

namespace elastic_rod_guidewire
{

ElasticRodCollisionMapping::ElasticRodCollisionMapping()
{
    m_jacobians.resize(1);
    m_jacobians[0] = &m_jacobian;
}

void ElasticRodCollisionMapping::init()
{
    auto* from = this->getFromModel();
    auto* to = this->getToModel();
    if (from == nullptr || to == nullptr)
    {
        Inherit::init();
        return;
    }

    const sofa::Size nodeCount = from->getSize();
    to->resize(nodeCount);

    Inherit::init();

    m_jacobian.compressedMatrix.resize(nodeCount * 3, nodeCount * 6);
    m_jacobian.compressedMatrix.reserve(nodeCount * 3);
    for (sofa::Size i = 0; i < nodeCount; ++i)
    {
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            const sofa::Size row = static_cast<sofa::Size>(3 * i + axis);
            const sofa::Size col = static_cast<sofa::Size>(6 * i + axis);
            m_jacobian.compressedMatrix.startVec(row);
            m_jacobian.compressedMatrix.insertBack(row, col) = kMToMm;
        }
    }
    m_jacobian.compressedMatrix.finalize();
}

void ElasticRodCollisionMapping::apply(const sofa::core::MechanicalParams*, OutDataVecCoord& out, const InDataVecCoord& in)
{
    OutVecCoord& child = *out.beginEdit();
    const InVecCoord& parent = in.getValue();
    child.resize(parent.size());
    for (std::size_t i = 0; i < parent.size(); ++i)
    {
        child[i][0] = kMToMm * parent[i][0];
        child[i][1] = kMToMm * parent[i][1];
        child[i][2] = kMToMm * parent[i][2];
    }
    out.endEdit();
}

void ElasticRodCollisionMapping::applyJ(const sofa::core::MechanicalParams*, OutDataVecDeriv& out, const InDataVecDeriv& in)
{
    OutVecDeriv& child = *out.beginEdit();
    const InVecDeriv& parent = in.getValue();
    child.resize(parent.size());
    for (std::size_t i = 0; i < parent.size(); ++i)
    {
        child[i][0] = kMToMm * parent[i][0];
        child[i][1] = kMToMm * parent[i][1];
        child[i][2] = kMToMm * parent[i][2];
    }
    out.endEdit();
}

void ElasticRodCollisionMapping::applyJT(const sofa::core::MechanicalParams*, InDataVecDeriv& out, const OutDataVecDeriv& in)
{
    InVecDeriv& parent = *out.beginEdit();
    const OutVecDeriv& child = in.getValue();
    if (parent.size() < child.size())
        parent.resize(child.size());
    for (std::size_t i = 0; i < child.size(); ++i)
    {
        parent[i][0] += kSceneForceToNewton * child[i][0];
        parent[i][1] += kSceneForceToNewton * child[i][1];
        parent[i][2] += kSceneForceToNewton * child[i][2];
    }
    out.endEdit();
}

void ElasticRodCollisionMapping::applyJT(const sofa::core::ConstraintParams*, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in)
{
    InMatrixDeriv& parent = *out.beginEdit();
    const OutMatrixDeriv& child = in.getValue();

    const auto rowEnd = child.end();
    for (auto rowIt = child.begin(); rowIt != rowEnd; ++rowIt)
    {
        auto colIt = rowIt.begin();
        const auto colEnd = rowIt.end();
        if (colIt == colEnd)
            continue;

        auto row = parent.writeLine(rowIt.index());
        while (colIt != colEnd)
        {
            InDeriv mapped;
            mapped.clear();
            mapped[0] = kMToMm * colIt.val()[0];
            mapped[1] = kMToMm * colIt.val()[1];
            mapped[2] = kMToMm * colIt.val()[2];
            row.addCol(colIt.index(), mapped);
            ++colIt;
        }
    }

    out.endEdit();
}

void ElasticRodCollisionMapping::computeAccFromMapping(const sofa::core::MechanicalParams*, OutDataVecDeriv& accOut, const InDataVecDeriv&, const InDataVecDeriv& accIn)
{
    OutVecDeriv& child = *accOut.beginEdit();
    const InVecDeriv& parent = accIn.getValue();
    child.resize(parent.size());
    for (std::size_t i = 0; i < parent.size(); ++i)
    {
        child[i][0] = kMToMm * parent[i][0];
        child[i][1] = kMToMm * parent[i][1];
        child[i][2] = kMToMm * parent[i][2];
    }
    accOut.endEdit();
}

const sofa::linearalgebra::BaseMatrix* ElasticRodCollisionMapping::getJ()
{
    return &m_jacobian;
}

const sofa::type::vector<sofa::linearalgebra::BaseMatrix*>* ElasticRodCollisionMapping::getJs()
{
    return &m_jacobians;
}

int ElasticRodCollisionMappingClass = sofa::core::RegisterObject(
    "Maps the SI native rod state [m,rad] to collision geometry in scene millimetres and propagates forces/constraints back consistently."
).add<ElasticRodCollisionMapping>();

template class sofa::core::Mapping<CarrierTypes, sofa::defaulttype::Vec3dTypes>;

} // namespace elastic_rod_guidewire
