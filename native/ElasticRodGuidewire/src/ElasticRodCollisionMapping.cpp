#define SOFA_CORE_MAPPING_CPP
#include <sofa/core/Mapping.inl>

#include <ElasticRodGuidewire/ElasticRodCollisionMapping.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>

namespace elastic_rod_guidewire
{

ElasticRodCollisionMapping::ElasticRodCollisionMapping()
    : d_selectedIndices(initData(&d_selectedIndices, "selectedIndices", "Optional parent rod node indices to expose as reduced collision samples."))
{
    m_jacobians.resize(1);
    m_jacobians[0] = &m_jacobian;
}

void ElasticRodCollisionMapping::refreshSelectedIndices(std::size_t parentNodeCount)
{
    m_selectedIndices.clear();
    const auto& configured = d_selectedIndices.getValue();
    if (configured.empty())
    {
        m_selectedIndices.resize(parentNodeCount);
        for (std::size_t i = 0; i < parentNodeCount; ++i)
            m_selectedIndices[i] = i;
        return;
    }

    m_selectedIndices.reserve(configured.size());
    for (unsigned int rawIndex : configured)
    {
        const std::size_t index = static_cast<std::size_t>(rawIndex);
        if (index >= parentNodeCount)
            continue;
        if (!m_selectedIndices.empty() && m_selectedIndices.back() == index)
            continue;
        m_selectedIndices.push_back(index);
    }

    if (m_selectedIndices.size() < 2u)
    {
        m_selectedIndices.resize(parentNodeCount);
        for (std::size_t i = 0; i < parentNodeCount; ++i)
            m_selectedIndices[i] = i;
    }
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
    refreshSelectedIndices(static_cast<std::size_t>(nodeCount));
    const sofa::Size outputCount = static_cast<sofa::Size>(m_selectedIndices.size());
    to->resize(outputCount);

    Inherit::init();

    m_jacobian.compressedMatrix.resize(outputCount * 3, nodeCount * 6);
    m_jacobian.compressedMatrix.reserve(outputCount * 3);
    for (sofa::Size i = 0; i < outputCount; ++i)
    {
        const sofa::Size parentIndex = static_cast<sofa::Size>(m_selectedIndices[static_cast<std::size_t>(i)]);
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            const sofa::Size row = static_cast<sofa::Size>(3 * i + axis);
            const sofa::Size col = static_cast<sofa::Size>(6 * parentIndex + axis);
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
    refreshSelectedIndices(parent.size());
    child.resize(m_selectedIndices.size());
    for (std::size_t i = 0; i < m_selectedIndices.size(); ++i)
    {
        const std::size_t parentIndex = m_selectedIndices[i];
        child[i][0] = kMToMm * parent[parentIndex][0];
        child[i][1] = kMToMm * parent[parentIndex][1];
        child[i][2] = kMToMm * parent[parentIndex][2];
    }
    out.endEdit();
}

void ElasticRodCollisionMapping::applyJ(const sofa::core::MechanicalParams*, OutDataVecDeriv& out, const InDataVecDeriv& in)
{
    OutVecDeriv& child = *out.beginEdit();
    const InVecDeriv& parent = in.getValue();
    refreshSelectedIndices(parent.size());
    child.resize(m_selectedIndices.size());
    for (std::size_t i = 0; i < m_selectedIndices.size(); ++i)
    {
        const std::size_t parentIndex = m_selectedIndices[i];
        child[i][0] = kMToMm * parent[parentIndex][0];
        child[i][1] = kMToMm * parent[parentIndex][1];
        child[i][2] = kMToMm * parent[parentIndex][2];
    }
    out.endEdit();
}

void ElasticRodCollisionMapping::applyJT(const sofa::core::MechanicalParams*, InDataVecDeriv& out, const OutDataVecDeriv& in)
{
    InVecDeriv& parent = *out.beginEdit();
    const OutVecDeriv& child = in.getValue();
    const std::size_t parentNodeCount = this->getFromModel() != nullptr
        ? static_cast<std::size_t>(this->getFromModel()->getSize())
        : parent.size();
    refreshSelectedIndices(parentNodeCount);
    if (parent.size() < parentNodeCount)
        parent.resize(parentNodeCount);
    const std::size_t count = std::min<std::size_t>(child.size(), m_selectedIndices.size());
    for (std::size_t i = 0; i < count; ++i)
    {
        const std::size_t parentIndex = m_selectedIndices[i];
        parent[parentIndex][0] += kSceneForceToNewton * child[i][0];
        parent[parentIndex][1] += kSceneForceToNewton * child[i][1];
        parent[parentIndex][2] += kSceneForceToNewton * child[i][2];
    }
    out.endEdit();
}

void ElasticRodCollisionMapping::applyJT(const sofa::core::ConstraintParams*, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in)
{
    InMatrixDeriv& parent = *out.beginEdit();
    const OutMatrixDeriv& child = in.getValue();
    const std::size_t parentNodeCount = this->getFromModel() != nullptr
        ? static_cast<std::size_t>(this->getFromModel()->getSize())
        : m_selectedIndices.size();
    refreshSelectedIndices(parentNodeCount);

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
            const std::size_t childIndex = static_cast<std::size_t>(colIt.index());
            if (childIndex < m_selectedIndices.size())
                row.addCol(m_selectedIndices[childIndex], mapped);
            ++colIt;
        }
    }

    out.endEdit();
}

void ElasticRodCollisionMapping::computeAccFromMapping(const sofa::core::MechanicalParams*, OutDataVecDeriv& accOut, const InDataVecDeriv&, const InDataVecDeriv& accIn)
{
    OutVecDeriv& child = *accOut.beginEdit();
    const InVecDeriv& parent = accIn.getValue();
    refreshSelectedIndices(parent.size());
    child.resize(m_selectedIndices.size());
    for (std::size_t i = 0; i < m_selectedIndices.size(); ++i)
    {
        const std::size_t parentIndex = m_selectedIndices[i];
        child[i][0] = kMToMm * parent[parentIndex][0];
        child[i][1] = kMToMm * parent[parentIndex][1];
        child[i][2] = kMToMm * parent[parentIndex][2];
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
