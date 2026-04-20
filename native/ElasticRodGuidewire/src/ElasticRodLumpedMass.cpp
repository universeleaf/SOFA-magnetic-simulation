#include <ElasticRodGuidewire/ElasticRodLumpedMass.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/linearalgebra/BaseMatrix.h>

#include <algorithm>
#include <vector>

namespace elastic_rod_guidewire
{

namespace
{
constexpr double kEps = 1.0e-12;
}

ElasticRodLumpedMass::ElasticRodLumpedMass()
    : d_initialNodes(initData(&d_initialNodes, "initialNodes", "Optional explicit initial guidewire node positions in scene units (mm)."))
    , d_undeformedNodes(initData(&d_undeformedNodes, "undeformedNodes", "Optional undeformed guidewire nodes in scene units (mm)."))
    , d_rho(initData(&d_rho, static_cast<Real>(6500.0), "rho", "Mass density in kg/m^3."))
    , d_rodRadius(initData(&d_rodRadius, static_cast<Real>(0.35), "rodRadius", "Guidewire contact outer radius in mm."))
    , d_mechanicalCoreRadiusMm(initData(&d_mechanicalCoreRadiusMm, static_cast<Real>(0.20), "mechanicalCoreRadiusMm", "Mechanical core radius used for mass and rotary inertia in mm."))
    , d_dt(initData(&d_dt, static_cast<Real>(1.0e-4), "dt", "Time step in seconds."))
    , d_youngHead(initData(&d_youngHead, static_cast<Real>(1.8e10), "youngHead", "Effective distal soft-tip Young modulus in Pa."))
    , d_youngBody(initData(&d_youngBody, static_cast<Real>(5.5e10), "youngBody", "Body Young modulus in Pa."))
    , d_shearHead(initData(&d_shearHead, static_cast<Real>(6.766917293233083e9), "shearHead", "Effective distal soft-tip shear modulus in Pa."))
    , d_shearBody(initData(&d_shearBody, static_cast<Real>(2.067669172932331e10), "shearBody", "Body shear modulus in Pa."))
    , d_rodLength(initData(&d_rodLength, static_cast<Real>(400.0), "rodLength", "Guidewire length in mm."))
    , d_magneticEdgeCount(initData(&d_magneticEdgeCount, 5u, "magneticEdgeCount", "Number of distal magnetic edges; semantically matches elasticRod.cpp."))
    , d_softTipEdgeCount(initData(&d_softTipEdgeCount, 8u, "softTipEdgeCount", "Number of distal edges that use the softer segmented tip stiffness."))
    , d_debugLumpedMass(initData(&d_debugLumpedMass, "debugLumpedMass", "Per-node translational lumped masses in kg."))
    , d_debugLumpedRotInertia(initData(&d_debugLumpedRotInertia, "debugLumpedRotInertia", "Per-node scalar twist inertias in kg.m^2."))
    , d_debugTotalMass(initData(&d_debugTotalMass, static_cast<Real>(0.0), "debugTotalMass", "Total rod mass in kg."))
{
}

void ElasticRodLumpedMass::configureCoreFromData(const VecCoord& positions)
{
    std::vector<Coord> initialCoords;
    if (!d_initialNodes.getValue().empty())
    {
        initialCoords.resize(d_initialNodes.getValue().size());
        for (std::size_t i = 0; i < d_initialNodes.getValue().size(); ++i)
        {
            initialCoords[i].clear();
            setCoordCenter(initialCoords[i], d_initialNodes.getValue()[i]);
            setCoordTheta(initialCoords[i], static_cast<Real>(0.0));
        }
    }
    else
    {
        initialCoords.resize(positions.size());
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            initialCoords[i].clear();
            setCoordCenter(initialCoords[i], coordCenter(positions[i]) / kMmToM);
            setCoordTheta(initialCoords[i], coordTheta(positions[i]));
            clearUnused(initialCoords[i]);
        }
    }

    std::vector<Vec3> undeformedNodes;
    undeformedNodes.reserve(d_undeformedNodes.getValue().size());
    for (const Vec3& p : d_undeformedNodes.getValue())
        undeformedNodes.push_back(p);

    m_core.configure(
        d_rho.getValue(),
        d_mechanicalCoreRadiusMm.getValue(),
        d_dt.getValue(),
        d_youngHead.getValue(),
        d_youngBody.getValue(),
        d_shearHead.getValue(),
        d_shearBody.getValue(),
        d_rodLength.getValue(),
        static_cast<std::size_t>(d_magneticEdgeCount.getValue()),
        static_cast<std::size_t>(d_softTipEdgeCount.getValue()));
    m_core.initialize(initialCoords, undeformedNodes);
    m_cachedNodeCount = positions.size();

    d_debugLumpedMass.setValue(VecReal(m_core.lumpedMassKg().begin(), m_core.lumpedMassKg().end()));
    m_effectiveThetaInertia.assign(m_core.lumpedThetaInertiaKgM2().size(), static_cast<Real>(0.0));
    for (std::size_t i = 0; i < m_effectiveThetaInertia.size(); ++i)
    {
        const Real charLen = i < m_core.voronoiLen().size() ? m_core.voronoiLen()[i] : static_cast<Real>(0.0);
        const Real generalizedFloor = m_core.lumpedMassKg()[i] * charLen * charLen;
        m_effectiveThetaInertia[i] = std::max(m_core.lumpedThetaInertiaKgM2()[i], generalizedFloor);
    }
    d_debugLumpedRotInertia.setValue(VecReal(m_effectiveThetaInertia.begin(), m_effectiveThetaInertia.end()));
    d_debugTotalMass.setValue(m_core.totalMassKg());
    m_dummyInertia = m_effectiveThetaInertia;
}

void ElasticRodLumpedMass::init()
{
    Inherit::init();
    if (this->mstate == nullptr)
    {
        msg_error() << "ElasticRodLumpedMass requires a Vec6d MechanicalObject in the same node.";
        return;
    }

    const VecCoord& x0 = this->mstate->read(sofa::core::vec_id::read_access::position)->getValue();
    configureCoreFromData(x0);
    msg_info() << "ElasticRodLumpedMass initialized: nodeCount=" << x0.size()
               << ", contactRadiusMm=" << d_rodRadius.getValue()
               << ", mechanicalCoreRadiusMm=" << d_mechanicalCoreRadiusMm.getValue()
               << ", totalMass=" << m_core.totalMassKg() << " kg";
}

void ElasticRodLumpedMass::reinit()
{
    Inherit::reinit();
    if (this->mstate == nullptr)
        return;
    const VecCoord& x0 = this->mstate->read(sofa::core::vec_id::read_access::position)->getValue();
    configureCoreFromData(x0);
}

void ElasticRodLumpedMass::addMDx(const sofa::core::MechanicalParams*, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor)
{
    VecDeriv& out = *f.beginEdit();
    const VecDeriv& in = dx.getValue();
    if (out.size() < in.size())
        out.resize(in.size());
    const std::size_t count = std::min<std::size_t>(in.size(), m_core.lumpedMassKg().size());
    for (std::size_t i = 0; i < count; ++i)
    {
        out[i][0] += factor * m_core.lumpedMassKg()[i] * in[i][0];
        out[i][1] += factor * m_core.lumpedMassKg()[i] * in[i][1];
        out[i][2] += factor * m_core.lumpedMassKg()[i] * in[i][2];
        out[i][3] += factor * m_effectiveThetaInertia[i] * in[i][3];
        out[i][4] += factor * m_dummyInertia[i] * in[i][4];
        out[i][5] += factor * m_dummyInertia[i] * in[i][5];
    }
    f.endEdit();
}

void ElasticRodLumpedMass::accFromF(const sofa::core::MechanicalParams*, DataVecDeriv& a, const DataVecDeriv& f)
{
    VecDeriv& out = *a.beginEdit();
    const VecDeriv& in = f.getValue();
    if (out.size() < in.size())
        out.resize(in.size());
    const std::size_t count = std::min<std::size_t>(in.size(), m_core.lumpedMassKg().size());
    for (std::size_t i = 0; i < count; ++i)
    {
        const Real m = std::max(m_core.lumpedMassKg()[i], static_cast<Real>(kEps));
        const Real I = std::max(m_effectiveThetaInertia[i], static_cast<Real>(kEps));
        out[i][0] += in[i][0] / m;
        out[i][1] += in[i][1] / m;
        out[i][2] += in[i][2] / m;
        out[i][3] += in[i][3] / I;
    }
    a.endEdit();
}

void ElasticRodLumpedMass::addForce(const sofa::core::MechanicalParams*, DataVecDeriv& f, const DataVecCoord&, const DataVecDeriv&)
{
    VecDeriv& out = *f.beginEdit();
    if (out.size() < m_cachedNodeCount)
        out.resize(m_cachedNodeCount);

    const Vec3 gravity = this->getContext() != nullptr
        ? kMmToM * Vec3(this->getContext()->getGravity())
        : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));

    const std::size_t count = std::min<std::size_t>(out.size(), m_core.lumpedMassKg().size());
    for (std::size_t i = 0; i < count; ++i)
    {
        out[i][0] += m_core.lumpedMassKg()[i] * gravity[0];
        out[i][1] += m_core.lumpedMassKg()[i] * gravity[1];
        out[i][2] += m_core.lumpedMassKg()[i] * gravity[2];
    }
    f.endEdit();
}

SReal ElasticRodLumpedMass::getKineticEnergy(const sofa::core::MechanicalParams*, const DataVecDeriv& v) const
{
    const VecDeriv& vel = v.getValue();
    const std::size_t count = std::min<std::size_t>(vel.size(), m_core.lumpedMassKg().size());
    Real energy = static_cast<Real>(0.0);
    for (std::size_t i = 0; i < count; ++i)
    {
        const Vec3 vM = derivCenter(vel[i]);
        energy += static_cast<Real>(0.5) * m_core.lumpedMassKg()[i] * sofa::type::dot(vM, vM);
        energy += static_cast<Real>(0.5) * m_effectiveThetaInertia[i] * vel[i][3] * vel[i][3];
    }
    return static_cast<SReal>(energy);
}

SReal ElasticRodLumpedMass::getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord& x) const
{
    const VecCoord& q = x.getValue();
    const Vec3 gravityM = this->getContext() != nullptr
        ? kMmToM * Vec3(this->getContext()->getGravity())
        : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    const std::size_t count = std::min<std::size_t>(q.size(), m_core.lumpedMassKg().size());
    Real energy = static_cast<Real>(0.0);
    for (std::size_t i = 0; i < count; ++i)
        energy -= m_core.lumpedMassKg()[i] * sofa::type::dot(gravityM, coordCenter(q[i]));
    return static_cast<SReal>(energy);
}

ElasticRodLumpedMass::Vec6 ElasticRodLumpedMass::getMomentum(const sofa::core::MechanicalParams*, const DataVecCoord&, const DataVecDeriv& v) const
{
    Vec6 momentum;
    momentum.clear();
    const VecDeriv& vel = v.getValue();
    const std::size_t count = std::min<std::size_t>(vel.size(), m_core.lumpedMassKg().size());
    for (std::size_t i = 0; i < count; ++i)
    {
        const Vec3 linear = m_core.lumpedMassKg()[i] * derivCenter(vel[i]);
        momentum[0] += linear[0];
        momentum[1] += linear[1];
        momentum[2] += linear[2];
        momentum[3] += m_effectiveThetaInertia[i] * vel[i][3];
    }
    return momentum;
}

void ElasticRodLumpedMass::addGravityToV(const sofa::core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if (mparams == nullptr)
        return;
    VecDeriv& v = *d_v.beginEdit();
    const Vec3 gravity = this->getContext() != nullptr
        ? kMmToM * Vec3(this->getContext()->getGravity())
        : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    const Real dt = static_cast<Real>(mparams->dt());
    for (auto& dof : v)
    {
        dof[0] += dt * gravity[0];
        dof[1] += dt * gravity[1];
        dof[2] += dt * gravity[2];
    }
    d_v.endEdit();
}

void ElasticRodLumpedMass::addMToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal mFact, unsigned int& offset)
{
    if (matrix == nullptr)
        return;
    const std::size_t count = m_core.lumpedMassKg().size();
    for (std::size_t i = 0; i < count; ++i)
    {
        matrix->add(offset + static_cast<unsigned int>(6 * i + 0), offset + static_cast<unsigned int>(6 * i + 0), mFact * m_core.lumpedMassKg()[i]);
        matrix->add(offset + static_cast<unsigned int>(6 * i + 1), offset + static_cast<unsigned int>(6 * i + 1), mFact * m_core.lumpedMassKg()[i]);
        matrix->add(offset + static_cast<unsigned int>(6 * i + 2), offset + static_cast<unsigned int>(6 * i + 2), mFact * m_core.lumpedMassKg()[i]);
        matrix->add(offset + static_cast<unsigned int>(6 * i + 3), offset + static_cast<unsigned int>(6 * i + 3), mFact * m_effectiveThetaInertia[i]);
        matrix->add(offset + static_cast<unsigned int>(6 * i + 4), offset + static_cast<unsigned int>(6 * i + 4), mFact * m_dummyInertia[i]);
        matrix->add(offset + static_cast<unsigned int>(6 * i + 5), offset + static_cast<unsigned int>(6 * i + 5), mFact * m_dummyInertia[i]);
    }
}

SReal ElasticRodLumpedMass::getElementMass(sofa::Index index) const
{
    if (index >= m_core.lumpedMassKg().size())
        return static_cast<SReal>(0.0);
    return static_cast<SReal>(m_core.lumpedMassKg()[index]);
}

void ElasticRodLumpedMass::getElementMass(sofa::Index index, sofa::linearalgebra::BaseMatrix* m) const
{
    if (m == nullptr || index >= m_core.lumpedMassKg().size())
        return;
    m->add(0, 0, m_core.lumpedMassKg()[index]);
    m->add(1, 1, m_core.lumpedMassKg()[index]);
    m->add(2, 2, m_core.lumpedMassKg()[index]);
    m->add(3, 3, m_effectiveThetaInertia[index]);
    m->add(4, 4, m_dummyInertia[index]);
    m->add(5, 5, m_dummyInertia[index]);
}

int ElasticRodLumpedMassClass = sofa::core::RegisterObject(
    "Reduced-state native lumped mass / scalar twist inertia model aligned with ElasticRodCompatCore."
).add<ElasticRodLumpedMass>();

} // namespace elastic_rod_guidewire
