#include <ElasticRodGuidewire/ElasticRodCompatCore.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace elastic_rod_guidewire
{

namespace
{
constexpr double kEps = 1.0e-12;
constexpr double kMinCurvatureDenom = 5.0e-3;
constexpr double kPi = 3.14159265358979323846;
const ElasticRodCompatCore::Vec3 kXAxis(1.0, 0.0, 0.0);
const ElasticRodCompatCore::Vec3 kYAxis(0.0, 1.0, 0.0);
const ElasticRodCompatCore::Vec3 kZAxis(0.0, 0.0, 1.0);

void rotateAxisAngle(ElasticRodCompatCore::Vec3& v, const ElasticRodCompatCore::Vec3& axis, double theta)
{
    if (std::abs(theta) <= kEps)
        return;

    const double cs = std::cos(theta);
    const double ss = std::sin(theta);
    v = cs * v + ss * axis.cross(v) + sofa::type::dot(axis, v) * (1.0 - cs) * axis;
}
}

ElasticRodCompatCore::Vec3 ElasticRodCompatCore::safeNormalize(const Vec3& v, const Vec3& fallback)
{
    const Real n = v.norm();
    if (n <= kEps)
    {
        const Real nf = fallback.norm();
        if (nf <= kEps)
            return kZAxis;
        return fallback / nf;
    }
    return v / n;
}

ElasticRodCompatCore::Quat ElasticRodCompatCore::quatFromZTo(const Vec3& direction)
{
    const Vec3 d = safeNormalize(direction, kZAxis);
    const Real c = std::clamp(sofa::type::dot(kZAxis, d), static_cast<Real>(-1.0), static_cast<Real>(1.0));
    if (c >= static_cast<Real>(1.0) - static_cast<Real>(1.0e-10))
        return Quat(0.0, 0.0, 0.0, 1.0);
    if (c <= static_cast<Real>(-1.0) + static_cast<Real>(1.0e-10))
        return Quat(1.0, 0.0, 0.0, 0.0);

    Vec3 axis = safeNormalize(kZAxis.cross(d), kXAxis);
    const Real angle = std::acos(c);
    Quat q(axis, angle);
    q.normalize();
    return q;
}

ElasticRodCompatCore::FrameAxes ElasticRodCompatCore::buildFrameFromTangent(const Vec3& tangentM, const Vec3& preferredM1)
{
    FrameAxes frame;
    frame.m3 = safeNormalize(tangentM, kZAxis);

    Vec3 d1 = preferredM1 - sofa::type::dot(preferredM1, frame.m3) * frame.m3;
    if (d1.norm() <= kEps)
    {
        const Vec3 ref = std::abs(frame.m3.z()) < static_cast<Real>(0.9) ? kZAxis : kYAxis;
        d1 = frame.m3.cross(ref);
    }
    frame.m1 = safeNormalize(d1, kXAxis);
    frame.m2 = safeNormalize(frame.m3.cross(frame.m1), kYAxis);
    frame.m1 = safeNormalize(frame.m2.cross(frame.m3), frame.m1);
    return frame;
}

ElasticRodCompatCore::FrameAxes ElasticRodCompatCore::rotateFrameAboutAxis(const FrameAxes& frame, Real theta)
{
    const Real cs = std::cos(theta);
    const Real ss = std::sin(theta);
    FrameAxes out;
    out.m3 = frame.m3;
    out.m1 = cs * frame.m1 + ss * frame.m2;
    out.m2 = -ss * frame.m1 + cs * frame.m2;
    return out;
}

ElasticRodCompatCore::Quat ElasticRodCompatCore::quatFromFrame(const FrameAxes& frame)
{
    const Real m00 = frame.m1[0];
    const Real m01 = frame.m2[0];
    const Real m02 = frame.m3[0];
    const Real m10 = frame.m1[1];
    const Real m11 = frame.m2[1];
    const Real m12 = frame.m3[1];
    const Real m20 = frame.m1[2];
    const Real m21 = frame.m2[2];
    const Real m22 = frame.m3[2];

    Quat q;
    const Real trace = m00 + m11 + m22;
    if (trace > static_cast<Real>(0.0))
    {
        const Real s = static_cast<Real>(0.5) / std::sqrt(trace + static_cast<Real>(1.0));
        q[3] = static_cast<Real>(0.25) / s;
        q[0] = (m21 - m12) * s;
        q[1] = (m02 - m20) * s;
        q[2] = (m10 - m01) * s;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const Real s = static_cast<Real>(2.0) * std::sqrt(std::max(static_cast<Real>(0.0), static_cast<Real>(1.0) + m00 - m11 - m22));
        q[3] = (m21 - m12) / s;
        q[0] = static_cast<Real>(0.25) * s;
        q[1] = (m01 + m10) / s;
        q[2] = (m02 + m20) / s;
    }
    else if (m11 > m22)
    {
        const Real s = static_cast<Real>(2.0) * std::sqrt(std::max(static_cast<Real>(0.0), static_cast<Real>(1.0) + m11 - m00 - m22));
        q[3] = (m02 - m20) / s;
        q[0] = (m01 + m10) / s;
        q[1] = static_cast<Real>(0.25) * s;
        q[2] = (m12 + m21) / s;
    }
    else
    {
        const Real s = static_cast<Real>(2.0) * std::sqrt(std::max(static_cast<Real>(0.0), static_cast<Real>(1.0) + m22 - m00 - m11));
        q[3] = (m10 - m01) / s;
        q[0] = (m02 + m20) / s;
        q[1] = (m12 + m21) / s;
        q[2] = static_cast<Real>(0.25) * s;
    }
    q.normalize();
    return q;
}

ElasticRodCompatCore::Vec3 ElasticRodCompatCore::parallelTransport(const Vec3& d1, const Vec3& t1, const Vec3& t2)
{
    Vec3 b = t1.cross(t2);
    if (b.norm() <= kEps)
        return buildFrameFromTangent(t2, d1).m1;

    b = safeNormalize(b, kXAxis);
    Vec3 tmp = b - sofa::type::dot(b, t1) * t1;
    b = safeNormalize(tmp, b);
    tmp = b - sofa::type::dot(b, t2) * t2;
    b = safeNormalize(tmp, b);

    const Vec3 n1 = t1.cross(b);
    const Vec3 n2 = t2.cross(b);
    Vec3 d1_2 = sofa::type::dot(d1, t1) * t2 + sofa::type::dot(d1, n1) * n2 + sofa::type::dot(d1, b) * b;
    d1_2 = d1_2 - sofa::type::dot(d1_2, t2) * t2;
    return safeNormalize(d1_2, d1);
}

ElasticRodCompatCore::Real ElasticRodCompatCore::signedAngle(const Vec3& u, const Vec3& v, const Vec3& n)
{
    const Vec3 uu = safeNormalize(u, kXAxis);
    const Vec3 vv = safeNormalize(v, kYAxis);
    const Vec3 w = uu.cross(vv);
    const Real angle = std::atan2(w.norm(), sofa::type::dot(uu, vv));
    return sofa::type::dot(n, w) < 0.0 ? -angle : angle;
}

void ElasticRodCompatCore::configure(
    Real rho,
    Real mechanicalCoreRadiusMm,
    Real dt,
    Real youngHead,
    Real youngBody,
    Real shearHead,
    Real shearBody,
    Real rodLengthMm,
    std::size_t magneticEdgeCount,
    std::size_t softTipEdgeCount)
{
    m_rho = rho;
    m_mechanicalCoreRadiusM = mechanicalCoreRadiusMm * static_cast<Real>(1.0e-3);
    m_dt = dt;
    m_youngHead = youngHead;
    m_youngBody = youngBody;
    m_shearHead = shearHead;
    m_shearBody = shearBody;
    m_rodLengthM = rodLengthMm * static_cast<Real>(1.0e-3);
    m_magneticEdgeCount = magneticEdgeCount;
    m_softTipEdgeCount = softTipEdgeCount;
}

std::vector<ElasticRodCompatCore::Vec3> ElasticRodCompatCore::buildUndeformedCentersM(
    const std::vector<Coord>& initialCoords,
    const std::vector<Vec3>& undeformedNodesMm) const
{
    std::vector<Vec3> out;
    if (!undeformedNodesMm.empty())
    {
        out.reserve(undeformedNodesMm.size());
        for (const Vec3& p : undeformedNodesMm)
            out.push_back(static_cast<Real>(1.0e-3) * p);
        return out;
    }

    out.reserve(initialCoords.size());
    for (const Coord& c : initialCoords)
        out.push_back(static_cast<Real>(1.0e-3) * coordCenter(c));
    return out;
}

void ElasticRodCompatCore::initialize(const std::vector<Coord>& initialCoords, const std::vector<Vec3>& undeformedNodesMm)
{
    m_initialCentersMm.clear();
    m_initialCentersMm.reserve(initialCoords.size());
    for (const Coord& c : initialCoords)
        m_initialCentersMm.push_back(coordCenter(c));

    m_undeformedCentersM = buildUndeformedCentersM(initialCoords, undeformedNodesMm);
    computeReferenceGeometry();
    computeStiffness();
    computeMassDistribution();
    computeRestCurvature();
    initializeCommittedReference(initialCoords);
}

void ElasticRodCompatCore::initializeCommittedReference(const std::vector<Coord>& initialCoords)
{
    const std::size_t n = initialCoords.size();
    if (n < 2)
    {
        m_committedReferenceFrames.clear();
        m_committedTangents.clear();
        m_committedRefTwist.clear();
        return;
    }

    const std::size_t ne = n - 1;
    m_committedTangents.assign(ne, kZAxis);
    for (std::size_t i = 0; i < ne; ++i)
    {
        const Vec3 edge = static_cast<Real>(1.0e-3) * (coordCenter(initialCoords[i + 1]) - coordCenter(initialCoords[i]));
        m_committedTangents[i] = safeNormalize(edge, kZAxis);
    }

    m_committedReferenceFrames.assign(ne, FrameAxes {});
    Vec3 d1 = m_committedTangents[0].cross(Vec3(0.0, 0.0, -1.0));
    if (d1.norm() <= kEps)
        d1 = m_committedTangents[0].cross(kYAxis);
    d1 = safeNormalize(d1, kXAxis);
    m_committedReferenceFrames[0] = buildFrameFromTangent(m_committedTangents[0], d1);
    for (std::size_t i = 1; i < ne; ++i)
    {
        d1 = parallelTransport(m_committedReferenceFrames[i - 1].m1, m_committedTangents[i - 1], m_committedTangents[i]);
        m_committedReferenceFrames[i] = buildFrameFromTangent(m_committedTangents[i], d1);
    }

    m_committedRefTwist.assign(ne, static_cast<Real>(0.0));
}

void ElasticRodCompatCore::computeReferenceGeometry()
{
    const std::size_t n = m_undeformedCentersM.size();
    if (n < 2)
    {
        m_refLen.clear();
        m_voronoiLen.clear();
        m_restFrames.clear();
        m_kappaBar.clear();
        m_undeformedTwist.clear();
        return;
    }

    const std::size_t ne = n - 1;
    m_refLen.assign(ne, 0.0);
    std::vector<Vec3> tangents(ne, kZAxis);
    for (std::size_t i = 0; i < ne; ++i)
    {
        const Vec3 dx = m_undeformedCentersM[i + 1] - m_undeformedCentersM[i];
        m_refLen[i] = dx.norm();
        tangents[i] = safeNormalize(dx, kZAxis);
    }

    m_voronoiLen.assign(n, 0.0);
    m_voronoiLen.front() = static_cast<Real>(0.5) * m_refLen.front();
    m_voronoiLen.back() = static_cast<Real>(0.5) * m_refLen.back();
    for (std::size_t i = 1; i + 1 < n; ++i)
        m_voronoiLen[i] = static_cast<Real>(0.5) * (m_refLen[i - 1] + m_refLen[i]);

    m_restFrames.assign(ne, FrameAxes {});
    Vec3 d1 = tangents[0].cross(Vec3(0.0, 0.0, -1.0));
    if (d1.norm() <= kEps)
        d1 = tangents[0].cross(kYAxis);
    d1 = safeNormalize(d1, kXAxis);
    m_restFrames[0] = buildFrameFromTangent(tangents[0], d1);
    for (std::size_t i = 1; i < ne; ++i)
    {
        d1 = parallelTransport(m_restFrames[i - 1].m1, tangents[i - 1], tangents[i]);
        m_restFrames[i] = buildFrameFromTangent(tangents[i], d1);
    }
}

void ElasticRodCompatCore::computeStiffness()
{
    const std::size_t ne = m_refLen.size();
    m_EA.assign(ne, 0.0);
    m_EI.assign(ne, 0.0);
    m_GJ.assign(ne, 0.0);
    const Real area = static_cast<Real>(kPi) * m_mechanicalCoreRadiusM * m_mechanicalCoreRadiusM;
    const Real I = static_cast<Real>(0.25 * kPi) * std::pow(m_mechanicalCoreRadiusM, 4);
    const Real J = static_cast<Real>(0.5 * kPi) * std::pow(m_mechanicalCoreRadiusM, 4);
    const std::size_t headEdges = std::min<std::size_t>(
        m_softTipEdgeCount > 0u ? m_softTipEdgeCount : m_magneticEdgeCount,
        ne);
    const std::size_t split = ne - headEdges;
    for (std::size_t i = 0; i < ne; ++i)
    {
        Real blend = static_cast<Real>(0.0);
        if (headEdges == 1u)
        {
            blend = i >= split ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
        }
        else if (headEdges > 1u && i >= split)
        {
            const Real alpha = std::clamp(
                static_cast<Real>(i - split + 1u) / static_cast<Real>(headEdges),
                static_cast<Real>(0.0),
                static_cast<Real>(1.0));
            // A real guidewire tip transitions into the soft magnetic segment
            // over a finite length. Using a smooth ramp instead of a stiffness
            // cliff reduces spurious wave reflection at the body-head interface.
            blend = static_cast<Real>(0.5) - static_cast<Real>(0.5) * std::cos(alpha * kPi);
        }
        const Real E = (static_cast<Real>(1.0) - blend) * m_youngBody + blend * m_youngHead;
        const Real G = (static_cast<Real>(1.0) - blend) * m_shearBody + blend * m_shearHead;
        m_EA[i] = E * area;
        m_EI[i] = E * I;
        m_GJ[i] = G * J;
    }
}

void ElasticRodCompatCore::computeMassDistribution()
{
    const std::size_t n = m_undeformedCentersM.size();
    m_lumpedMassKg.assign(n, 0.0);
    m_lumpedThetaInertiaKgM2.assign(n, 0.0);
    m_totalMassKg = static_cast<Real>(0.0);
    if (n < 2)
        return;

    const Real area = static_cast<Real>(kPi) * m_mechanicalCoreRadiusM * m_mechanicalCoreRadiusM;
    for (std::size_t e = 0; e < m_refLen.size(); ++e)
    {
        const Real segmentMass = m_rho * area * m_refLen[e];
        const Real nodalMass = static_cast<Real>(0.5) * segmentMass;
        m_lumpedMassKg[e] += nodalMass;
        m_lumpedMassKg[e + 1] += nodalMass;
        const Real thetaInertia = static_cast<Real>(0.5) * segmentMass * m_mechanicalCoreRadiusM * m_mechanicalCoreRadiusM;
        m_lumpedThetaInertiaKgM2[e] += thetaInertia;
        m_totalMassKg += segmentMass;
    }
}

void ElasticRodCompatCore::computeRestCurvature()
{
    const std::size_t n = m_undeformedCentersM.size();
    const std::size_t ne = m_refLen.size();
    m_kappaBar.assign(n, Vec2(0.0, 0.0));
    m_undeformedTwist.assign(ne, 0.0);
    if (n < 3 || ne < 2)
        return;

    for (std::size_t i = 1; i < ne; ++i)
    {
        const Vec3& t0 = m_restFrames[i - 1].m3;
        const Vec3& t1 = m_restFrames[i].m3;
        const Real denom = std::max(static_cast<Real>(1.0) + sofa::type::dot(t0, t1), static_cast<Real>(kMinCurvatureDenom));
        const Vec3 kb = static_cast<Real>(2.0) * t0.cross(t1) / denom;
        const Vec3 m1e = m_restFrames[i - 1].m1;
        const Vec3 m2e = m_restFrames[i - 1].m2;
        const Vec3 m1f = m_restFrames[i].m1;
        const Vec3 m2f = m_restFrames[i].m2;
        m_kappaBar[i][0] = static_cast<Real>(0.5) * sofa::type::dot(kb, m2e + m2f);
        m_kappaBar[i][1] = static_cast<Real>(-0.5) * sofa::type::dot(kb, m1e + m1f);

        const Vec3 transported = parallelTransport(m1e, t0, t1);
        m_undeformedTwist[i] = signedAngle(transported, m1f, t1);
    }
}

ElasticRodCompatCore::State ElasticRodCompatCore::computeState(const std::vector<Coord>& coords) const
{
    State state;
    const std::size_t n = coords.size();
    if (n == 0)
        return state;

    state.centersM.reserve(n);
    for (const Coord& c : coords)
        state.centersM.push_back(static_cast<Real>(1.0e-3) * coordCenter(c));
    if (n < 2)
        return state;

    const std::size_t ne = n - 1;
    state.theta.assign(ne, static_cast<Real>(0.0));
    state.edgeLenM.assign(ne, static_cast<Real>(0.0));
    state.tangents.assign(ne, kZAxis);
    state.referenceFrames.assign(ne, FrameAxes {});
    state.materialFrames.assign(ne, FrameAxes {});
    state.kb.assign(n, Vec3(0.0, 0.0, 0.0));
    state.kappa.assign(n, Vec2(0.0, 0.0));
    state.refTwist.assign(ne, static_cast<Real>(0.0));
    state.twist.assign(ne, static_cast<Real>(0.0));

    for (std::size_t i = 0; i < ne; ++i)
    {
        const Vec3 edge = state.centersM[i + 1] - state.centersM[i];
        state.edgeLenM[i] = edge.norm();
        const Vec3 fallback = i < m_committedTangents.size()
            ? m_committedTangents[i]
            : (i < m_restFrames.size() ? m_restFrames[i].m3 : kZAxis);
        state.tangents[i] = safeNormalize(edge, fallback);
        state.theta[i] = coordTheta(coords[i]);
    }

    const bool hasCommittedReference =
        m_committedReferenceFrames.size() == ne &&
        m_committedTangents.size() == ne;

    if (hasCommittedReference)
    {
        for (std::size_t i = 0; i < ne; ++i)
        {
            const Vec3 d1 = parallelTransport(m_committedReferenceFrames[i].m1, m_committedTangents[i], state.tangents[i]);
            state.referenceFrames[i] = buildFrameFromTangent(state.tangents[i], d1);
        }
    }
    else
    {
        Vec3 d1 = state.tangents[0].cross(Vec3(0.0, 0.0, -1.0));
        if (d1.norm() <= kEps)
            d1 = state.tangents[0].cross(kYAxis);
        d1 = safeNormalize(d1, kXAxis);
        state.referenceFrames[0] = buildFrameFromTangent(state.tangents[0], d1);
        for (std::size_t i = 1; i < ne; ++i)
        {
            d1 = parallelTransport(state.referenceFrames[i - 1].m1, state.tangents[i - 1], state.tangents[i]);
            state.referenceFrames[i] = buildFrameFromTangent(state.tangents[i], d1);
        }
    }

    for (std::size_t i = 0; i < ne; ++i)
        state.materialFrames[i] = rotateFrameAboutAxis(state.referenceFrames[i], state.theta[i]);

    for (std::size_t i = 1; i < ne; ++i)
    {
        Vec3 ut = parallelTransport(state.referenceFrames[i - 1].m1, state.tangents[i - 1], state.tangents[i]);
        const Real oldRefTwist = i < m_committedRefTwist.size() ? m_committedRefTwist[i] : static_cast<Real>(0.0);
        rotateAxisAngle(ut, state.tangents[i], oldRefTwist);
        state.refTwist[i] = oldRefTwist + signedAngle(ut, state.referenceFrames[i].m1, state.tangents[i]);
        state.twist[i] = state.theta[i] - state.theta[i - 1] + state.refTwist[i];
    }

    for (std::size_t i = 1; i < ne; ++i)
    {
        const Vec3& t0 = state.tangents[i - 1];
        const Vec3& t1 = state.tangents[i];
        const Real denom = std::max(static_cast<Real>(1.0) + sofa::type::dot(t0, t1), static_cast<Real>(kMinCurvatureDenom));
        state.kb[i] = static_cast<Real>(2.0) * t0.cross(t1) / denom;

        const Vec3 m1e = state.materialFrames[i - 1].m1;
        const Vec3 m2e = state.materialFrames[i - 1].m2;
        const Vec3 m1f = state.materialFrames[i].m1;
        const Vec3 m2f = state.materialFrames[i].m2;
        state.kappa[i][0] = static_cast<Real>(0.5) * sofa::type::dot(state.kb[i], m2e + m2f);
        state.kappa[i][1] = static_cast<Real>(-0.5) * sofa::type::dot(state.kb[i], m1e + m1f);
    }

    return state;
}

void ElasticRodCompatCore::commitState(const State& state)
{
    m_committedReferenceFrames = state.referenceFrames;
    m_committedTangents = state.tangents;
    m_committedRefTwist = state.refTwist;
}

void ElasticRodCompatCore::computeBoundaryTargets(
    Real commandedInsertionMm,
    Real commandedTwistRad,
    const Vec3& insertionDirection,
    std::size_t pushNodeCount,
    std::vector<Vec3>& targetCentersMm,
    std::vector<Real>& targetTheta) const
{
    const Vec3 dir = safeNormalize(insertionDirection, kZAxis);
    const std::size_t count = std::min<std::size_t>(pushNodeCount, m_initialCentersMm.size());
    targetCentersMm.resize(count);
    targetTheta.resize(count, static_cast<Real>(0.0));
    const Vec3 base = count > 0 ? m_initialCentersMm.front() : Vec3(0.0, 0.0, 0.0);
    for (std::size_t i = 0; i < count; ++i)
    {
        targetCentersMm[i] = base + (commandedInsertionMm + supportArcLengthMm(i)) * dir;
        targetTheta[i] = (i == 0) ? commandedTwistRad : static_cast<Real>(0.0);
    }
}

ElasticRodCompatCore::Real ElasticRodCompatCore::supportArcLengthMm(std::size_t nodeIndex) const
{
    if (nodeIndex == 0 || m_refLen.empty())
        return static_cast<Real>(0.0);

    const std::size_t edgeCount = std::min<std::size_t>(nodeIndex, m_refLen.size());
    Real arcLenM = static_cast<Real>(0.0);
    for (std::size_t e = 0; e < edgeCount; ++e)
        arcLenM += m_refLen[e];
    return static_cast<Real>(1.0e3) * arcLenM;
}

std::size_t ElasticRodCompatCore::magneticStartEdge() const
{
    const std::size_t ne = m_refLen.size();
    const std::size_t mag = std::min(m_magneticEdgeCount, ne);
    return ne - mag;
}

} // namespace elastic_rod_guidewire
