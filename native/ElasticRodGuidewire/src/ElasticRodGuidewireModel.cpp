#include <ElasticRodGuidewire/ElasticRodGuidewireModel.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace elastic_rod_guidewire
{

namespace
{
using Model = ElasticRodGuidewireModel;
using Real = Model::Real;
using Coord = Model::Coord;
using Deriv = Model::Deriv;
using VecCoord = Model::VecCoord;
using VecDeriv = Model::VecDeriv;
using Vec3 = Model::Vec3;
using VecReal = Model::VecReal;
using VecVec3 = Model::VecVec3;
using Vec2 = ElasticRodCompatCore::Vec2;

constexpr Real kEps = static_cast<Real>(1.0e-12);
constexpr Real kMmToM = static_cast<Real>(1.0e-3);
constexpr Real kTranslationEpsM = static_cast<Real>(2.5e-6);
constexpr Real kThetaEpsRad = static_cast<Real>(5.0e-4);
constexpr Real kAbnormalStretchRatio = static_cast<Real>(0.50);
constexpr Real kMToMm = static_cast<Real>(1.0e3);
constexpr Real kMinCurvatureDenom = static_cast<Real>(5.0e-3);
constexpr Real kPi = static_cast<Real>(3.14159265358979323846);
const Vec3 kZAxis(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(1.0));

inline void zeroDeriv(Deriv& d)
{
    d.clear();
}

inline void addCenterScene(Deriv& d, const Vec3& forceScene)
{
    d[0] += forceScene[0];
    d[1] += forceScene[1];
    d[2] += forceScene[2];
}

inline void addTheta(Deriv& d, Real torqueNm)
{
    d[3] += torqueNm;
}

inline void addLocalDof(Deriv& d, unsigned int localDof, Real value)
{
    if (localDof < 3u)
        d[localDof] += value;
    else if (localDof == 3u)
        d[3] += value;
}

inline Real derivComponent(const Deriv& d, unsigned int localDof)
{
    if (localDof < 3u)
        return d[localDof];
    if (localDof == 3u)
        return d[3];
    return static_cast<Real>(0.0);
}

inline Vec3 projectorAxial(const Vec3& axis, const Vec3& value)
{
    return sofa::type::dot(value, axis) * axis;
}

inline Vec3 projectorLateral(const Vec3& axis, const Vec3& value)
{
    return value - projectorAxial(axis, value);
}

inline Vec3 safeNormalize(const Vec3& v, const Vec3& fallback)
{
    return ElasticRodCompatCore::safeNormalize(v, fallback);
}

inline Real averageCoeff(const std::vector<Real>& values, std::size_t i)
{
    if (values.empty())
        return static_cast<Real>(0.0);
    if (i == 0)
        return values.front();
    if (i >= values.size())
        return values.back();
    return static_cast<Real>(0.5) * (values[i - 1] + values[i]);
}

inline Real translationalForceN(const Deriv& d)
{
    return derivCenter(d).norm();
}

inline Vec3 solverToScene(const Vec3& pM)
{
    return kMToMm * pM;
}

inline Vec3 sceneToSolver(const Vec3& pMm)
{
    return kMmToM * pMm;
}

inline Real smoothstep01(Real x)
{
    const Real u = std::clamp(x, static_cast<Real>(0.0), static_cast<Real>(1.0));
    return u * u * (static_cast<Real>(3.0) - static_cast<Real>(2.0) * u);
}

inline Real smoothstepRange(Real x, Real edge0, Real edge1)
{
    if (edge1 <= edge0 + kEps)
        return x >= edge1 ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
    return smoothstep01((x - edge0) / (edge1 - edge0));
}

inline Real strictBarrierVelocityGate(Real clearanceMm, Real activationMarginMm, Real safetyMarginMm)
{
    if (!std::isfinite(clearanceMm))
        return static_cast<Real>(0.0);
    const Real activationBandMm = std::max(activationMarginMm - safetyMarginMm, static_cast<Real>(0.0));
    if (activationBandMm <= kEps)
        return clearanceMm < activationMarginMm ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
    return smoothstepRange(
        activationMarginMm - clearanceMm,
        static_cast<Real>(0.0),
        activationBandMm);
}

inline Real strictBarrierPenetrationDepthM(Real clearanceMm, Real safetyMarginMm)
{
    if (!std::isfinite(clearanceMm))
        return static_cast<Real>(0.0);
    return kMmToM * std::max(safetyMarginMm - clearanceMm, static_cast<Real>(0.0));
}

inline void rotateAxisAngle(Vec3& v, const Vec3& axis, Real theta)
{
    if (std::abs(theta) <= kEps)
        return;

    const Real cs = std::cos(theta);
    const Real ss = std::sin(theta);
    v = cs * v + ss * axis.cross(v) + sofa::type::dot(axis, v) * (static_cast<Real>(1.0) - cs) * axis;
}

inline Vec3 closestPointOnSegment(const Vec3& p, const Vec3& a, const Vec3& b, Real& u)
{
    const Vec3 ab = b - a;
    const Real ab2 = sofa::type::dot(ab, ab);
    u = ab2 <= kEps ? static_cast<Real>(0.0) : std::clamp(sofa::type::dot(p - a, ab) / ab2, static_cast<Real>(0.0), static_cast<Real>(1.0));
    return a + ab * u;
}

bool solveSymmetric3x3(const double a[3][3], const Vec3& rhs, Vec3& x)
{
    double m[3][4] = {
        {a[0][0], a[0][1], a[0][2], rhs[0]},
        {a[1][0], a[1][1], a[1][2], rhs[1]},
        {a[2][0], a[2][1], a[2][2], rhs[2]},
    };

    for (int col = 0; col < 3; ++col)
    {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row)
        {
            if (std::abs(m[row][col]) > std::abs(m[pivot][col]))
                pivot = row;
        }
        if (std::abs(m[pivot][col]) <= 1.0e-12)
            return false;
        if (pivot != col)
        {
            for (int k = col; k < 4; ++k)
                std::swap(m[col][k], m[pivot][k]);
        }

        const double diag = m[col][col];
        for (int k = col; k < 4; ++k)
            m[col][k] /= diag;

        for (int row = 0; row < 3; ++row)
        {
            if (row == col)
                continue;
            const double factor = m[row][col];
            for (int k = col; k < 4; ++k)
                m[row][k] -= factor * m[col][k];
        }
    }

    x = Vec3(m[0][3], m[1][3], m[2][3]);
    return true;
}

struct TubeConstraintSample
{
    bool valid {false};
    Vec3 projectionMm {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
    Vec3 tangentMm {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(1.0)};
    Real radiusMm {static_cast<Real>(0.0)};
    Real clearanceMm {std::numeric_limits<Real>::infinity()};
    Vec3 outwardNormalM {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
};

bool projectPointToTube(
    const VecVec3& tubeNodes,
    const VecReal& tubeRadiiMm,
    const Vec3& pointM,
    Real rodRadiusMm,
    Real safetyMarginMm,
    TubeConstraintSample& sample)
{
    sample = TubeConstraintSample {};
    if (tubeNodes.size() < 2 || tubeRadiiMm.size() != tubeNodes.size())
        return false;

    const Vec3 pointMm = solverToScene(pointM);
    Real bestD2 = std::numeric_limits<Real>::max();
    for (std::size_t i = 0; i + 1 < tubeNodes.size(); ++i)
    {
        Real u = static_cast<Real>(0.0);
        const Vec3 proj = closestPointOnSegment(pointMm, tubeNodes[i], tubeNodes[i + 1], u);
        const Vec3 deltaMm = pointMm - proj;
        const Real d2 = sofa::type::dot(deltaMm, deltaMm);
        if (d2 >= bestD2)
            continue;

        bestD2 = d2;
        sample.valid = true;
        sample.projectionMm = proj;
        sample.tangentMm = ElasticRodCompatCore::safeNormalize(tubeNodes[i + 1] - tubeNodes[i], kZAxis);
        sample.radiusMm = static_cast<Real>(1.0 - u) * tubeRadiiMm[i] + u * tubeRadiiMm[i + 1];
    }

    if (!sample.valid)
        return false;

    const Vec3 radialMm = pointMm - sample.projectionMm;
    const Real radialNormMm = radialMm.norm();
    sample.clearanceMm = sample.radiusMm - rodRadiusMm - safetyMarginMm - radialNormMm;
    if (radialNormMm > kEps)
        sample.outwardNormalM = safeNormalize(sceneToSolver(radialMm), kZAxis);
    else
        sample.outwardNormalM = Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    return true;
}

}

ElasticRodGuidewireModel::ElasticRodGuidewireModel()
    : d_initialNodes(initData(&d_initialNodes, "initialNodes", "Optional explicit initial guidewire node positions in scene units (mm)."))
    , d_undeformedNodes(initData(&d_undeformedNodes, "undeformedNodes", "Optional undeformed guidewire nodes in scene units (mm)."))
    , d_rho(initData(&d_rho, static_cast<Real>(6500.0), "rho", "Mass density in kg/m^3."))
    , d_rodRadius(initData(&d_rodRadius, static_cast<Real>(0.35), "rodRadius", "Guidewire contact outer radius in mm."))
    , d_mechanicalCoreRadiusMm(initData(&d_mechanicalCoreRadiusMm, static_cast<Real>(0.20), "mechanicalCoreRadiusMm", "Mechanical core radius used for EA/EI/GJ and inertial terms in mm."))
    , d_dt(initData(&d_dt, static_cast<Real>(1.0e-4), "dt", "Time step in seconds."))
    , d_youngHead(initData(&d_youngHead, static_cast<Real>(1.8e10), "youngHead", "Effective distal soft-tip Young modulus in Pa."))
    , d_youngBody(initData(&d_youngBody, static_cast<Real>(5.5e10), "youngBody", "Body Young modulus in Pa."))
    , d_shearHead(initData(&d_shearHead, static_cast<Real>(6.766917293233083e9), "shearHead", "Effective distal soft-tip shear modulus in Pa."))
    , d_shearBody(initData(&d_shearBody, static_cast<Real>(2.067669172932331e10), "shearBody", "Body shear modulus in Pa."))
    , d_rodLength(initData(&d_rodLength, static_cast<Real>(400.0), "rodLength", "Guidewire length in mm."))
    , d_magneticEdgeCount(initData(&d_magneticEdgeCount, 5u, "magneticEdgeCount", "Number of distal magnetic edges."))
    , d_softTipEdgeCount(initData(&d_softTipEdgeCount, 8u, "softTipEdgeCount", "Number of distal edges that use the softer segmented tip stiffness."))
    , d_pushNodeCount(initData(&d_pushNodeCount, 2u, "pushNodeCount", "Number of proximal support-block nodes driven together inside the introducer."))
    , d_axialDriveNodeCount(initData(&d_axialDriveNodeCount, 2u, "axialDriveNodeCount", "Number of proximal nodes that receive axial insertion driving; remaining support nodes only receive lateral sheath support."))
    , d_useDynamicSupportWindows(initData(&d_useDynamicSupportWindows, false, "useDynamicSupportWindows", "Enable strict dynamic support/drive windows instead of legacy prefix support counts."))
    , d_supportNodeIndices(initData(&d_supportNodeIndices, "supportNodeIndices", "Optional dynamic support-window candidate node indices for strict elasticrod mode."))
    , d_driveNodeIndices(initData(&d_driveNodeIndices, "driveNodeIndices", "Optional dynamic drive-window candidate node indices for strict elasticrod mode."))
    , d_supportWindowLengthMm(initData(&d_supportWindowLengthMm, static_cast<Real>(30.0), "supportWindowLengthMm", "Length of the strict native support window behind the entry in mm."))
    , d_supportReleaseDistanceMm(initData(&d_supportReleaseDistanceMm, static_cast<Real>(6.0), "supportReleaseDistanceMm", "Cosine release distance inside the external sheath before the vessel entry in mm for strict native support."))
    , d_driveWindowLengthMm(initData(&d_driveWindowLengthMm, static_cast<Real>(12.0), "driveWindowLengthMm", "Length of the strict native drive window outside the vessel entry in mm."))
    , d_driveWindowOutsideOffsetMm(initData(&d_driveWindowOutsideOffsetMm, static_cast<Real>(0.5), "driveWindowOutsideOffsetMm", "Outer offset of the strict native drive window from the vessel entry in mm."))
    , d_driveWindowMinNodeCount(initData(&d_driveWindowMinNodeCount, 5u, "driveWindowMinNodeCount", "Minimum number of nodes retained in the strict native drive window."))
    , d_commandedInsertion(initData(&d_commandedInsertion, static_cast<Real>(0.0), "commandedInsertion", "Commanded proximal insertion in mm."))
    , d_commandedTwist(initData(&d_commandedTwist, static_cast<Real>(0.0), "commandedTwist", "Commanded proximal twist in radians."))
    , d_insertionDirection(initData(&d_insertionDirection, Vec3(0.0, 0.0, 1.0), "insertionDirection", "Insertion axis direction in world coordinates."))
    , d_proximalAxialStiffness(initData(&d_proximalAxialStiffness, static_cast<Real>(600.0), "proximalAxialStiffness", "Proximal axial boundary stiffness in N/m."))
    , d_proximalLateralStiffness(initData(&d_proximalLateralStiffness, static_cast<Real>(2500.0), "proximalLateralStiffness", "Proximal lateral boundary stiffness in N/m."))
    , d_proximalAngularStiffness(initData(&d_proximalAngularStiffness, static_cast<Real>(1.0e-3), "proximalAngularStiffness", "Proximal twist boundary stiffness in N.m/rad."))
    , d_proximalLinearDamping(initData(&d_proximalLinearDamping, static_cast<Real>(0.0), "proximalLinearDamping", "Proximal translational damping in N.s/m."))
    , d_proximalAngularDamping(initData(&d_proximalAngularDamping, static_cast<Real>(0.0), "proximalAngularDamping", "Proximal twist damping in N.m.s/rad."))
    , d_edgeAxialDamping(initData(&d_edgeAxialDamping, static_cast<Real>(12.0), "edgeAxialDamping", "Kelvin-Voigt axial damping applied to each active rod edge in N.s/m."))
    , d_axialStretchStiffnessScale(initData(&d_axialStretchStiffnessScale, static_cast<Real>(1.0), "axialStretchStiffnessScale", "Multiplier applied only to axial EA to make the rod closer to inextensible without changing bend/twist stiffness."))
    , d_axialStretchUseBodyFloor(initData(&d_axialStretchUseBodyFloor, false, "axialStretchUseBodyFloor", "If true, axial EA on magnetic-head edges is clamped to at least the body Young modulus before the axial stiffness scale is applied."))
    , d_useImplicitStretch(initData(&d_useImplicitStretch, true, "useImplicitStretch", "Whether axial stretch contributes an implicit tangent. Enabled by default to preserve the previous strict-path behavior while keeping a dedicated switch for stretch-Jacobian experiments."))
    , d_useImplicitBendTwist(initData(&d_useImplicitBendTwist, false, "useImplicitBendTwist", "Whether bend/twist contributes an implicit tangent. Disabled by default because the current reduced Hessian remains numerically fragile under strict dynamics."))
    , d_useKinematicSupportBlock(initData(&d_useKinematicSupportBlock, false, "useKinematicSupportBlock", "If true, the introducer support block is treated as a prescribed boundary segment instead of a free dynamic rod segment. Disabled by default because hard support projection injects a non-physical interface discontinuity at the distal end of the support block."))
    , d_commitReferenceStateEachStep(initData(&d_commitReferenceStateEachStep, false, "commitReferenceStateEachStep", "If true, update the rod reference transport state from the current configuration at every animation step. Disabled by default in strict mode because repeated reference commits amplify pre-contact drift."))
    , d_debugRefLen(initData(&d_debugRefLen, "debugRefLen", "Reference edge lengths in mm."))
    , d_debugVoronoiLen(initData(&d_debugVoronoiLen, "debugVoronoiLen", "Voronoi lengths in mm."))
    , d_debugEA(initData(&d_debugEA, "debugEA", "Stretch stiffness EA in SI units."))
    , d_debugEI(initData(&d_debugEI, "debugEI", "Bending stiffness EI in SI units."))
    , d_debugGJ(initData(&d_debugGJ, "debugGJ", "Twist stiffness GJ in SI units."))
    , d_debugEdgeLengthMm(initData(&d_debugEdgeLengthMm, "debugEdgeLengthMm", "Current per-edge lengths in mm."))
    , d_debugStretch(initData(&d_debugStretch, "debugStretch", "Per-edge axial strain."))
    , d_debugKappa(initData(&d_debugKappa, "debugKappa", "Per-node curvature vector [kappa1, kappa2, 0]."))
    , d_debugTwist(initData(&d_debugTwist, "debugTwist", "Per-edge twist error relative to the undeformed rod in radians."))
    , d_debugTipProgress(initData(&d_debugTipProgress, static_cast<Real>(0.0), "debugTipProgress", "Current commanded insertion in mm."))
    , d_debugTotalMass(initData(&d_debugTotalMass, static_cast<Real>(0.0), "debugTotalMass", "Total rod mass in kg."))
    , d_debugAbnormalEdgeIndex(initData(&d_debugAbnormalEdgeIndex, -1, "debugAbnormalEdgeIndex", "First edge index whose stretch entered a non-physical range; -1 means no anomaly."))
    , d_debugAbnormalEdgeLengthMm(initData(&d_debugAbnormalEdgeLengthMm, static_cast<Real>(0.0), "debugAbnormalEdgeLengthMm", "Current length of the first abnormal edge in mm."))
    , d_debugAbnormalEdgeRefLengthMm(initData(&d_debugAbnormalEdgeRefLengthMm, static_cast<Real>(0.0), "debugAbnormalEdgeRefLengthMm", "Reference length of the first abnormal edge in mm."))
    , d_debugMaxAxialBoundaryErrorMm(initData(&d_debugMaxAxialBoundaryErrorMm, static_cast<Real>(0.0), "debugMaxAxialBoundaryErrorMm", "Maximum proximal axial boundary error magnitude in mm."))
    , d_debugMaxLateralBoundaryErrorMm(initData(&d_debugMaxLateralBoundaryErrorMm, static_cast<Real>(0.0), "debugMaxLateralBoundaryErrorMm", "Maximum proximal lateral boundary error magnitude in mm."))
    , d_debugMaxInternalForceN(initData(&d_debugMaxInternalForceN, static_cast<Real>(0.0), "debugMaxInternalForceN", "Maximum internal rod force magnitude in N before proximal boundary driving is added."))
    , d_debugMaxStretchForceN(initData(&d_debugMaxStretchForceN, static_cast<Real>(0.0), "debugMaxStretchForceN", "Maximum per-edge axial stretch force magnitude in N for the current configuration."))
    , d_debugMaxBoundaryForceN(initData(&d_debugMaxBoundaryForceN, static_cast<Real>(0.0), "debugMaxBoundaryForceN", "Maximum proximal boundary translational force magnitude in N for the current configuration."))
    , d_debugMaxBoundaryTorqueNm(initData(&d_debugMaxBoundaryTorqueNm, static_cast<Real>(0.0), "debugMaxBoundaryTorqueNm", "Maximum proximal boundary torque magnitude in N.m for the current configuration."))
    , d_debugDriveReactionN(initData(&d_debugDriveReactionN, static_cast<Real>(0.0), "debugDriveReactionN", "Estimated total axial reaction carried by the proximal insertion driver in N."))
    , d_debugMaxBendResidual(initData(&d_debugMaxBendResidual, static_cast<Real>(0.0), "debugMaxBendResidual", "Maximum local bend/twist residual norm used by the reduced rod Gauss-Newton block."))
    , d_tubeNodes(initData(&d_tubeNodes, "tubeNodes", "Centerline points used by the strict lumen barrier in scene units (mm)."))
    , d_tubeRadiiMm(initData(&d_tubeRadiiMm, "tubeRadiiMm", "Per-node lumen radius in mm matching tubeNodes."))
    , d_nodeInitialPathSmm(initData(&d_nodeInitialPathSmm, "nodeInitialPathSmm", "Initial per-node path coordinate in mm relative to the vessel entry; negative values remain outside the lumen until commanded insertion advances them past the entry."))
    , d_strictLumenBarrierEnabled(initData(&d_strictLumenBarrierEnabled, false, "strictLumenBarrierEnabled", "Enable the strict lumen barrier that keeps nodes inside the profile."))
    , d_strictLumenActivationMarginMm(initData(&d_strictLumenActivationMarginMm, static_cast<Real>(0.25), "strictLumenActivationMarginMm", "Margin (mm) ahead of the wall where the barrier begins to push."))
    , d_strictLumenSafetyMarginMm(initData(&d_strictLumenSafetyMarginMm, static_cast<Real>(0.05), "strictLumenSafetyMarginMm", "Safety margin (mm) kept inside the wall before barrier fully ramps up."))
    , d_strictLumenBarrierStiffness(initData(&d_strictLumenBarrierStiffness, static_cast<Real>(5000.0), "strictLumenBarrierStiffness", "Radial stiffness (N/m) of the strict lumen barrier."))
    , d_strictLumenBarrierDamping(initData(&d_strictLumenBarrierDamping, static_cast<Real>(0.20), "strictLumenBarrierDamping", "Radial damping (N.s/m) of the strict lumen barrier."))
    , d_strictLumenBarrierMaxForcePerNodeN(initData(&d_strictLumenBarrierMaxForcePerNodeN, static_cast<Real>(0.35), "strictLumenBarrierMaxForcePerNodeN", "Per-node cap (N) for the strict barrier force."))
    , d_strictLumenEntryExtensionMm(initData(&d_strictLumenEntryExtensionMm, static_cast<Real>(0.0), "strictLumenEntryExtensionMm", "Outside-entry extension (mm) where the strict lumen barrier still protects the hidden external support corridor."))
    , d_strictLumenEntrySupportRadiusMm(initData(&d_strictLumenEntrySupportRadiusMm, static_cast<Real>(0.0), "strictLumenEntrySupportRadiusMm", "Radius (mm) of the hidden strict external support corridor used just outside the vessel entry."))
    , d_debugMinLumenClearanceMm(initData(&d_debugMinLumenClearanceMm, std::numeric_limits<Real>::infinity(), "debugMinLumenClearanceMm", "Minimum lumen clearance among all nodes (mm)."))
    , d_debugBarrierForceVector(initData(&d_debugBarrierForceVector, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)), "debugBarrierForceVector", "Aggregated strict lumen barrier force in N (scene units)."))
    , d_debugBarrierActiveNodeCount(initData(&d_debugBarrierActiveNodeCount, 0u, "debugBarrierActiveNodeCount", "Number of nodes currently influenced by the strict lumen barrier."))
    , d_debugMaxHeadStretch(initData(&d_debugMaxHeadStretch, static_cast<Real>(0.0), "debugMaxHeadStretch", "Maximum stretch magnitude on the magnetic head edges."))
{
    this->f_listening.setValue(true);
}

void ElasticRodGuidewireModel::refreshLumenProfile()
{
    m_tubeNodesCached.clear();
    m_tubeRadiiCachedMm.clear();
    m_tubeCum.clear();
    const auto& nodes = d_tubeNodes.getValue();
    const auto& radii = d_tubeRadiiMm.getValue();
    if (nodes.size() < 2 || nodes.size() != radii.size())
    {
        m_haveLumenProfile = false;
        return;
    }

    m_tubeNodesCached.reserve(nodes.size());
    m_tubeRadiiCachedMm.reserve(radii.size());
    m_tubeCum.assign(nodes.size(), static_cast<Real>(0.0));
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        m_tubeNodesCached.push_back(nodes[i]);
        m_tubeRadiiCachedMm.push_back(static_cast<Real>(radii[i]));
        if (i > 0)
            m_tubeCum[i] = m_tubeCum[i - 1] + (m_tubeNodesCached[i] - m_tubeNodesCached[i - 1]).norm();
    }
    m_haveLumenProfile = true;
}

bool ElasticRodGuidewireModel::hasLumenProfile() const
{
    return m_haveLumenProfile && m_tubeCum.size() > 1;
}

Real ElasticRodGuidewireModel::tubeRadiusMm(Real s) const
{
    if (!hasLumenProfile())
        return static_cast<Real>(0.0);
    if (s <= m_tubeCum.front())
        return m_tubeRadiiCachedMm.front();
    if (s >= m_tubeCum.back())
        return m_tubeRadiiCachedMm.back();
    const auto it = std::upper_bound(m_tubeCum.begin(), m_tubeCum.end(), s);
    if (it == m_tubeCum.begin())
        return m_tubeRadiiCachedMm.front();
    const std::size_t idx = static_cast<std::size_t>(std::distance(m_tubeCum.begin(), it));
    const Real denom = std::max(m_tubeCum[idx] - m_tubeCum[idx - 1], kEps);
    const Real alpha = (s - m_tubeCum[idx - 1]) / denom;
    return (static_cast<Real>(1.0) - alpha) * m_tubeRadiiCachedMm[idx - 1] + alpha * m_tubeRadiiCachedMm[idx];
}

bool ElasticRodGuidewireModel::tubePointAtS(Real s, Vec3& pointMm, Real& radiusMm) const
{
    if (!hasLumenProfile())
        return false;
    if (s <= m_tubeCum.front())
    {
        pointMm = m_tubeNodesCached.front();
        radiusMm = m_tubeRadiiCachedMm.front();
        return true;
    }
    if (s >= m_tubeCum.back())
    {
        pointMm = m_tubeNodesCached.back();
        radiusMm = m_tubeRadiiCachedMm.back();
        return true;
    }
    const auto it = std::upper_bound(m_tubeCum.begin(), m_tubeCum.end(), s);
    if (it == m_tubeCum.begin())
    {
        pointMm = m_tubeNodesCached.front();
        radiusMm = m_tubeRadiiCachedMm.front();
        return true;
    }
    const std::size_t idx = static_cast<std::size_t>(std::distance(m_tubeCum.begin(), it));
    const Real denom = std::max(m_tubeCum[idx] - m_tubeCum[idx - 1], kEps);
    const Real alpha = (s - m_tubeCum[idx - 1]) / denom;
    pointMm = (static_cast<Real>(1.0) - alpha) * m_tubeNodesCached[idx - 1] + alpha * m_tubeNodesCached[idx];
    radiusMm = (static_cast<Real>(1.0) - alpha) * m_tubeRadiiCachedMm[idx - 1] + alpha * m_tubeRadiiCachedMm[idx];
    return true;
}

bool ElasticRodGuidewireModel::projectToTube(const Vec3& point, Vec3& closestPoint, Real& outS) const
{
    if (!hasLumenProfile())
        return false;

    const std::size_t segmentCount = m_tubeNodesCached.size() - 1;
    Real bestDist2 = std::numeric_limits<Real>::infinity();
    for (std::size_t i = 0; i < segmentCount; ++i)
    {
        Real u;
        const Vec3 proj = closestPointOnSegment(point, m_tubeNodesCached[i], m_tubeNodesCached[i + 1], u);
        const Vec3 delta = point - proj;
        const Real dist2 = sofa::type::dot(delta, delta);
        if (dist2 < bestDist2)
        {
            bestDist2 = dist2;
            closestPoint = proj;
            outS = m_tubeCum[i] + (m_tubeCum[i + 1] - m_tubeCum[i]) * u;
        }
    }

    return true;
}

bool ElasticRodGuidewireModel::sampleStrictLumenCandidateAtS(
    const Vec3& pointMm,
    Real s,
    Real safetyMarginMm,
    Vec3& projectionMm,
    Real& clearanceMm,
    Vec3& outwardNormalM) const
{
    if (!hasLumenProfile())
        return false;

    const Real sClamped = std::clamp(s, m_tubeCum.front(), m_tubeCum.back());
    Real radiusMm = static_cast<Real>(0.0);
    if (!tubePointAtS(sClamped, projectionMm, radiusMm))
        return false;

    const Vec3 radialMm = pointMm - projectionMm;
    const Real radialNormMm = radialMm.norm();
    clearanceMm = radiusMm - d_rodRadius.getValue() - safetyMarginMm - radialNormMm;
    outwardNormalM = radialNormMm > kEps
        ? safeNormalize(sceneToSolver(radialMm), kZAxis)
        : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    return true;
}

bool ElasticRodGuidewireModel::sampleStrictEntrySupportConstraint(
    const Vec3& pointMm,
    Real safetyMarginMm,
    Vec3& projectionMm,
    Real& projS,
    Real& clearanceMm,
    Vec3& outwardNormalM) const
{
    if (m_tubeNodesCached.empty())
        return false;

    const Real entryExtensionMm = std::max(d_strictLumenEntryExtensionMm.getValue(), static_cast<Real>(0.0));
    const Real supportRadiusMm = std::max(d_strictLumenEntrySupportRadiusMm.getValue(), static_cast<Real>(0.0));
    if (entryExtensionMm <= kEps || supportRadiusMm <= kEps)
        return false;

    const Vec3 axis = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    const Vec3 entryPointMm = m_tubeNodesCached.front();
    const Real axialMm = sofa::type::dot(pointMm - entryPointMm, axis);
    if (axialMm >= static_cast<Real>(0.0) || axialMm < -entryExtensionMm)
        return false;

    projectionMm = entryPointMm + axialMm * axis;
    projS = static_cast<Real>(0.0);
    const Vec3 radialMm = pointMm - projectionMm;
    const Real radialNormMm = radialMm.norm();
    clearanceMm = supportRadiusMm - d_rodRadius.getValue() - safetyMarginMm - radialNormMm;
    outwardNormalM = radialNormMm > kEps
        ? safeNormalize(sceneToSolver(radialMm), kZAxis)
        : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    return true;
}

bool ElasticRodGuidewireModel::sampleStrictLumenConstraint(
    std::size_t nodeIndex,
    const Vec3& pointMm,
    Real safetyMarginMm,
    Vec3& projectionMm,
    Real& projS,
    Real& clearanceMm,
    Vec3& outwardNormalM) const
{
    if (sampleStrictEntrySupportConstraint(pointMm, safetyMarginMm, projectionMm, projS, clearanceMm, outwardNormalM))
        return true;
    if (!hasLumenProfile())
        return false;

    bool haveCandidate = false;
    const Real localSearchWindowMm = std::max(static_cast<Real>(2.0), static_cast<Real>(6.0) * d_rodRadius.getValue());
    auto updateCandidate = [&](Real candidateS) -> void
    {
        Vec3 candidateProjectionMm(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        Real candidateClearanceMm = std::numeric_limits<Real>::infinity();
        Vec3 candidateOutwardNormalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (!sampleStrictLumenCandidateAtS(pointMm, candidateS, safetyMarginMm, candidateProjectionMm, candidateClearanceMm, candidateOutwardNormalM))
            return;
        if (!haveCandidate || candidateClearanceMm < clearanceMm)
        {
            haveCandidate = true;
            projectionMm = candidateProjectionMm;
            projS = std::clamp(candidateS, m_tubeCum.front(), m_tubeCum.back());
            clearanceMm = candidateClearanceMm;
            outwardNormalM = candidateOutwardNormalM;
        }
    };
    auto updateLocalNeighborhood = [&](Real centerS) -> void
    {
        updateCandidate(centerS - localSearchWindowMm);
        updateCandidate(centerS + localSearchWindowMm);
    };

    const auto& nodeInitialPathSmm = d_nodeInitialPathSmm.getValue();
    Real nominalS = static_cast<Real>(0.0);
    bool haveNominal = false;
    if (hasBoundaryDriver() && nodeIndex < nodeInitialPathSmm.size())
    {
        nominalS = nodeInitialPathSmm[nodeIndex] + d_commandedInsertion.getValue();
        if (nominalS < static_cast<Real>(0.0))
            return false;
        haveNominal = true;
        updateCandidate(nominalS);
        updateLocalNeighborhood(nominalS);
    }

    Vec3 nearestProjectionMm(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Real nearestS = static_cast<Real>(0.0);
    if (projectToTube(pointMm, nearestProjectionMm, nearestS))
    {
        updateCandidate(nearestS);
        updateLocalNeighborhood(nearestS);
        if (haveNominal)
            updateCandidate(static_cast<Real>(0.5) * (nominalS + nearestS));
    }

    return haveCandidate;
}

bool ElasticRodGuidewireModel::sampleStrictLumenConstraintForEdge(
    std::size_t edgeIndex,
    const Vec3& pointMm,
    Real safetyMarginMm,
    Vec3& projectionMm,
    Real& projS,
    Real& clearanceMm,
    Vec3& outwardNormalM) const
{
    if (sampleStrictEntrySupportConstraint(pointMm, safetyMarginMm, projectionMm, projS, clearanceMm, outwardNormalM))
        return true;
    if (!hasLumenProfile())
        return false;

    bool haveCandidate = false;
    const Real localSearchWindowMm = std::max(static_cast<Real>(2.0), static_cast<Real>(6.0) * d_rodRadius.getValue());
    auto updateCandidate = [&](Real candidateS) -> void
    {
        Vec3 candidateProjectionMm(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        Real candidateClearanceMm = std::numeric_limits<Real>::infinity();
        Vec3 candidateOutwardNormalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (!sampleStrictLumenCandidateAtS(pointMm, candidateS, safetyMarginMm, candidateProjectionMm, candidateClearanceMm, candidateOutwardNormalM))
            return;
        if (!haveCandidate || candidateClearanceMm < clearanceMm)
        {
            haveCandidate = true;
            projectionMm = candidateProjectionMm;
            projS = std::clamp(candidateS, m_tubeCum.front(), m_tubeCum.back());
            clearanceMm = candidateClearanceMm;
            outwardNormalM = candidateOutwardNormalM;
        }
    };
    auto updateLocalNeighborhood = [&](Real centerS) -> void
    {
        updateCandidate(centerS - localSearchWindowMm);
        updateCandidate(centerS + localSearchWindowMm);
    };

    const auto& nodeInitialPathSmm = d_nodeInitialPathSmm.getValue();
    Real nominalS = static_cast<Real>(0.0);
    bool haveNominal = false;
    if (hasBoundaryDriver() && edgeIndex + 1u < nodeInitialPathSmm.size())
    {
        const Real nominalS0 = nodeInitialPathSmm[edgeIndex] + d_commandedInsertion.getValue();
        const Real nominalS1 = nodeInitialPathSmm[edgeIndex + 1u] + d_commandedInsertion.getValue();
        if (nominalS0 < static_cast<Real>(0.0) || nominalS1 < static_cast<Real>(0.0))
            return false;
        nominalS = static_cast<Real>(0.5) * (nominalS0 + nominalS1);
        haveNominal = true;
        updateCandidate(nominalS);
        updateLocalNeighborhood(nominalS);
    }

    Vec3 nearestProjectionMm(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Real nearestS = static_cast<Real>(0.0);
    if (projectToTube(pointMm, nearestProjectionMm, nearestS))
    {
        updateCandidate(nearestS);
        updateLocalNeighborhood(nearestS);
        if (haveNominal)
            updateCandidate(static_cast<Real>(0.5) * (nominalS + nearestS));
    }

    return haveCandidate;
}

bool ElasticRodGuidewireModel::strictBarrierPointEligible(const Vec3& pointMm) const
{
    if (!d_strictLumenBarrierEnabled.getValue() || !hasLumenProfile())
        return false;
    if (hasBoundaryDriver())
        return true;
    if (m_tubeNodesCached.empty())
        return true;

    Vec3 projectionMm(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Real projS = static_cast<Real>(0.0);
    if (!projectToTube(pointMm, projectionMm, projS))
        return false;

    const Vec3 axis = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    const Vec3 entryPointMm = m_tubeNodesCached.front();
    const Real axialMm = sofa::type::dot(pointMm - entryPointMm, axis);
    const Real entryExtensionMm = std::max(d_strictLumenEntryExtensionMm.getValue(), static_cast<Real>(0.0));
    if (axialMm < static_cast<Real>(0.0))
        return entryExtensionMm > kEps && axialMm >= -entryExtensionMm;

    return projS >= static_cast<Real>(0.0);
}

Real ElasticRodGuidewireModel::strictBarrierNodeWeight(std::size_t nodeIndex, std::size_t nodeCount, const Vec3& pointMm) const
{
    const Real baseWeight = boundaryPenaltyWeight(nodeIndex, nodeCount);
    if (baseWeight > static_cast<Real>(0.0))
        return baseWeight;
    return strictBarrierPointEligible(pointMm) ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
}

Real ElasticRodGuidewireModel::strictBarrierEdgeWeight(std::size_t edgeIndex, std::size_t nodeCount, const Vec3& midpointMm) const
{
    if (nodeCount < 2u || !strictBarrierPointEligible(midpointMm))
        return static_cast<Real>(0.0);

    const std::size_t edgeCount = nodeCount - 1u;
    if (edgeIndex >= edgeCount)
        return static_cast<Real>(0.0);

    const std::size_t distalProtectedEdges = std::min<std::size_t>(
        edgeCount,
        std::max<std::size_t>(
            std::max<std::size_t>(magneticEdgeCount(), static_cast<std::size_t>(d_softTipEdgeCount.getValue())),
            2u)
            + 1u);
    const std::size_t protectedStart = edgeCount > distalProtectedEdges ? edgeCount - distalProtectedEdges : 0u;
    if (edgeIndex < protectedStart)
        return static_cast<Real>(0.0);

    const Real alpha = distalProtectedEdges > 1u
        ? static_cast<Real>(edgeIndex - protectedStart) / static_cast<Real>(distalProtectedEdges - 1u)
        : static_cast<Real>(1.0);
    return static_cast<Real>(0.35) + static_cast<Real>(0.40) * smoothstep01(alpha);
}

std::size_t ElasticRodGuidewireModel::magneticEdgeCount() const
{
    return m_core.magneticEdgeCount();
}

void ElasticRodGuidewireModel::configureCoreFromData(const VecCoord& positions)
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
            setCoordCenter(initialCoords[i], solverToScene(coordCenter(positions[i])));
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
    refreshLumenProfile();

    m_cachedNodeCount = positions.size();
    m_bendTwistBlocks.assign(m_cachedNodeCount, LocalBendTwistBlock {});
    m_restSecondDiffM.assign(m_cachedNodeCount, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
    m_restTwistDiff.assign(m_cachedNodeCount, static_cast<Real>(0.0));
    const auto& restCentersM = m_core.undeformedCentersM();
    for (std::size_t i = 1; i + 1 < restCentersM.size(); ++i)
        m_restSecondDiffM[i] = restCentersM[i + 1] - static_cast<Real>(2.0) * restCentersM[i] + restCentersM[i - 1];
    for (std::size_t i = 1; i < positions.size(); ++i)
        m_restTwistDiff[i] = coordTheta(positions[i]) - coordTheta(positions[i - 1]);

    VecReal refLenMm(m_core.refLen().size(), static_cast<Real>(0.0));
    VecReal voronoiLenMm(m_core.voronoiLen().size(), static_cast<Real>(0.0));
    for (std::size_t i = 0; i < refLenMm.size(); ++i)
        refLenMm[i] = m_core.refLen()[i] / kMmToM;
    for (std::size_t i = 0; i < voronoiLenMm.size(); ++i)
        voronoiLenMm[i] = m_core.voronoiLen()[i] / kMmToM;
    d_debugRefLen.setValue(refLenMm);
    d_debugVoronoiLen.setValue(voronoiLenMm);
    VecReal effectiveEA(m_core.edgeCount(), static_cast<Real>(0.0));
    for (std::size_t i = 0; i < effectiveEA.size(); ++i)
        effectiveEA[i] = effectiveAxialEA(i);
    d_debugEA.setValue(effectiveEA);
    d_debugEI.setValue(VecReal(m_core.EI().begin(), m_core.EI().end()));
    d_debugGJ.setValue(VecReal(m_core.GJ().begin(), m_core.GJ().end()));
    d_debugTotalMass.setValue(m_core.totalMassKg());
}

ElasticRodCompatCore::State ElasticRodGuidewireModel::computeCurrentState(const VecCoord& q) const
{
    std::vector<Coord> coords(q.size());
    for (std::size_t i = 0; i < q.size(); ++i)
    {
        coords[i].clear();
        setCoordCenter(coords[i], solverToScene(coordCenter(q[i])));
        setCoordTheta(coords[i], coordTheta(q[i]));
        clearUnused(coords[i]);
    }
    return m_core.computeState(coords);
}

void ElasticRodGuidewireModel::buildRigidState(const VecCoord& q, RigidVecCoord& out) const
{
    const auto state = computeCurrentState(q);
    out.resize(q.size());
    const std::size_t edgeCount = state.materialFrames.size();
    for (std::size_t i = 0; i < q.size(); ++i)
    {
        out[i].getCenter() = solverToScene(coordCenter(q[i]));
        if (edgeCount == 0)
        {
            out[i].getOrientation() = ElasticRodCompatCore::quatFromZTo(kZAxis);
        }
        else
        {
            const std::size_t edgeIndex = std::min(i, edgeCount - 1);
            out[i].getOrientation() = ElasticRodCompatCore::quatFromFrame(state.materialFrames[edgeIndex]);
            out[i].getOrientation().normalize();
        }
    }
}

void ElasticRodGuidewireModel::buildRigidVelocity(const VecCoord& q, const VecDeriv& v, RigidVecDeriv& out) const
{
    const auto state = computeCurrentState(q);
    out.resize(v.size());
    const std::size_t edgeCount = state.materialFrames.size();
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        out[i].getVCenter() = solverToScene(derivCenter(v[i]));
        const Vec3 axis = edgeCount == 0 ? kZAxis : state.materialFrames[std::min(i, edgeCount - 1)].m3;
        out[i].getVOrientation() = derivTheta(v[i]) * axis;
    }
}

void ElasticRodGuidewireModel::init()
{
    Inherit::init();
    if (this->mstate == nullptr)
    {
        msg_error() << "ElasticRodGuidewireModel requires a Vec6d MechanicalObject in the same node.";
        return;
    }

    const VecCoord& x0 = this->mstate->read(sofa::core::vec_id::read_access::position)->getValue();
    configureCoreFromData(x0);
    refreshLumenProfile();
    refreshBendTwistCache(x0);
    updateDebugState(x0);
    msg_info() << "ElasticRodGuidewireModel initialized: nodeCount=" << x0.size()
               << ", edgeCount=" << m_core.edgeCount()
               << ", magneticEdgeCount=" << m_core.magneticEdgeCount()
               << ", softTipEdgeCount=" << d_softTipEdgeCount.getValue()
               << ", contactRadiusMm=" << d_rodRadius.getValue()
               << ", mechanicalCoreRadiusMm=" << d_mechanicalCoreRadiusMm.getValue()
               << ", totalMass=" << m_core.totalMassKg() << " kg";
}

void ElasticRodGuidewireModel::reinit()
{
    Inherit::reinit();
    if (this->mstate == nullptr)
        return;
    const VecCoord& x0 = this->mstate->read(sofa::core::vec_id::read_access::position)->getValue();
    configureCoreFromData(x0);
    refreshLumenProfile();
    refreshBendTwistCache(x0);
    updateDebugState(x0);
}

void ElasticRodGuidewireModel::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (this->mstate == nullptr)
        return;

    const bool isAnimateBegin = sofa::simulation::AnimateBeginEvent::checkEventType(event);
    const bool isAnimateEnd = sofa::simulation::AnimateEndEvent::checkEventType(event);
    if (!isAnimateBegin && !isAnimateEnd)
        return;

    const VecCoord& q = this->mstate->read(sofa::core::vec_id::read_access::position)->getValue();
    if (m_cachedNodeCount != q.size())
        configureCoreFromData(q);

    if (isAnimateBegin && useKinematicSupportBlock())
        projectSupportBlockState(true);

    if (isAnimateEnd)
    {
        const VecCoord& projected = this->mstate->read(sofa::core::vec_id::read_access::position)->getValue();
        refreshBendTwistCache(projected);
        updateDebugState(projected);
        if (d_commitReferenceStateEachStep.getValue())
            m_core.commitState(computeCurrentState(projected));
    }
}
void ElasticRodGuidewireModel::applyPerturbationScene(Coord& coord, unsigned int dof, Real delta) const
{
    if (dof < 3)
    {
        Vec3 c = coordCenter(coord);
        c[dof] += delta;
        setCoordCenter(coord, c);
    }
    else if (dof == 3)
    {
        setCoordTheta(coord, coordTheta(coord) + delta);
    }
    clearUnused(coord);
}

void ElasticRodGuidewireModel::applyPerturbationSI(Coord& coord, unsigned int dof, Real delta) const
{
    applyPerturbationScene(coord, dof, delta);
}

void ElasticRodGuidewireModel::applyLocalPerturbationSI(VecCoord& q, const LocalBendTwistBlock& block, unsigned int dofIndex, Real delta) const
{
    const unsigned int localNode = dofIndex / kActiveNodeDofCount;
    const unsigned int localDof = dofIndex % kActiveNodeDofCount;
    if (localNode >= kLocalNodeCount)
        return;
    applyPerturbationSI(q[block.nodes[localNode]], localDof, delta);
}

void ElasticRodGuidewireModel::computeLocalBendTwistEvaluation(const VecCoord& q, std::size_t interiorIndex, LocalBendTwistEvaluation& evaluation) const
{
    evaluation = LocalBendTwistEvaluation {};
    if (interiorIndex == 0 || interiorIndex + 1 >= q.size())
        return;
    if (interiorIndex >= m_core.voronoiLen().size() || interiorIndex >= m_core.kappaBar().size() || interiorIndex >= m_core.undeformedTwist().size())
        return;

    const std::size_t edgePrev = interiorIndex - 1;
    const std::size_t edgeNext = interiorIndex;
    if (edgeNext >= m_core.edgeCount())
        return;

    const Vec3 p0 = coordCenter(q[edgePrev]);
    const Vec3 p1 = coordCenter(q[interiorIndex]);
    const Vec3 p2 = coordCenter(q[interiorIndex + 1]);
    const Vec3 edge0 = p1 - p0;
    const Vec3 edge1 = p2 - p1;
    const Real ell0 = edge0.norm();
    const Real ell1 = edge1.norm();
    if (ell0 <= kEps || ell1 <= kEps)
        return;

    const auto& committedFrames = m_core.committedReferenceFrames();
    const auto& committedTangents = m_core.committedTangents();
    const auto& committedRefTwist = m_core.committedRefTwist();
    const auto& restFrames = m_core.restFrames();

    const Vec3 fallback0 = edgePrev < committedTangents.size()
        ? committedTangents[edgePrev]
        : (edgePrev < restFrames.size() ? restFrames[edgePrev].m3 : kZAxis);
    const Vec3 fallback1 = edgeNext < committedTangents.size()
        ? committedTangents[edgeNext]
        : (edgeNext < restFrames.size() ? restFrames[edgeNext].m3 : kZAxis);
    const Vec3 t0 = ElasticRodCompatCore::safeNormalize(edge0, fallback0);
    const Vec3 t1 = ElasticRodCompatCore::safeNormalize(edge1, fallback1);

    ElasticRodCompatCore::FrameAxes ref0;
    ElasticRodCompatCore::FrameAxes ref1;
    if (edgePrev < committedFrames.size() && edgePrev < committedTangents.size())
    {
        const Vec3 d1 = ElasticRodCompatCore::parallelTransport(committedFrames[edgePrev].m1, committedTangents[edgePrev], t0);
        ref0 = ElasticRodCompatCore::buildFrameFromTangent(t0, d1);
    }
    else if (edgePrev < restFrames.size())
    {
        ref0 = ElasticRodCompatCore::buildFrameFromTangent(t0, restFrames[edgePrev].m1);
    }
    else
    {
        ref0 = ElasticRodCompatCore::buildFrameFromTangent(t0, Vec3(static_cast<Real>(1.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
    }

    if (edgeNext < committedFrames.size() && edgeNext < committedTangents.size())
    {
        const Vec3 d1 = ElasticRodCompatCore::parallelTransport(committedFrames[edgeNext].m1, committedTangents[edgeNext], t1);
        ref1 = ElasticRodCompatCore::buildFrameFromTangent(t1, d1);
    }
    else if (edgeNext < restFrames.size())
    {
        ref1 = ElasticRodCompatCore::buildFrameFromTangent(t1, restFrames[edgeNext].m1);
    }
    else
    {
        ref1 = ElasticRodCompatCore::buildFrameFromTangent(t1, Vec3(static_cast<Real>(1.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
    }

    const Real thetaPrev = coordTheta(q[edgePrev]);
    const Real thetaNext = coordTheta(q[edgeNext]);
    const auto material0 = ElasticRodCompatCore::rotateFrameAboutAxis(ref0, thetaPrev);
    const auto material1 = ElasticRodCompatCore::rotateFrameAboutAxis(ref1, thetaNext);

    Vec3 transported = ElasticRodCompatCore::parallelTransport(ref0.m1, t0, t1);
    const Real oldRefTwist = edgeNext < committedRefTwist.size() ? committedRefTwist[edgeNext] : static_cast<Real>(0.0);
    rotateAxisAngle(transported, t1, oldRefTwist);
    const Real refTwist = oldRefTwist + ElasticRodCompatCore::signedAngle(transported, ref1.m1, t1);

    const Real denom = std::max(static_cast<Real>(1.0) + sofa::type::dot(t0, t1), kMinCurvatureDenom);
    const Vec3 kb = static_cast<Real>(2.0) * t0.cross(t1) / denom;
    const Vec2 kappa(
        static_cast<Real>(0.5) * sofa::type::dot(kb, material0.m2 + material1.m2),
        static_cast<Real>(-0.5) * sofa::type::dot(kb, material0.m1 + material1.m1));
    const Vec2 deltaKappa = kappa - m_core.kappaBar()[interiorIndex];
    const Real twist = thetaNext - thetaPrev + refTwist;
    const Real twistError = twist - m_core.undeformedTwist()[interiorIndex];

    const Real voronoi = std::max(m_core.voronoiLen()[interiorIndex], static_cast<Real>(kEps));
    evaluation.valid = true;
    evaluation.deltaKappa = deltaKappa;
    evaluation.twistError = twistError;
    evaluation.bendCoeff = averageCoeff(m_core.EI(), interiorIndex) / voronoi;
    evaluation.twistCoeff = averageCoeff(m_core.GJ(), interiorIndex) / voronoi;
    evaluation.energySI =
        static_cast<Real>(0.5) * evaluation.bendCoeff * sofa::type::dot(deltaKappa, deltaKappa)
        + static_cast<Real>(0.5) * evaluation.twistCoeff * twistError * twistError;
}

Real ElasticRodGuidewireModel::localBendTwistEnergySI(const VecCoord& q, std::size_t interiorIndex) const
{
    LocalBendTwistEvaluation evaluation;
    computeLocalBendTwistEvaluation(q, interiorIndex, evaluation);
    return evaluation.valid ? evaluation.energySI : static_cast<Real>(0.0);
}

void ElasticRodGuidewireModel::computeLocalBendTwistResidual(const VecCoord& q, std::size_t interiorIndex, std::array<Real, kResidualCount>& residual) const
{
    residual.fill(static_cast<Real>(0.0));
    LocalBendTwistEvaluation evaluation;
    computeLocalBendTwistEvaluation(q, interiorIndex, evaluation);
    if (!evaluation.valid)
        return;

    residual[0] = std::sqrt(std::max(evaluation.bendCoeff, static_cast<Real>(0.0))) * evaluation.deltaKappa[0];
    residual[1] = std::sqrt(std::max(evaluation.bendCoeff, static_cast<Real>(0.0))) * evaluation.deltaKappa[1];
    residual[2] = std::sqrt(std::max(evaluation.twistCoeff, static_cast<Real>(0.0))) * evaluation.twistError;
}

void ElasticRodGuidewireModel::computeLocalBendTwistBlock(const VecCoord& q, std::size_t interiorIndex, LocalBendTwistBlock& block) const
{
    block = LocalBendTwistBlock {};
    if (interiorIndex == 0 || interiorIndex + 1 >= q.size())
        return;

    block.active = true;
    block.nodes = {interiorIndex - 1, interiorIndex, interiorIndex + 1};
    computeLocalBendTwistResidual(q, interiorIndex, block.residual);

    const Real baseEnergy = localBendTwistEnergySI(q, interiorIndex);
    std::array<Real, kLocalDofCount> eps {};
    std::array<Real, kLocalDofCount> ePlus {};
    std::array<Real, kLocalDofCount> eMinus {};
    std::array<Real, kResidualCount * kLocalDofCount> jacobian {};
    for (unsigned int dofIndex = 0; dofIndex < kLocalDofCount; ++dofIndex)
    {
        eps[dofIndex] = (dofIndex % kActiveNodeDofCount) < 3 ? kTranslationEpsM : kThetaEpsRad;

        VecCoord qPlus(q);
        VecCoord qMinus(q);
        applyLocalPerturbationSI(qPlus, block, dofIndex, eps[dofIndex]);
        applyLocalPerturbationSI(qMinus, block, dofIndex, -eps[dofIndex]);

        std::array<Real, kResidualCount> resPlus {};
        std::array<Real, kResidualCount> resMinus {};
        computeLocalBendTwistResidual(qPlus, interiorIndex, resPlus);
        computeLocalBendTwistResidual(qMinus, interiorIndex, resMinus);
        for (unsigned int r = 0; r < kResidualCount; ++r)
            jacobian[r * kLocalDofCount + dofIndex] = (resPlus[r] - resMinus[r]) / (static_cast<Real>(2.0) * eps[dofIndex]);

        ePlus[dofIndex] = localBendTwistEnergySI(qPlus, interiorIndex);
        eMinus[dofIndex] = localBendTwistEnergySI(qMinus, interiorIndex);

        Real force = static_cast<Real>(0.0);
        for (unsigned int r = 0; r < kResidualCount; ++r)
            force -= jacobian[r * kLocalDofCount + dofIndex] * block.residual[r];
        block.forceSI[dofIndex] = force;
        block.stiffnessSI[dofIndex * kLocalDofCount + dofIndex] =
            (ePlus[dofIndex] - static_cast<Real>(2.0) * baseEnergy + eMinus[dofIndex]) / (eps[dofIndex] * eps[dofIndex]);
    }

    for (unsigned int row = 0; row < kLocalDofCount; ++row)
    {
        for (unsigned int col = row + 1; col < kLocalDofCount; ++col)
        {
            VecCoord qPP(q);
            VecCoord qPM(q);
            VecCoord qMP(q);
            VecCoord qMM(q);
            applyLocalPerturbationSI(qPP, block, row, eps[row]);
            applyLocalPerturbationSI(qPP, block, col, eps[col]);
            applyLocalPerturbationSI(qPM, block, row, eps[row]);
            applyLocalPerturbationSI(qPM, block, col, -eps[col]);
            applyLocalPerturbationSI(qMP, block, row, -eps[row]);
            applyLocalPerturbationSI(qMP, block, col, eps[col]);
            applyLocalPerturbationSI(qMM, block, row, -eps[row]);
            applyLocalPerturbationSI(qMM, block, col, -eps[col]);

            const Real hij =
                (localBendTwistEnergySI(qPP, interiorIndex)
                - localBendTwistEnergySI(qPM, interiorIndex)
                - localBendTwistEnergySI(qMP, interiorIndex)
                + localBendTwistEnergySI(qMM, interiorIndex))
                / (static_cast<Real>(4.0) * eps[row] * eps[col]);
            block.stiffnessSI[row * kLocalDofCount + col] = hij;
            block.stiffnessSI[col * kLocalDofCount + row] = hij;
        }
    }

    constexpr Real invSqrt3 = static_cast<Real>(0.57735026918962576451);
    std::array<Real, kLocalDofCount * 3> rigidTranslationModes {};
    for (unsigned int localNode = 0; localNode < kLocalNodeCount; ++localNode)
    {
        for (unsigned int axis = 0; axis < 3; ++axis)
            rigidTranslationModes[(localNode * kActiveNodeDofCount + axis) * 3 + axis] = invSqrt3;
    }

    std::array<Real, kLocalDofCount> projectedForce = block.forceSI;
    for (unsigned int mode = 0; mode < 3; ++mode)
    {
        Real dotMode = static_cast<Real>(0.0);
        for (unsigned int i = 0; i < kLocalDofCount; ++i)
            dotMode += rigidTranslationModes[i * 3 + mode] * projectedForce[i];
        for (unsigned int i = 0; i < kLocalDofCount; ++i)
            projectedForce[i] -= rigidTranslationModes[i * 3 + mode] * dotMode;
    }
    block.forceSI = projectedForce;

    std::array<Real, kLocalDofCount * kLocalDofCount> projectedStiffness = block.stiffnessSI;
    for (unsigned int mode = 0; mode < 3; ++mode)
    {
        std::array<Real, kLocalDofCount> hr {};
        std::array<Real, kLocalDofCount> rth {};
        Real rthr = static_cast<Real>(0.0);
        for (unsigned int i = 0; i < kLocalDofCount; ++i)
        {
            for (unsigned int j = 0; j < kLocalDofCount; ++j)
            {
                hr[i] += projectedStiffness[i * kLocalDofCount + j] * rigidTranslationModes[j * 3 + mode];
                rth[j] += rigidTranslationModes[i * 3 + mode] * projectedStiffness[i * kLocalDofCount + j];
            }
        }
        for (unsigned int i = 0; i < kLocalDofCount; ++i)
            rthr += rigidTranslationModes[i * 3 + mode] * hr[i];

        for (unsigned int i = 0; i < kLocalDofCount; ++i)
        {
            for (unsigned int j = 0; j < kLocalDofCount; ++j)
            {
                projectedStiffness[i * kLocalDofCount + j] +=
                    -hr[i] * rigidTranslationModes[j * 3 + mode]
                    -rigidTranslationModes[i * 3 + mode] * rth[j]
                    + rigidTranslationModes[i * 3 + mode] * rthr * rigidTranslationModes[j * 3 + mode];
            }
        }
    }
    block.stiffnessSI = projectedStiffness;
}

Real ElasticRodGuidewireModel::blockMatrixValue(const LocalBendTwistBlock& block, unsigned int row, unsigned int col) const
{
    return block.stiffnessSI[row * kLocalDofCount + col];
}

bool ElasticRodGuidewireModel::useDynamicStrictWindows() const
{
    return d_useDynamicSupportWindows.getValue()
        && !d_nodeInitialPathSmm.getValue().empty()
        && d_supportWindowLengthMm.getValue() > static_cast<Real>(kEps);
}

bool ElasticRodGuidewireModel::hasBoundaryDriver() const
{
    if (useDynamicStrictWindows())
        return true;
    return static_cast<std::size_t>(d_pushNodeCount.getValue()) > 0u
        || static_cast<std::size_t>(d_axialDriveNodeCount.getValue()) > 0u;
}

bool ElasticRodGuidewireModel::nodeListedIn(const VecUInt& indices, std::size_t nodeIndex) const
{
    if (indices.empty())
        return true;
    for (const unsigned int idx : indices)
    {
        if (static_cast<std::size_t>(idx) == nodeIndex)
            return true;
    }
    return false;
}

bool ElasticRodGuidewireModel::useFullExternalSupportZone() const
{
    return useDynamicStrictWindows() && d_supportNodeIndices.getValue().empty();
}

bool ElasticRodGuidewireModel::useFullExternalDriveZone() const
{
    return useDynamicStrictWindows() && d_driveNodeIndices.getValue().empty();
}

Real ElasticRodGuidewireModel::nodeNominalPathSmm(std::size_t nodeIndex) const
{
    const auto& nodeInitialPathSmm = d_nodeInitialPathSmm.getValue();
    if (!hasBoundaryDriver())
    {
        if (nodeIndex < nodeInitialPathSmm.size())
            return nodeInitialPathSmm[nodeIndex];
        return m_core.supportArcLengthMm(nodeIndex);
    }
    if (nodeIndex < nodeInitialPathSmm.size())
        return nodeInitialPathSmm[nodeIndex] + d_commandedInsertion.getValue();
    return d_commandedInsertion.getValue() + m_core.supportArcLengthMm(nodeIndex);
}

Real ElasticRodGuidewireModel::supportWindowWeight(std::size_t nodeIndex) const
{
    if (!useDynamicStrictWindows() || !nodeListedIn(d_supportNodeIndices.getValue(), nodeIndex))
        return static_cast<Real>(0.0);

    const Real releaseDistanceMm = std::max(d_supportReleaseDistanceMm.getValue(), static_cast<Real>(0.0));
    const Real nominalS = nodeNominalPathSmm(nodeIndex);
    if (nominalS > static_cast<Real>(0.0) + static_cast<Real>(kEps))
        return static_cast<Real>(0.0);
    if (useFullExternalSupportZone())
    {
        const Real clampedReleaseMm = std::max(releaseDistanceMm, static_cast<Real>(0.0));
        if (clampedReleaseMm <= static_cast<Real>(kEps))
            return static_cast<Real>(1.0);
        const Real releaseStartS = -clampedReleaseMm;
        if (nominalS <= releaseStartS + static_cast<Real>(kEps))
            return static_cast<Real>(1.0);

        const Real alpha = std::clamp(
            (nominalS - releaseStartS) / clampedReleaseMm,
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        return static_cast<Real>(0.5) * (static_cast<Real>(1.0) + std::cos(alpha * kPi));
    }

    const Real supportLengthMm = std::max(d_supportWindowLengthMm.getValue(), static_cast<Real>(0.0));
    if (supportLengthMm <= static_cast<Real>(kEps))
        return static_cast<Real>(0.0);
    const Real supportMinS = -supportLengthMm;
    if (nominalS < supportMinS - static_cast<Real>(kEps))
        return static_cast<Real>(0.0);
    const Real clampedReleaseMm = std::min(releaseDistanceMm, supportLengthMm);
    if (clampedReleaseMm <= static_cast<Real>(kEps))
        return static_cast<Real>(1.0);

    const Real releaseStartS = -clampedReleaseMm;
    if (nominalS <= releaseStartS + static_cast<Real>(kEps))
        return static_cast<Real>(1.0);

    const Real alpha = std::clamp(
        (nominalS - releaseStartS) / clampedReleaseMm,
        static_cast<Real>(0.0),
        static_cast<Real>(1.0));
    return static_cast<Real>(0.5) * (static_cast<Real>(1.0) + std::cos(alpha * kPi));
}

Real ElasticRodGuidewireModel::rigidSupportWeight(std::size_t nodeIndex) const
{
    if (!useDynamicStrictWindows() || !nodeListedIn(d_supportNodeIndices.getValue(), nodeIndex))
        return static_cast<Real>(0.0);

    const Real nominalS = nodeNominalPathSmm(nodeIndex);
    if (nominalS > static_cast<Real>(0.0) + static_cast<Real>(kEps))
        return static_cast<Real>(0.0);
    if (useFullExternalSupportZone())
    {
        const Real releaseDistanceMm = std::max(d_supportReleaseDistanceMm.getValue(), static_cast<Real>(0.0));
        if (releaseDistanceMm <= static_cast<Real>(kEps))
            return static_cast<Real>(1.0);
        return nominalS <= -releaseDistanceMm + static_cast<Real>(kEps)
            ? static_cast<Real>(1.0)
            : static_cast<Real>(0.0);
    }

    const Real supportLengthMm = std::max(d_supportWindowLengthMm.getValue(), static_cast<Real>(0.0));
    if (supportLengthMm <= static_cast<Real>(kEps))
        return static_cast<Real>(0.0);
    const Real supportMinS = -supportLengthMm;
    if (nominalS < supportMinS - static_cast<Real>(kEps))
        return static_cast<Real>(0.0);
    const Real releaseDistanceMm = std::min(
        std::max(d_supportReleaseDistanceMm.getValue(), static_cast<Real>(0.0)),
        supportLengthMm);
    if (releaseDistanceMm <= static_cast<Real>(kEps))
        return static_cast<Real>(1.0);
    const Real releaseStartS = -releaseDistanceMm;
    return nominalS <= releaseStartS + static_cast<Real>(kEps)
        ? static_cast<Real>(1.0)
        : static_cast<Real>(0.0);
}

bool ElasticRodGuidewireModel::driveWindowNodeSelected(std::size_t nodeIndex) const
{
    if (!useDynamicStrictWindows())
        return nodeIndex < std::min<std::size_t>(static_cast<std::size_t>(d_axialDriveNodeCount.getValue()), supportNodeCount());
    if (!nodeListedIn(d_driveNodeIndices.getValue(), nodeIndex))
        return false;

    const Real nominalSAtNode = nodeNominalPathSmm(nodeIndex);
    if (useFullExternalDriveZone())
        return nominalSAtNode <= static_cast<Real>(0.0) + static_cast<Real>(kEps);

    const Real supportLengthMm = std::max(d_supportWindowLengthMm.getValue(), static_cast<Real>(0.0));
    if (supportLengthMm > static_cast<Real>(kEps) && nominalSAtNode < -supportLengthMm - static_cast<Real>(kEps))
        return false;

    const std::size_t nodeCount = std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount());
    if (nodeCount == 0u)
        return false;

    const Real windowLengthMm = std::max(d_driveWindowLengthMm.getValue(), static_cast<Real>(0.0));
    const Real outsideOffsetMm = std::max(d_driveWindowOutsideOffsetMm.getValue(), static_cast<Real>(0.0));
    const Real driveMax = -outsideOffsetMm;
    const Real driveMin = -(outsideOffsetMm + windowLengthMm);
    const unsigned int minCount = std::max(d_driveWindowMinNodeCount.getValue(), 1u);

    std::vector<std::size_t> inside;
    inside.reserve(nodeCount);
    for (std::size_t i = 0; i < nodeCount; ++i)
    {
        if (!nodeListedIn(d_driveNodeIndices.getValue(), i))
            continue;
        const Real nominalS = nodeNominalPathSmm(i);
        if (supportLengthMm > static_cast<Real>(kEps) && nominalS < -supportLengthMm - static_cast<Real>(kEps))
            continue;
        if (nominalS >= driveMin - static_cast<Real>(kEps) && nominalS <= driveMax + static_cast<Real>(kEps))
            inside.push_back(i);
    }
    if (inside.size() >= static_cast<std::size_t>(minCount))
        return std::find(inside.begin(), inside.end(), nodeIndex) != inside.end();

    const Real targetS = driveMax - static_cast<Real>(0.5) * windowLengthMm;
    std::vector<std::pair<std::pair<Real, Real>, std::size_t>> ranked;
    ranked.reserve(nodeCount);
    for (std::size_t i = 0; i < nodeCount; ++i)
    {
        if (!nodeListedIn(d_driveNodeIndices.getValue(), i))
            continue;
        const Real nominalS = nodeNominalPathSmm(i);
        if (supportLengthMm > static_cast<Real>(kEps) && nominalS < -supportLengthMm - static_cast<Real>(kEps))
            continue;
        if (nominalS > driveMax + static_cast<Real>(kEps))
            continue;
        ranked.push_back({{std::abs(nominalS - targetS), std::abs(nominalS - driveMax)}, i});
    }
    if (ranked.empty())
        return false;
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b)
    {
        if (a.first.first != b.first.first)
            return a.first.first < b.first.first;
        if (a.first.second != b.first.second)
            return a.first.second < b.first.second;
        return a.second < b.second;
    });
    const std::size_t count = std::min<std::size_t>(static_cast<std::size_t>(minCount), ranked.size());
    for (std::size_t i = 0; i < count; ++i)
    {
        if (ranked[i].second == nodeIndex)
            return true;
    }
    return false;
}

std::size_t ElasticRodGuidewireModel::primaryBoundaryNodeIndex() const
{
    if (m_cachedNodeCount == 0u)
        return 0u;
    if (!hasBoundaryDriver())
        return 0u;
    if (!useDynamicStrictWindows())
        return 0u;

    std::size_t bestIndex = 0u;
    Real bestS = std::numeric_limits<Real>::infinity();
    bool found = false;
    const std::size_t nodeCount = std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount());
    for (std::size_t i = 0; i < nodeCount; ++i)
    {
        const bool active = supportWindowWeight(i) > static_cast<Real>(0.0) || driveWindowNodeSelected(i);
        if (!active)
            continue;
        const Real nominalS = nodeNominalPathSmm(i);
        if (!found || nominalS < bestS)
        {
            found = true;
            bestS = nominalS;
            bestIndex = i;
        }
    }
    return found ? bestIndex : 0u;
}

void ElasticRodGuidewireModel::computeBoundaryTargets(std::size_t nodeCount, std::vector<Vec3>& targetCenters, std::vector<Real>& targetTheta) const
{
    targetCenters.clear();
    targetTheta.clear();
    if (!hasBoundaryDriver())
        return;
    const Vec3 axis = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    if (!useDynamicStrictWindows())
    {
        m_core.computeBoundaryTargets(
            d_commandedInsertion.getValue(),
            d_commandedTwist.getValue(),
            axis,
            static_cast<std::size_t>(d_pushNodeCount.getValue()),
            targetCenters,
            targetTheta);
        return;
    }

    targetCenters.resize(nodeCount, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
    targetTheta.assign(nodeCount, static_cast<Real>(0.0));
    const Vec3 entryPointMm = !m_tubeNodesCached.empty()
        ? m_tubeNodesCached.front()
        : (!m_core.initialCentersMm().empty() ? m_core.initialCentersMm().front() : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));

    for (std::size_t i = 0; i < nodeCount; ++i)
    {
        const Real nominalS = nodeNominalPathSmm(i);
        // Strict dynamic windows represent only the short external introducer /
        // entry corridor. Once a material point exits that corridor it must be
        // released back to free rod + contact, not softly dragged along the
        // vessel centerline. The previous centerline blending acted like a
        // hidden path-following guide and injected pre-contact bending/stretch
        // into the soft distal head.
        targetCenters[i] = entryPointMm + nominalS * axis;
    }

    if (!targetTheta.empty())
        targetTheta[std::min(primaryBoundaryNodeIndex(), targetTheta.size() - 1u)] = d_commandedTwist.getValue();
}

std::size_t ElasticRodGuidewireModel::supportNodeCount() const
{
    if (useDynamicStrictWindows())
    {
        std::size_t count = 0u;
        const std::size_t nodeCount = std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount());
        for (std::size_t i = 0; i < nodeCount; ++i)
        {
            if (supportWindowWeight(i) > static_cast<Real>(0.0))
                ++count;
        }
        return count;
    }
    return std::min<std::size_t>(
        static_cast<std::size_t>(d_pushNodeCount.getValue()),
        std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount()));
}

std::size_t ElasticRodGuidewireModel::axialDriveNodeCount() const
{
    if (useDynamicStrictWindows())
    {
        std::size_t count = 0u;
        const std::size_t nodeCount = std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount());
        for (std::size_t i = 0; i < nodeCount; ++i)
        {
            if (driveWindowNodeSelected(i))
                ++count;
        }
        return count;
    }
    return std::min<std::size_t>(
        static_cast<std::size_t>(d_axialDriveNodeCount.getValue()),
        supportNodeCount());
}

std::size_t ElasticRodGuidewireModel::supportReleaseNodeCount() const
{
    if (useDynamicStrictWindows())
    {
        std::size_t count = 0u;
        const std::size_t nodeCount = std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount());
        for (std::size_t i = 0; i < nodeCount; ++i)
        {
            const Real weight = supportWindowWeight(i);
            if (weight > static_cast<Real>(0.0) && rigidSupportWeight(i) <= static_cast<Real>(0.0))
                ++count;
        }
        return count;
    }
    const std::size_t supportCount = supportNodeCount();
    if (!useKinematicSupportBlock() || supportCount <= 1)
        return supportCount;

    constexpr std::size_t kReleaseNodes = 2u;
    const std::size_t rigidCount = rigidSupportNodeCount();
    if (supportCount <= rigidCount)
        return static_cast<std::size_t>(0);

    return std::min<std::size_t>(kReleaseNodes, supportCount - rigidCount);
}

std::size_t ElasticRodGuidewireModel::rigidSupportNodeCount() const
{
    if (useDynamicStrictWindows())
    {
        std::size_t count = 0u;
        const std::size_t nodeCount = std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount());
        for (std::size_t i = 0; i < nodeCount; ++i)
        {
            if (rigidSupportWeight(i) > static_cast<Real>(0.0))
                ++count;
        }
        return count;
    }
    const std::size_t supportCount = supportNodeCount();
    if (!useKinematicSupportBlock() || supportCount == 0)
        return static_cast<std::size_t>(0);

    constexpr std::size_t kMinRigidGuideNodes = 4u;
    constexpr std::size_t kReleaseNodes = 2u;
    if (supportCount <= kMinRigidGuideNodes + kReleaseNodes)
        return std::min<std::size_t>(supportCount, kMinRigidGuideNodes);

    // Keep only a short prescribed introducer segment, then hand over to the
    // mapped virtual sheath springs. For soft option.txt DER parameters this
    // avoids a long rigid/free interface that tends to buckle at the distal end
    // of the kinematic block before magnetic steering can take effect.
    return kMinRigidGuideNodes;
}

bool ElasticRodGuidewireModel::useKinematicSupportBlock() const
{
    if (!d_useKinematicSupportBlock.getValue())
        return false;
    const std::size_t minNodes = useDynamicStrictWindows() ? 1u : 2u;
    return supportNodeCount() >= minNodes;
}

bool ElasticRodGuidewireModel::isSupportInteriorEdge(std::size_t edgeIndex) const
{
    if (useDynamicStrictWindows())
        return supportConstitutiveEdgeWeight(edgeIndex) < static_cast<Real>(1.0) - static_cast<Real>(1.0e-6);
    const std::size_t supportCount = useKinematicSupportBlock() ? rigidSupportNodeCount() : supportNodeCount();
    return supportCount >= 2 && (edgeIndex + 1) < supportCount;
}

bool ElasticRodGuidewireModel::isSupportInteriorBlock(std::size_t interiorIndex) const
{
    if (useDynamicStrictWindows())
        return supportConstitutiveBlockWeight(interiorIndex) < static_cast<Real>(1.0) - static_cast<Real>(1.0e-6);
    const std::size_t supportCount = useKinematicSupportBlock() ? rigidSupportNodeCount() : supportNodeCount();
    return supportCount >= 3 && (interiorIndex + 1) < supportCount;
}

void ElasticRodGuidewireModel::projectSupportBlockState(bool updateVelocity)
{
    if (this->mstate == nullptr || !useKinematicSupportBlock() || !hasBoundaryDriver())
        return;

    this->mstate->vRealloc(sofa::core::mechanicalparams::defaultInstance(), sofa::core::vec_id::write_access::freePosition);
    this->mstate->vRealloc(sofa::core::mechanicalparams::defaultInstance(), sofa::core::vec_id::write_access::freeVelocity);

    auto q = this->mstate->writePositions();
    if (q.size() == 0)
        return;
    DataVecCoord* qFreeData = this->mstate->write(sofa::core::vec_id::write_access::freePosition);
    VecCoord* qFree = nullptr;
    if (qFreeData != nullptr)
        qFree = qFreeData->beginEdit();

    std::vector<Vec3> targetCenters;
    std::vector<Real> targetTheta;
    const Vec3 axis = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    computeBoundaryTargets(q.size(), targetCenters, targetTheta);

    const Real dt = std::max(d_dt.getValue(), static_cast<Real>(1.0e-12));
    const Real insertionMm = d_commandedInsertion.getValue();
    const Real twistRad = d_commandedTwist.getValue();
    const Real insertionRateMmS = m_haveProjectedBoundaryState
        ? (insertionMm - m_lastProjectedInsertionMm) / dt
        : static_cast<Real>(0.0);
    const Real twistRateRadS = m_haveProjectedBoundaryState
        ? (twistRad - m_lastProjectedTwistRad) / dt
        : static_cast<Real>(0.0);

    for (std::size_t i = 0; i < q.size() && i < targetCenters.size(); ++i)
    {
        Real projectionWeight = static_cast<Real>(1.0);
        if (useDynamicStrictWindows())
        {
            projectionWeight = supportWindowWeight(i);
            if (projectionWeight <= static_cast<Real>(0.0))
                continue;
        }
        else if (i >= std::min<std::size_t>(q.size(), rigidSupportNodeCount()))
        {
            continue;
        }
        const Vec3 targetCenterM = sceneToSolver(targetCenters[i]);
        const Real targetThetaValue = i < targetTheta.size() ? targetTheta[i] : static_cast<Real>(0.0);
        if (projectionWeight >= static_cast<Real>(1.0) - static_cast<Real>(1.0e-9))
        {
            setCoordCenter(q[i], targetCenterM);
            setCoordTheta(q[i], targetThetaValue);
        }
        else
        {
            const Vec3 blendedCenter = coordCenter(q[i]) + projectionWeight * (targetCenterM - coordCenter(q[i]));
            const Real blendedTheta = coordTheta(q[i]) + projectionWeight * (targetThetaValue - coordTheta(q[i]));
            setCoordCenter(q[i], blendedCenter);
            setCoordTheta(q[i], blendedTheta);
        }
        clearUnused(q[i]);
        if (qFree != nullptr && i < qFree->size())
        {
            if (projectionWeight >= static_cast<Real>(1.0) - static_cast<Real>(1.0e-9))
            {
                setCoordCenter((*qFree)[i], targetCenterM);
                setCoordTheta((*qFree)[i], targetThetaValue);
            }
            else
            {
                const Vec3 blendedCenter = coordCenter((*qFree)[i]) + projectionWeight * (targetCenterM - coordCenter((*qFree)[i]));
                const Real blendedTheta = coordTheta((*qFree)[i]) + projectionWeight * (targetThetaValue - coordTheta((*qFree)[i]));
                setCoordCenter((*qFree)[i], blendedCenter);
                setCoordTheta((*qFree)[i], blendedTheta);
            }
            clearUnused((*qFree)[i]);
        }
    }

    if (updateVelocity)
    {
        auto v = this->mstate->writeVelocities();
        DataVecDeriv* vFreeData = this->mstate->write(sofa::core::vec_id::write_access::freeVelocity);
        VecDeriv* vFree = nullptr;
        if (vFreeData != nullptr)
            vFree = vFreeData->beginEdit();
        const Vec3 targetVel = kMmToM * insertionRateMmS * axis;
        const std::size_t velocityCount = std::min<std::size_t>(q.size(), v.size());
        const std::size_t twistNode = std::min(primaryBoundaryNodeIndex(), velocityCount > 0 ? velocityCount - 1u : 0u);
        for (std::size_t i = 0; i < velocityCount; ++i)
        {
            Real projectionWeight = static_cast<Real>(1.0);
            if (useDynamicStrictWindows())
            {
                projectionWeight = supportWindowWeight(i);
                if (projectionWeight <= static_cast<Real>(0.0))
                    continue;
            }
            else if (i >= std::min<std::size_t>(q.size(), rigidSupportNodeCount()))
            {
                continue;
            }
            const Real targetThetaRate = i == twistNode ? twistRateRadS : static_cast<Real>(0.0);
            if (projectionWeight >= static_cast<Real>(1.0) - static_cast<Real>(1.0e-9))
            {
                setDerivCenter(v[i], targetVel);
                setDerivTheta(v[i], targetThetaRate);
            }
            else
            {
                const Vec3 blendedVel = derivCenter(v[i]) + projectionWeight * (targetVel - derivCenter(v[i]));
                const Real blendedThetaRate = derivTheta(v[i]) + projectionWeight * (targetThetaRate - derivTheta(v[i]));
                setDerivCenter(v[i], blendedVel);
                setDerivTheta(v[i], blendedThetaRate);
            }
            clearUnused(v[i]);
            if (vFree != nullptr && i < vFree->size())
            {
                if (projectionWeight >= static_cast<Real>(1.0) - static_cast<Real>(1.0e-9))
                {
                    setDerivCenter((*vFree)[i], targetVel);
                    setDerivTheta((*vFree)[i], targetThetaRate);
                }
                else
                {
                    const Vec3 blendedVel = derivCenter((*vFree)[i]) + projectionWeight * (targetVel - derivCenter((*vFree)[i]));
                    const Real blendedThetaRate = derivTheta((*vFree)[i]) + projectionWeight * (targetThetaRate - derivTheta((*vFree)[i]));
                    setDerivCenter((*vFree)[i], blendedVel);
                    setDerivTheta((*vFree)[i], blendedThetaRate);
                }
                clearUnused((*vFree)[i]);
            }
        }
        if (vFreeData != nullptr)
            vFreeData->endEdit();
    }
    if (qFreeData != nullptr)
        qFreeData->endEdit();

    m_lastProjectedInsertionMm = insertionMm;
    m_lastProjectedTwistRad = twistRad;
    m_haveProjectedBoundaryState = true;
}

void ElasticRodGuidewireModel::refreshBendTwistCache(const VecCoord& q)
{
    Real maxResidual = static_cast<Real>(0.0);
    if (m_bendTwistBlocks.size() != q.size())
        m_bendTwistBlocks.assign(q.size(), LocalBendTwistBlock {});
    for (std::size_t i = 1; i + 1 < q.size(); ++i)
    {
        computeLocalBendTwistBlock(q, i, m_bendTwistBlocks[i]);
        const auto& block = m_bendTwistBlocks[i];
        if (!block.active)
            continue;
        Real residualNorm = static_cast<Real>(0.0);
        for (Real value : block.residual)
            residualNorm += value * value;
        maxResidual = std::max(maxResidual, std::sqrt(residualNorm));
    }
    d_debugMaxBendResidual.setValue(maxResidual);
}

Real ElasticRodGuidewireModel::boundaryWeight(std::size_t idx, std::size_t count) const
{
    if (!hasBoundaryDriver())
        return static_cast<Real>(0.0);
    if (useDynamicStrictWindows())
        return supportWindowWeight(idx);
    if (count <= 1)
        return static_cast<Real>(1.0);

    const Real alpha = static_cast<Real>(idx) / static_cast<Real>(count - 1);
    constexpr Real kPi = static_cast<Real>(3.14159265358979323846);
    return static_cast<Real>(0.5) * (static_cast<Real>(1.0) + std::cos(alpha * kPi));
}

Real ElasticRodGuidewireModel::axialDriveWeight(std::size_t idx, std::size_t count) const
{
    if (!hasBoundaryDriver())
        return static_cast<Real>(0.0);
    if (useDynamicStrictWindows())
    {
        if (useKinematicSupportBlock())
            return supportWindowWeight(idx);
        return driveWindowNodeSelected(idx) ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
    }
    if (idx >= count)
        return static_cast<Real>(0.0);
    if (!useKinematicSupportBlock())
        return static_cast<Real>(1.0);
    return boundaryWeight(idx, count);
}

Real ElasticRodGuidewireModel::boundaryPenaltyWeight(std::size_t idx, std::size_t count) const
{
    if (!hasBoundaryDriver())
        return static_cast<Real>(0.0);
    if (useDynamicStrictWindows())
        return supportWindowWeight(idx);
    if (idx >= count)
        return static_cast<Real>(0.0);

    if (!useKinematicSupportBlock())
        return boundaryWeight(idx, count);

    const std::size_t rigidCount = rigidSupportNodeCount();
    if (idx < rigidCount || count <= rigidCount)
    {
        // Kinematic support needs an actual restoring force during the implicit solve.
        // A pure AnimateBegin projection lets the support nodes drift again before the
        // step ends, which shows up as edge-0/edge-1 stretch and re-injects energy.
        return static_cast<Real>(1.0);
    }

    const std::size_t releaseCount = supportReleaseNodeCount();
    if (releaseCount == 0u || idx >= rigidCount + releaseCount)
        return static_cast<Real>(0.0);

    const Real beta = static_cast<Real>(idx - rigidCount + 1u) / static_cast<Real>(releaseCount);
    constexpr Real kPi = static_cast<Real>(3.14159265358979323846);
    return static_cast<Real>(0.5) * (static_cast<Real>(1.0) + std::cos(beta * kPi));
}

Real ElasticRodGuidewireModel::supportConstitutiveEdgeWeight(std::size_t edgeIndex) const
{
    if (useDynamicStrictWindows())
        return static_cast<Real>(1.0);
    if (!useKinematicSupportBlock())
        return static_cast<Real>(1.0);

    const std::size_t rigidCount = rigidSupportNodeCount();
    if (rigidCount < 2)
        return static_cast<Real>(1.0);
    if ((edgeIndex + 1) < rigidCount)
        return static_cast<Real>(0.0);

    const std::size_t transitionEdges = supportReleaseNodeCount() + 1u;
    if (transitionEdges == 0u)
        return static_cast<Real>(1.0);

    const std::size_t firstTransitionEdge = rigidCount - 1u;
    if (edgeIndex < firstTransitionEdge)
        return static_cast<Real>(0.0);

    const std::size_t transitionIndex = edgeIndex - firstTransitionEdge;
    if (transitionIndex >= transitionEdges)
        return static_cast<Real>(1.0);

    const Real alpha = static_cast<Real>(transitionIndex + 1u) / static_cast<Real>(transitionEdges);
    constexpr Real kPi = static_cast<Real>(3.14159265358979323846);
    return static_cast<Real>(0.5) * (static_cast<Real>(1.0) - std::cos(alpha * kPi));
}

Real ElasticRodGuidewireModel::supportConstitutiveBlockWeight(std::size_t interiorIndex) const
{
    if (useDynamicStrictWindows())
        return static_cast<Real>(1.0);
    if (!useKinematicSupportBlock())
        return static_cast<Real>(1.0);
    if (interiorIndex == 0u)
        return static_cast<Real>(1.0);

    const Real wPrev = supportConstitutiveEdgeWeight(interiorIndex - 1u);
    const Real wNext = supportConstitutiveEdgeWeight(interiorIndex);
    return static_cast<Real>(0.5) * (wPrev + wNext);
}

void ElasticRodGuidewireModel::updateDebugState(const VecCoord& q)
{
    const auto state = computeCurrentState(q);
    VecReal edgeLenMm(state.edgeLenM.size(), static_cast<Real>(0.0));
    VecReal stretch(state.edgeLenM.size(), static_cast<Real>(0.0));
    VecVec3 kappa(state.kappa.size(), Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
    VecReal twist(state.twist.size(), static_cast<Real>(0.0));

    int abnormalIndex = -1;
    Real abnormalLenMm = static_cast<Real>(0.0);
    Real abnormalRefMm = static_cast<Real>(0.0);

    for (std::size_t i = 0; i < state.edgeLenM.size(); ++i)
    {
        edgeLenMm[i] = state.edgeLenM[i] / kMmToM;
        const Real refLen = i < m_core.refLen().size() ? m_core.refLen()[i] : static_cast<Real>(0.0);
        if (refLen > kEps)
            stretch[i] = (state.edgeLenM[i] - refLen) / refLen;
        if (abnormalIndex < 0)
        {
            const bool abnormal = !std::isfinite(edgeLenMm[i]) || !std::isfinite(stretch[i]) || std::abs(stretch[i]) > kAbnormalStretchRatio;
            if (abnormal)
            {
                abnormalIndex = static_cast<int>(i);
                abnormalLenMm = edgeLenMm[i];
                abnormalRefMm = refLen / kMmToM;
            }
        }
    }

    for (std::size_t i = 0; i < state.kappa.size(); ++i)
        kappa[i] = Vec3(state.kappa[i][0], state.kappa[i][1], static_cast<Real>(0.0));
    for (std::size_t i = 0; i < state.twist.size(); ++i)
        twist[i] = state.twist[i] - m_core.undeformedTwist()[i];

    const Vec3 insertionDir = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    std::vector<Vec3> targetCenters;
    std::vector<Real> targetTheta;
    computeBoundaryTargets(q.size(), targetCenters, targetTheta);

    Real axialErrMm = static_cast<Real>(0.0);
    Real lateralErrMm = static_cast<Real>(0.0);
    Real minLumenClearanceMm = std::numeric_limits<Real>::infinity();
    const std::size_t nodeCount = std::min(targetCenters.size(), q.size());
    for (std::size_t i = 0; i < nodeCount; ++i)
    {
        const Real lateralWeight = boundaryPenaltyWeight(i, nodeCount);
        const Real axialWeight = axialDriveWeight(i, nodeCount);
        if (lateralWeight <= static_cast<Real>(0.0) && axialWeight <= static_cast<Real>(0.0))
            continue;
        const Vec3 dxMm = solverToScene(coordCenter(q[i])) - targetCenters[i];
        if (axialWeight > static_cast<Real>(0.0))
            axialErrMm = std::max(axialErrMm, std::abs(sofa::type::dot(dxMm, insertionDir)));
        if (lateralWeight > static_cast<Real>(0.0))
            lateralErrMm = std::max(lateralErrMm, projectorLateral(insertionDir, dxMm).norm());
    }

    if (d_strictLumenBarrierEnabled.getValue() && hasLumenProfile())
    {
        const Real safetyMarginMm = std::max(d_strictLumenSafetyMarginMm.getValue(), static_cast<Real>(0.0));
        for (std::size_t i = 0; i < q.size(); ++i)
        {
            const Vec3 pointMm = solverToScene(coordCenter(q[i]));
            const Real barrierWeight = strictBarrierNodeWeight(i, q.size(), pointMm);
            if (barrierWeight <= static_cast<Real>(0.0))
                continue;

            Vec3 closestPointMm;
            Real projS = static_cast<Real>(0.0);
            Real clearanceMm = std::numeric_limits<Real>::infinity();
            Vec3 outwardNormalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
            if (!sampleStrictLumenConstraint(i, pointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, outwardNormalM))
                continue;
            minLumenClearanceMm = std::min(minLumenClearanceMm, clearanceMm);
        }
        for (std::size_t i = 0; i + 1 < q.size(); ++i)
        {
            const Vec3 midpointMm = solverToScene(static_cast<Real>(0.5) * (coordCenter(q[i]) + coordCenter(q[i + 1])));
            const Real barrierWeight = strictBarrierEdgeWeight(i, q.size(), midpointMm);
            if (barrierWeight <= static_cast<Real>(0.0))
                continue;

            Vec3 closestPointMm;
            Real projS = static_cast<Real>(0.0);
            Real clearanceMm = std::numeric_limits<Real>::infinity();
            Vec3 outwardNormalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
            if (!sampleStrictLumenConstraintForEdge(i, midpointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, outwardNormalM))
                continue;
            minLumenClearanceMm = std::min(minLumenClearanceMm, clearanceMm);
        }
    }

    Real maxHeadStretch = static_cast<Real>(0.0);
    const std::size_t headEdges = std::min<std::size_t>(magneticEdgeCount(), stretch.size());
    const std::size_t headStart = stretch.size() > headEdges ? stretch.size() - headEdges : 0u;
    for (std::size_t i = headStart; i < stretch.size(); ++i)
        maxHeadStretch = std::max(maxHeadStretch, std::abs(stretch[i]));

    d_debugEdgeLengthMm.setValue(edgeLenMm);
    d_debugStretch.setValue(stretch);
    d_debugKappa.setValue(kappa);
    d_debugTwist.setValue(twist);
    d_debugTipProgress.setValue(hasBoundaryDriver() ? d_commandedInsertion.getValue() : static_cast<Real>(0.0));
    d_debugAbnormalEdgeIndex.setValue(abnormalIndex);
    d_debugAbnormalEdgeLengthMm.setValue(abnormalLenMm);
    d_debugAbnormalEdgeRefLengthMm.setValue(abnormalRefMm);
    d_debugMaxAxialBoundaryErrorMm.setValue(axialErrMm);
    d_debugMaxLateralBoundaryErrorMm.setValue(lateralErrMm);
    d_debugMinLumenClearanceMm.setValue(minLumenClearanceMm);
    d_debugMaxHeadStretch.setValue(maxHeadStretch);
}

Real ElasticRodGuidewireModel::stretchEnergySI(const Coord& a, const Coord& b, std::size_t edgeIndex) const
{
    const Vec3 edgeM = coordCenter(b) - coordCenter(a);
    const Real ell = edgeM.norm();
    const Real refLen = m_core.refLen()[edgeIndex];
    const Real k = effectiveAxialEA(edgeIndex) / std::max(refLen, static_cast<Real>(kEps));
    const Real delta = ell - refLen;
    return static_cast<Real>(0.5) * k * delta * delta;
}

Real ElasticRodGuidewireModel::effectiveAxialEA(std::size_t edgeIndex) const
{
    if (edgeIndex >= m_core.edgeCount())
        return static_cast<Real>(0.0);

    Real ea = m_core.EA()[edgeIndex];
    const Real scale = std::max(d_axialStretchStiffnessScale.getValue(), static_cast<Real>(1.0));
    if (d_axialStretchUseBodyFloor.getValue())
    {
        const Real coreRadiusM = std::max(d_mechanicalCoreRadiusMm.getValue(), static_cast<Real>(0.0)) * kMmToM;
        const Real area = static_cast<Real>(kPi) * coreRadiusM * coreRadiusM;
        const Real bodyEA = std::max(d_youngBody.getValue(), static_cast<Real>(0.0)) * area;
        ea = std::max(ea, bodyEA);
    }
    return ea * scale;
}

Real ElasticRodGuidewireModel::bendTwistEnergySI(const VecCoord& q, std::size_t interiorIndex) const
{
    return localBendTwistEnergySI(q, interiorIndex);
}

Real ElasticRodGuidewireModel::boundaryEnergySI(const VecCoord& q) const
{
    if (q.empty())
        return static_cast<Real>(0.0);

    const Vec3 axis = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    std::vector<Vec3> targetCenters;
    std::vector<Real> targetTheta;
    computeBoundaryTargets(q.size(), targetCenters, targetTheta);

    Real energy = static_cast<Real>(0.0);
    const std::size_t nodeCount = std::min(targetCenters.size(), q.size());
    for (std::size_t i = 0; i < nodeCount; ++i)
    {
        const Vec3 dxM = coordCenter(q[i]) - sceneToSolver(targetCenters[i]);
        const Real lateralWeight = boundaryPenaltyWeight(i, nodeCount);
        const Real axialWeight = axialDriveWeight(i, nodeCount);
        if (lateralWeight > static_cast<Real>(0.0))
        {
            const Vec3 lateralError = projectorLateral(axis, dxM);
            energy += static_cast<Real>(0.5) * lateralWeight
                * d_proximalLateralStiffness.getValue()
                * sofa::type::dot(lateralError, lateralError);
        }
        if (axialWeight > static_cast<Real>(0.0))
        {
            const Vec3 axialError = projectorAxial(axis, dxM);
            energy += static_cast<Real>(0.5) * axialWeight
                * d_proximalAxialStiffness.getValue()
                * sofa::type::dot(axialError, axialError);
        }
    }
    if (!targetTheta.empty())
    {
        const std::size_t twistNode = std::min(primaryBoundaryNodeIndex(), std::min(q.size(), targetTheta.size()) - 1u);
        const Real thetaError = coordTheta(q[twistNode]) - targetTheta[twistNode];
        energy += static_cast<Real>(0.5) * d_proximalAngularStiffness.getValue() * thetaError * thetaError;
    }
    if (d_strictLumenBarrierEnabled.getValue() && hasLumenProfile())
    {
        const Real activationMarginM = kMmToM * std::max(d_strictLumenActivationMarginMm.getValue(), static_cast<Real>(0.0));
        const Real stiffness = std::max(d_strictLumenBarrierStiffness.getValue(), static_cast<Real>(0.0));
        const Real safetyMarginMm = std::max(d_strictLumenSafetyMarginMm.getValue(), static_cast<Real>(0.0));
        for (std::size_t i = 0; i < q.size(); ++i)
        {
            const Vec3 pointMm = solverToScene(coordCenter(q[i]));
            const Real barrierWeight = strictBarrierNodeWeight(i, q.size(), pointMm);
            if (barrierWeight <= static_cast<Real>(0.0))
                continue;
            Vec3 closestPointMm;
            Real projS = static_cast<Real>(0.0);
            Real clearanceMm = std::numeric_limits<Real>::infinity();
            Vec3 outwardNormalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
            if (!sampleStrictLumenConstraint(i, pointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, outwardNormalM))
                continue;
            const Real clearanceM = kMmToM * clearanceMm;
            const Real penetration = std::max(activationMarginM - clearanceM, static_cast<Real>(0.0));
            energy += static_cast<Real>(0.5) * barrierWeight * stiffness * penetration * penetration;
        }
        for (std::size_t i = 0; i + 1 < q.size(); ++i)
        {
            const Vec3 midpointMm = solverToScene(static_cast<Real>(0.5) * (coordCenter(q[i]) + coordCenter(q[i + 1])));
            const Real barrierWeight = strictBarrierEdgeWeight(i, q.size(), midpointMm);
            if (barrierWeight <= static_cast<Real>(0.0))
                continue;
            Vec3 closestPointMm;
            Real projS = static_cast<Real>(0.0);
            Real clearanceMm = std::numeric_limits<Real>::infinity();
            Vec3 outwardNormalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
            if (!sampleStrictLumenConstraintForEdge(i, midpointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, outwardNormalM))
                continue;
            const Real edgeActivationMarginM = kMmToM * std::min(
                std::max(d_strictLumenActivationMarginMm.getValue(), static_cast<Real>(0.0)),
                safetyMarginMm + static_cast<Real>(0.25));
            const Real clearanceM = kMmToM * clearanceMm;
            const Real penetration = std::max(edgeActivationMarginM - clearanceM, static_cast<Real>(0.0));
            energy += static_cast<Real>(0.5) * barrierWeight * static_cast<Real>(0.65) * stiffness * penetration * penetration;
        }
    }
    return energy;
}

void ElasticRodGuidewireModel::applyStrictLumenBarrier(
    const VecCoord& q,
    const VecDeriv& v,
    VecDeriv& f,
    Real& minClearanceMm,
    Vec3& totalBarrierForce,
    unsigned int& activeNodeCount) const
{
    minClearanceMm = std::numeric_limits<Real>::infinity();
    totalBarrierForce = Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    activeNodeCount = 0u;
    if (!d_strictLumenBarrierEnabled.getValue() || !hasLumenProfile())
        return;

    const Real activationMarginM = kMmToM * std::max(d_strictLumenActivationMarginMm.getValue(), static_cast<Real>(0.0));
    const Real activationMarginMm = std::max(d_strictLumenActivationMarginMm.getValue(), static_cast<Real>(0.0));
    const Real safetyMarginMm = std::max(d_strictLumenSafetyMarginMm.getValue(), static_cast<Real>(0.0));
    const Real stiffness = std::max(d_strictLumenBarrierStiffness.getValue(), static_cast<Real>(0.0));
    const Real damping = std::max(d_strictLumenBarrierDamping.getValue(), static_cast<Real>(0.0));
    const Real forceCap = std::max(d_strictLumenBarrierMaxForcePerNodeN.getValue(), static_cast<Real>(0.0));

    for (std::size_t i = 0; i < q.size(); ++i)
    {
        const Vec3 pointMm = solverToScene(coordCenter(q[i]));
        const Real barrierWeight = strictBarrierNodeWeight(i, q.size(), pointMm);
        if (barrierWeight <= static_cast<Real>(0.0))
            continue;

        Vec3 closestPointMm;
        Real projS = static_cast<Real>(0.0);
        Real clearanceMm = std::numeric_limits<Real>::infinity();
        Vec3 outwardNormalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (!sampleStrictLumenConstraint(i, pointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, outwardNormalM))
            continue;
        minClearanceMm = std::min(minClearanceMm, clearanceMm);

        const Real activationDepthM = std::max(activationMarginM - kMmToM * clearanceMm, static_cast<Real>(0.0));
        if (activationDepthM <= static_cast<Real>(0.0))
            continue;

        if (outwardNormalM.norm() <= kEps)
            continue;
        const Real velocityGate = strictBarrierVelocityGate(clearanceMm, activationMarginMm, safetyMarginMm);
        const Real penetrationDepthM = strictBarrierPenetrationDepthM(clearanceMm, safetyMarginMm);
        const Real outwardSpeed = i < v.size()
            ? std::max(sofa::type::dot(derivCenter(v[i]), outwardNormalM), static_cast<Real>(0.0))
            : static_cast<Real>(0.0);
        const Real forceMagnitudeN = std::min(
            barrierWeight * (stiffness * penetrationDepthM + velocityGate * damping * outwardSpeed),
            forceCap);
        if (forceMagnitudeN <= static_cast<Real>(0.0))
            continue;

        const Vec3 barrierForceN = -forceMagnitudeN * outwardNormalM;
        addCenterScene(f[i], barrierForceN);
        totalBarrierForce += barrierForceN;
        if (penetrationDepthM > static_cast<Real>(0.0))
            ++activeNodeCount;
    }

    for (std::size_t i = 0; i + 1 < q.size(); ++i)
    {
        const Vec3 midpointM = static_cast<Real>(0.5) * (coordCenter(q[i]) + coordCenter(q[i + 1]));
        const Vec3 midpointMm = solverToScene(midpointM);
        const Real barrierWeight = strictBarrierEdgeWeight(i, q.size(), midpointMm);
        if (barrierWeight <= static_cast<Real>(0.0))
            continue;

        Vec3 closestPointMm;
        Real projS = static_cast<Real>(0.0);
        Real clearanceMm = std::numeric_limits<Real>::infinity();
        Vec3 outwardNormalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (!sampleStrictLumenConstraintForEdge(i, midpointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, outwardNormalM))
            continue;
        minClearanceMm = std::min(minClearanceMm, clearanceMm);

        const Real edgeActivationMarginMm = std::min(activationMarginMm, safetyMarginMm + static_cast<Real>(0.25));
        const Real activationDepthM = std::max(kMmToM * edgeActivationMarginMm - kMmToM * clearanceMm, static_cast<Real>(0.0));
        if (activationDepthM <= static_cast<Real>(0.0))
            continue;
        if (outwardNormalM.norm() <= kEps)
            continue;

        const Vec3 midpointVelocityM = (i + 1 < v.size())
            ? static_cast<Real>(0.5) * (derivCenter(v[i]) + derivCenter(v[i + 1]))
            : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        const Real outwardSpeed = std::max(sofa::type::dot(midpointVelocityM, outwardNormalM), static_cast<Real>(0.0));
        const Real forceMagnitudeN = std::min(
            barrierWeight * (
                static_cast<Real>(0.65) * stiffness * activationDepthM
                + static_cast<Real>(0.70) * damping * outwardSpeed),
            static_cast<Real>(0.70) * forceCap);
        if (forceMagnitudeN <= static_cast<Real>(0.0))
            continue;

        const Vec3 barrierForceN = -forceMagnitudeN * outwardNormalM;
        addCenterScene(f[i], static_cast<Real>(0.5) * barrierForceN);
        addCenterScene(f[i + 1], static_cast<Real>(0.5) * barrierForceN);
        totalBarrierForce += barrierForceN;
        ++activeNodeCount;
    }
}

Real ElasticRodGuidewireModel::accumulateStretchForces(const VecCoord& q, const VecDeriv& v, VecDeriv& f) const
{
    Real maxStretchForce = static_cast<Real>(0.0);
    const std::size_t edgeCount = std::min<std::size_t>(m_core.edgeCount(), q.size() > 0 ? q.size() - 1 : 0);
    const Real axialC = std::max(d_edgeAxialDamping.getValue(), static_cast<Real>(0.0));
    for (std::size_t i = 0; i < edgeCount; ++i)
    {
        const Real weight = supportConstitutiveEdgeWeight(i);
        if (weight <= static_cast<Real>(0.0))
            continue;

        const Vec3 edgeM = coordCenter(q[i + 1]) - coordCenter(q[i]);
        const Real ell = edgeM.norm();
        if (ell <= static_cast<Real>(kEps))
            continue;
        const Vec3 dir = edgeM / ell;
        const Real refLen = m_core.refLen()[i];
        const Real k = effectiveAxialEA(i) / std::max(refLen, static_cast<Real>(kEps));
        const Vec3 dvM = i + 1 < v.size()
            ? (derivCenter(v[i + 1]) - derivCenter(v[i]))
            : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        const Real axialRate = sofa::type::dot(dvM, dir);
        const Vec3 forceN = weight * (k * (ell - refLen) + axialC * axialRate) * dir;
        maxStretchForce = std::max(maxStretchForce, forceN.norm());
        addCenterScene(f[i], forceN);
        addCenterScene(f[i + 1], -forceN);
    }
    return maxStretchForce;
}

void ElasticRodGuidewireModel::accumulateBendTwistForces(const VecCoord& q, VecDeriv& f) const
{
    for (std::size_t i = 1; i + 1 < q.size(); ++i)
    {
        const Real weight = supportConstitutiveBlockWeight(i);
        if (weight <= static_cast<Real>(0.0))
            continue;
        if (i >= m_bendTwistBlocks.size())
            continue;
        const auto& block = m_bendTwistBlocks[i];
        if (!block.active)
            continue;

        for (unsigned int dofIndex = 0; dofIndex < kLocalDofCount; ++dofIndex)
        {
            const unsigned int localNode = dofIndex / kActiveNodeDofCount;
            const unsigned int localDof = dofIndex % kActiveNodeDofCount;
            addLocalDof(f[block.nodes[localNode]], localDof, weight * block.forceSI[dofIndex]);
        }
    }
}

void ElasticRodGuidewireModel::accumulateBoundaryForces(const VecCoord& q, const VecDeriv& v, VecDeriv& f, Real& maxForceN, Real& maxTorqueNm, Real& driveReactionN) const
{
    maxForceN = static_cast<Real>(0.0);
    maxTorqueNm = static_cast<Real>(0.0);
    driveReactionN = static_cast<Real>(0.0);
    if (q.empty())
        return;

    const Vec3 axis = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    std::vector<Vec3> targetCenters;
    std::vector<Real> targetTheta;
    computeBoundaryTargets(q.size(), targetCenters, targetTheta);

    const Real axialK = d_proximalAxialStiffness.getValue();
    const Real lateralK = d_proximalLateralStiffness.getValue();
    const Real twistK = d_proximalAngularStiffness.getValue();
    const Real linearC = d_proximalLinearDamping.getValue();
    const Real twistC = d_proximalAngularDamping.getValue();

    const std::size_t nodeCount = std::min(targetCenters.size(), q.size());
    for (std::size_t i = 0; i < nodeCount; ++i)
    {
        const Vec3 dxM = coordCenter(q[i]) - sceneToSolver(targetCenters[i]);
        const Vec3 velM = derivCenter(v[i]);
        const Real lateralWeight = boundaryPenaltyWeight(i, nodeCount);
        const Real axialWeight = axialDriveWeight(i, nodeCount);
        Vec3 forceN(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (lateralWeight > static_cast<Real>(0.0))
        {
            const Vec3 lateralDx = projectorLateral(axis, dxM);
            const Vec3 lateralVelM = projectorLateral(axis, velM);
            forceN += lateralWeight * (
                -lateralK * lateralDx
                - linearC * lateralVelM);
        }
        if (axialWeight > static_cast<Real>(0.0))
        {
            const Vec3 axialDx = projectorAxial(axis, dxM);
            const Vec3 axialVelM = projectorAxial(axis, velM);
            const Vec3 axialForceN = axialWeight * (
                -axialK * axialDx
                - linearC * axialVelM);
            forceN += axialForceN;
            driveReactionN += std::abs(sofa::type::dot(axialForceN, axis));
        }
        maxForceN = std::max(maxForceN, forceN.norm());
        addCenterScene(f[i], forceN);
    }
    if (!targetTheta.empty())
    {
        const std::size_t twistNode = std::min(primaryBoundaryNodeIndex(), std::min(q.size(), targetTheta.size()) - 1u);
        const Real thetaError = coordTheta(q[twistNode]) - targetTheta[twistNode];
        const Real thetaRate = derivTheta(v[twistNode]);
        const Real torqueNm = -twistK * thetaError - twistC * thetaRate;
        maxTorqueNm = std::max(maxTorqueNm, std::abs(torqueNm));
        addTheta(f[twistNode], torqueNm);
    }
}

void ElasticRodGuidewireModel::addForce(const sofa::core::MechanicalParams*, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    VecDeriv& out = *f.beginEdit();
    const VecCoord& q = x.getValue();
    const VecDeriv& vel = v.getValue();
    if (out.size() < q.size())
        out.resize(q.size());
    if (m_cachedNodeCount != q.size())
        configureCoreFromData(q);

    refreshBendTwistCache(q);
    // Debug state is refreshed on AnimateEnd after the solver has converged.
    // Recomputing the full diagnostic package inside addForce() puts the whole
    // lumen/barrier scan on the force-assembly hot path and scales with solver
    // iterations, which is too expensive for realtime GUI runs.

    VecDeriv local(q.size());
    for (auto& d : local)
        zeroDeriv(d);

    const Real maxStretchForce = accumulateStretchForces(q, vel, local);
    accumulateBendTwistForces(q, local);

    Real maxInternalForce = static_cast<Real>(0.0);
    for (const auto& d : local)
        maxInternalForce = std::max(maxInternalForce, translationalForceN(d));

    Real maxBoundaryForce = static_cast<Real>(0.0);
    Real maxBoundaryTorque = static_cast<Real>(0.0);
    Real driveReactionN = static_cast<Real>(0.0);
    accumulateBoundaryForces(q, vel, local, maxBoundaryForce, maxBoundaryTorque, driveReactionN);

    Real minLumenClearanceMm = std::numeric_limits<Real>::infinity();
    Vec3 totalBarrierForce(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    unsigned int barrierActiveNodeCount = 0u;
    applyStrictLumenBarrier(q, vel, local, minLumenClearanceMm, totalBarrierForce, barrierActiveNodeCount);

    for (std::size_t i = 0; i < local.size(); ++i)
    {
        for (unsigned int c = 0; c < 6; ++c)
            out[i][c] += local[i][c];
    }

    d_debugMaxInternalForceN.setValue(maxInternalForce);
    d_debugMaxStretchForceN.setValue(maxStretchForce);
    d_debugMaxBoundaryForceN.setValue(maxBoundaryForce);
    d_debugMaxBoundaryTorqueNm.setValue(maxBoundaryTorque);
    d_debugDriveReactionN.setValue(driveReactionN);
    d_debugMinLumenClearanceMm.setValue(minLumenClearanceMm);
    d_debugBarrierForceVector.setValue(totalBarrierForce);
    d_debugBarrierActiveNodeCount.setValue(barrierActiveNodeCount);

    f.endEdit();
}

void ElasticRodGuidewireModel::applyStretchDForce(const VecCoord& q, const VecDeriv& dx, VecDeriv& df, Real kFactor, Real bFactor) const
{
    const std::size_t edgeCount = std::min<std::size_t>(m_core.edgeCount(), q.size() > 0 ? q.size() - 1 : 0);
    const Real axialC = std::max(d_edgeAxialDamping.getValue(), static_cast<Real>(0.0));
    for (std::size_t i = 0; i < edgeCount; ++i)
    {
        const Real weight = supportConstitutiveEdgeWeight(i);
        if (weight <= static_cast<Real>(0.0))
            continue;

        const Vec3 edgeM = coordCenter(q[i + 1]) - coordCenter(q[i]);
        const Real ell = edgeM.norm();
        if (ell <= static_cast<Real>(kEps))
            continue;
        const Vec3 dir = edgeM / ell;
        const Vec3 dEdgeM = derivCenter(dx[i + 1]) - derivCenter(dx[i]);
        const Vec3 axial = projectorAxial(dir, dEdgeM);
        const Real k = effectiveAxialEA(i) / std::max(m_core.refLen()[i], static_cast<Real>(kEps));
        // addDForce must use the true derivative of the axial spring force.
        // For an edge force f_i = k * (dx_j - dx_i)_axial, the Jacobian action is
        // df_i = +k * axial and df_j = -k * axial. The previous sign inverted this
        // operator and made the implicit solve behave like an anti-spring.
        const Vec3 forceN = weight * (kFactor * k + bFactor * axialC) * axial;
        addCenterScene(df[i], forceN);
        addCenterScene(df[i + 1], -forceN);
    }
}

void ElasticRodGuidewireModel::applyBendTwistDForce(const VecDeriv& dx, VecDeriv& df, Real kFactor) const
{
    for (std::size_t i = 1; i + 1 < dx.size(); ++i)
    {
        const Real weight = supportConstitutiveBlockWeight(i);
        if (weight <= static_cast<Real>(0.0))
            continue;
        if (i >= m_bendTwistBlocks.size())
            continue;
        const auto& block = m_bendTwistBlocks[i];
        if (!block.active)
            continue;

        for (unsigned int row = 0; row < kLocalDofCount; ++row)
        {
            Real localValue = static_cast<Real>(0.0);
            for (unsigned int col = 0; col < kLocalDofCount; ++col)
            {
                const unsigned int localNode = col / kActiveNodeDofCount;
                const unsigned int localDof = col % kActiveNodeDofCount;
                localValue += blockMatrixValue(block, row, col) * derivComponent(dx[block.nodes[localNode]], localDof);
            }

            const unsigned int localNode = row / kActiveNodeDofCount;
            const unsigned int localDof = row % kActiveNodeDofCount;
            addLocalDof(df[block.nodes[localNode]], localDof, -weight * kFactor * localValue);
        }
    }
}
void ElasticRodGuidewireModel::applyBoundaryDForce(const VecCoord& q, const VecDeriv& dx, VecDeriv& df, Real kFactor, Real bFactor) const
{
    if (dx.empty())
        return;

    const Vec3 axis = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    const Real axialK = d_proximalAxialStiffness.getValue();
    const Real lateralK = d_proximalLateralStiffness.getValue();
    const Real twistK = d_proximalAngularStiffness.getValue();
    const Real linearC = d_proximalLinearDamping.getValue();
    const Real twistC = d_proximalAngularDamping.getValue();

    const std::size_t nodeCount = useDynamicStrictWindows()
        ? std::min<std::size_t>(dx.size(), m_core.nodeCount())
        : std::min<std::size_t>(
            static_cast<std::size_t>(d_pushNodeCount.getValue()),
            std::min<std::size_t>(dx.size(), m_core.nodeCount()));
    for (std::size_t i = 0; i < nodeCount; ++i)
    {
        const Vec3 dCenterM = derivCenter(dx[i]);
        const Vec3 axialDelta = projectorAxial(axis, dCenterM);
        const Vec3 lateralDelta = projectorLateral(axis, dCenterM);
        const Real lateralWeight = boundaryPenaltyWeight(i, nodeCount);
        Vec3 dForceN = lateralWeight * (
            -kFactor * lateralK * lateralDelta
            -bFactor * linearC * lateralDelta);
        const Real axialWeight = axialDriveWeight(i, nodeCount);
        if (axialWeight > static_cast<Real>(0.0))
        {
            dForceN += axialWeight * (
                -kFactor * axialK * axialDelta
                -bFactor * linearC * axialDelta);
        }
        addCenterScene(df[i], dForceN);
    }
    if (nodeCount > 0u)
    {
        const std::size_t twistNode = std::min(primaryBoundaryNodeIndex(), nodeCount - 1u);
        addTheta(df[twistNode], -(kFactor * twistK + bFactor * twistC) * derivTheta(dx[twistNode]));
    }

    if (!d_strictLumenBarrierEnabled.getValue() || !hasLumenProfile())
        return;

    const Real stiffness = std::max(d_strictLumenBarrierStiffness.getValue(), static_cast<Real>(0.0));
    const Real damping = std::max(d_strictLumenBarrierDamping.getValue(), static_cast<Real>(0.0));
    const Real activationMarginMm = std::max(d_strictLumenActivationMarginMm.getValue(), static_cast<Real>(0.0));
    const Real safetyMarginMm = std::max(d_strictLumenSafetyMarginMm.getValue(), static_cast<Real>(0.0));
    for (std::size_t i = 0; i < std::min(q.size(), dx.size()); ++i)
    {
        const Vec3 pointMm = solverToScene(coordCenter(q[i]));
        const Real barrierWeight = strictBarrierNodeWeight(i, q.size(), pointMm);
        if (barrierWeight <= static_cast<Real>(0.0))
            continue;

        Vec3 closestPointMm;
        Real projS = static_cast<Real>(0.0);
        Real clearanceMm = std::numeric_limits<Real>::infinity();
        Vec3 normalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (!sampleStrictLumenConstraint(i, pointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, normalM))
            continue;
        if (normalM.norm() <= kEps)
            continue;
        if (clearanceMm >= activationMarginMm)
            continue;

        const Vec3 dCenterM = derivCenter(dx[i]);
        const Vec3 normalDelta = projectorAxial(normalM, dCenterM);
        const Real velocityGate = strictBarrierVelocityGate(clearanceMm, activationMarginMm, safetyMarginMm);
        const Real penetrationDepthM = strictBarrierPenetrationDepthM(clearanceMm, safetyMarginMm);
        if (penetrationDepthM <= static_cast<Real>(0.0) && velocityGate <= static_cast<Real>(0.0))
            continue;
        const Vec3 dForceN = -barrierWeight * (
            (penetrationDepthM > static_cast<Real>(0.0) ? kFactor * stiffness : static_cast<Real>(0.0))
            + velocityGate * bFactor * damping) * normalDelta;
        addCenterScene(df[i], dForceN);
    }

    for (std::size_t i = 0; i + 1 < std::min(q.size(), dx.size()); ++i)
    {
        const Vec3 midpointMm = solverToScene(static_cast<Real>(0.5) * (coordCenter(q[i]) + coordCenter(q[i + 1])));
        const Real barrierWeight = strictBarrierEdgeWeight(i, q.size(), midpointMm);
        if (barrierWeight <= static_cast<Real>(0.0))
            continue;

        Vec3 closestPointMm;
        Real projS = static_cast<Real>(0.0);
        Real clearanceMm = std::numeric_limits<Real>::infinity();
        Vec3 normalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (!sampleStrictLumenConstraintForEdge(i, midpointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, normalM))
            continue;
        if (normalM.norm() <= kEps)
            continue;
        const Real edgeActivationMarginMm = std::min(activationMarginMm, safetyMarginMm + static_cast<Real>(0.25));
        if (clearanceMm >= edgeActivationMarginMm)
            continue;

        const Vec3 midpointDeltaM = static_cast<Real>(0.5) * (derivCenter(dx[i]) + derivCenter(dx[i + 1]));
        const Vec3 normalDelta = projectorAxial(normalM, midpointDeltaM);
        const Vec3 dForceN = -barrierWeight * (
            static_cast<Real>(0.65) * kFactor * stiffness
            + static_cast<Real>(0.70) * bFactor * damping) * normalDelta;
        addCenterScene(df[i], static_cast<Real>(0.5) * dForceN);
        addCenterScene(df[i + 1], static_cast<Real>(0.5) * dForceN);
    }
}

void ElasticRodGuidewireModel::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    VecDeriv& out = *df.beginEdit();
    const VecDeriv& dpos = dx.getValue();
    if (out.size() < dpos.size())
        out.resize(dpos.size());
    if (this->mstate != nullptr)
    {
        const VecCoord& q = this->mstate->read(sofa::core::vec_id::read_access::position)->getValue();
        if (m_cachedNodeCount != q.size())
            configureCoreFromData(q);
        refreshBendTwistCache(q);

        VecDeriv local(dpos.size());
        for (auto& d : local)
            zeroDeriv(d);
        const Real kFactor = mparams != nullptr ? static_cast<Real>(mparams->kFactor()) : static_cast<Real>(1.0);
        const Real bFactor = mparams != nullptr ? static_cast<Real>(mparams->bFactor()) : static_cast<Real>(0.0);
        if (d_useImplicitStretch.getValue())
            applyStretchDForce(q, dpos, local, kFactor, bFactor);
        applyBoundaryDForce(q, dpos, local, kFactor, bFactor);
        if (d_useImplicitBendTwist.getValue())
            applyBendTwistDForce(dpos, local, kFactor);
        for (std::size_t i = 0; i < local.size(); ++i)
        {
            for (unsigned int c = 0; c < 6; ++c)
                out[i][c] += local[i][c];
        }
    }
    df.endEdit();
}

void ElasticRodGuidewireModel::addBoundaryKToMatrix(const VecCoord& q, sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset) const
{
    if (matrix == nullptr || m_cachedNodeCount == 0)
        return;

    const Vec3 axis = ElasticRodCompatCore::safeNormalize(d_insertionDirection.getValue(), kZAxis);
    const Real axialK = d_proximalAxialStiffness.getValue();
    const Real lateralK = d_proximalLateralStiffness.getValue();
    const Real twistK = d_proximalAngularStiffness.getValue();
    const Real uu[3][3] = {
        {axis[0] * axis[0], axis[0] * axis[1], axis[0] * axis[2]},
        {axis[1] * axis[0], axis[1] * axis[1], axis[1] * axis[2]},
        {axis[2] * axis[0], axis[2] * axis[1], axis[2] * axis[2]},
    };

    const std::size_t nodeCount = useDynamicStrictWindows()
        ? std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount())
        : std::min<std::size_t>(
            static_cast<std::size_t>(d_pushNodeCount.getValue()),
            std::min<std::size_t>(m_cachedNodeCount, m_core.nodeCount()));
    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        const Real lateralWeight = boundaryPenaltyWeight(node, nodeCount);
        const Real axialWeight = axialDriveWeight(node, nodeCount);
        if (lateralWeight <= static_cast<Real>(0.0) && axialWeight <= static_cast<Real>(0.0))
            continue;

        for (unsigned int r = 0; r < 3; ++r)
        {
            for (unsigned int c = 0; c < 3; ++c)
            {
                const Real identity = r == c ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
                const unsigned int row = offset + static_cast<unsigned int>(6 * node + r);
                const unsigned int col = offset + static_cast<unsigned int>(6 * node + c);
                matrix->add(
                    row,
                    col,
                    -kFact * (
                        lateralWeight * lateralK * (identity - uu[r][c])
                        + axialWeight * axialK * uu[r][c]));
            }
        }
    }
    if (nodeCount > 0u)
    {
        const std::size_t twistNode = std::min(primaryBoundaryNodeIndex(), nodeCount - 1u);
        matrix->add(offset + static_cast<unsigned int>(6 * twistNode + 3u), offset + static_cast<unsigned int>(6 * twistNode + 3u), -kFact * twistK);
    }

    if (!d_strictLumenBarrierEnabled.getValue() || !hasLumenProfile())
        return;

    const Real stiffness = std::max(d_strictLumenBarrierStiffness.getValue(), static_cast<Real>(0.0));
    const Real activationMarginMm = std::max(d_strictLumenActivationMarginMm.getValue(), static_cast<Real>(0.0));
    const Real safetyMarginMm = std::max(d_strictLumenSafetyMarginMm.getValue(), static_cast<Real>(0.0));
    for (std::size_t node = 0; node < q.size(); ++node)
    {
        const Vec3 pointMm = solverToScene(coordCenter(q[node]));
        const Real barrierWeight = strictBarrierNodeWeight(node, q.size(), pointMm);
        if (barrierWeight <= static_cast<Real>(0.0))
            continue;

        Vec3 closestPointMm;
        Real projS = static_cast<Real>(0.0);
        Real clearanceMm = std::numeric_limits<Real>::infinity();
        Vec3 normalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (!sampleStrictLumenConstraint(node, pointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, normalM))
            continue;
        if (normalM.norm() <= kEps)
            continue;
        if (clearanceMm >= activationMarginMm)
            continue;
        if (strictBarrierPenetrationDepthM(clearanceMm, safetyMarginMm) <= static_cast<Real>(0.0))
            continue;

        for (unsigned int r = 0; r < 3; ++r)
        {
            for (unsigned int c = 0; c < 3; ++c)
            {
                const unsigned int row = offset + static_cast<unsigned int>(6 * node + r);
                const unsigned int col = offset + static_cast<unsigned int>(6 * node + c);
                matrix->add(row, col, -kFact * barrierWeight * stiffness * normalM[r] * normalM[c]);
            }
        }
    }

    for (std::size_t edge = 0; edge + 1 < q.size(); ++edge)
    {
        const Vec3 midpointMm = solverToScene(static_cast<Real>(0.5) * (coordCenter(q[edge]) + coordCenter(q[edge + 1])));
        const Real barrierWeight = strictBarrierEdgeWeight(edge, q.size(), midpointMm);
        if (barrierWeight <= static_cast<Real>(0.0))
            continue;

        Vec3 closestPointMm;
        Real projS = static_cast<Real>(0.0);
        Real clearanceMm = std::numeric_limits<Real>::infinity();
        Vec3 normalM(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (!sampleStrictLumenConstraintForEdge(edge, midpointMm, safetyMarginMm, closestPointMm, projS, clearanceMm, normalM))
            continue;
        if (normalM.norm() <= kEps)
            continue;
        const Real edgeActivationMarginMm = std::min(activationMarginMm, safetyMarginMm + static_cast<Real>(0.25));
        if (clearanceMm >= edgeActivationMarginMm)
            continue;

        for (unsigned int r = 0; r < 3; ++r)
        {
            for (unsigned int c = 0; c < 3; ++c)
            {
                const Real value = -static_cast<Real>(0.25) * static_cast<Real>(0.65) * kFact * barrierWeight * stiffness * normalM[r] * normalM[c];
                const unsigned int rowI = offset + static_cast<unsigned int>(6 * edge + r);
                const unsigned int rowJ = offset + static_cast<unsigned int>(6 * (edge + 1) + r);
                const unsigned int colI = offset + static_cast<unsigned int>(6 * edge + c);
                const unsigned int colJ = offset + static_cast<unsigned int>(6 * (edge + 1) + c);
                matrix->add(rowI, colI, value);
                matrix->add(rowI, colJ, value);
                matrix->add(rowJ, colI, value);
                matrix->add(rowJ, colJ, value);
            }
        }
    }
}

void ElasticRodGuidewireModel::addStretchKToMatrix(const VecCoord& q, sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset) const
{
    if (matrix == nullptr)
        return;
    const std::size_t edgeCount = std::min<std::size_t>(m_core.edgeCount(), q.size() > 0 ? q.size() - 1 : 0);
    for (std::size_t i = 0; i < edgeCount; ++i)
    {
        const Real weight = supportConstitutiveEdgeWeight(i);
        if (weight <= static_cast<Real>(0.0))
            continue;

        const Vec3 edgeM = coordCenter(q[i + 1]) - coordCenter(q[i]);
        const Real ell = edgeM.norm();
        if (ell <= static_cast<Real>(kEps))
            continue;
        const Vec3 dir = edgeM / ell;
        const Real k = effectiveAxialEA(i) / std::max(m_core.refLen()[i], static_cast<Real>(kEps));
        for (unsigned int r = 0; r < 3; ++r)
        {
            for (unsigned int c = 0; c < 3; ++c)
            {
                const Real kij = weight * k * dir[r] * dir[c];
                const Real value = -kFact * kij;
                const unsigned int rowI = offset + static_cast<unsigned int>(6 * i + r);
                const unsigned int rowJ = offset + static_cast<unsigned int>(6 * (i + 1) + r);
                const unsigned int colI = offset + static_cast<unsigned int>(6 * i + c);
                const unsigned int colJ = offset + static_cast<unsigned int>(6 * (i + 1) + c);
                matrix->add(rowI, colI, value);
                matrix->add(rowI, colJ, -value);
                matrix->add(rowJ, colI, -value);
                matrix->add(rowJ, colJ, value);
            }
        }
    }
}

void ElasticRodGuidewireModel::addBendTwistKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset) const
{
    if (matrix == nullptr)
        return;
    for (std::size_t i = 1; i + 1 < m_cachedNodeCount; ++i)
    {
        const Real weight = supportConstitutiveBlockWeight(i);
        if (weight <= static_cast<Real>(0.0))
            continue;
        if (i >= m_bendTwistBlocks.size())
            continue;
        const auto& block = m_bendTwistBlocks[i];
        if (!block.active)
            continue;

        for (unsigned int row = 0; row < kLocalDofCount; ++row)
        {
            const unsigned int rowNode = row / kActiveNodeDofCount;
            const unsigned int rowDof = row % kActiveNodeDofCount;
            const unsigned int rowIndex = offset + static_cast<unsigned int>(6 * block.nodes[rowNode] + rowDof);
            for (unsigned int col = 0; col < kLocalDofCount; ++col)
            {
                const unsigned int colNode = col / kActiveNodeDofCount;
                const unsigned int colDof = col % kActiveNodeDofCount;
                const unsigned int colIndex = offset + static_cast<unsigned int>(6 * block.nodes[colNode] + colDof);
                matrix->add(rowIndex, colIndex, -kFact * weight * blockMatrixValue(block, row, col));
            }
        }
    }
}

void ElasticRodGuidewireModel::addKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset)
{
    if (matrix == nullptr || this->mstate == nullptr)
        return;
    const VecCoord& q = this->mstate->read(sofa::core::vec_id::read_access::position)->getValue();
    if (m_cachedNodeCount != q.size())
        configureCoreFromData(q);
    refreshBendTwistCache(q);
    if (d_useImplicitStretch.getValue())
        addStretchKToMatrix(q, matrix, kFact, offset);
    addBoundaryKToMatrix(q, matrix, kFact, offset);
    if (d_useImplicitBendTwist.getValue())
        addBendTwistKToMatrix(matrix, kFact, offset);
}

SReal ElasticRodGuidewireModel::getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord& x) const
{
    const VecCoord& q = x.getValue();
    Real total = static_cast<Real>(0.0);
    const std::size_t edgeCount = std::min<std::size_t>(m_core.edgeCount(), q.size() > 0 ? q.size() - 1 : 0);
    for (std::size_t i = 0; i < edgeCount; ++i)
        total += stretchEnergySI(q[i], q[i + 1], i);
    for (std::size_t i = 1; i + 1 < q.size(); ++i)
        total += bendTwistEnergySI(q, i);
    total += boundaryEnergySI(q);
    return static_cast<SReal>(total);
}

int ElasticRodGuidewireModelClass = sofa::core::RegisterObject(
    "Reduced-state native elastic rod guidewire model using an SI Vec6d carrier state [x,y,z,theta,0,0]."
).add<ElasticRodGuidewireModel>();

} // namespace elastic_rod_guidewire
