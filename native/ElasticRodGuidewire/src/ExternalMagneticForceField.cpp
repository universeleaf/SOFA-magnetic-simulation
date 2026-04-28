#include <ElasticRodGuidewire/ExternalMagneticForceField.h>
#include <ElasticRodGuidewire/ElasticRodGuidewireModel.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/linearalgebra/BaseMatrix.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace elastic_rod_guidewire
{

namespace
{
using Vec3 = ExternalMagneticForceField::Vec3;
using Coord = ExternalMagneticForceField::Coord;
using Deriv = ExternalMagneticForceField::Deriv;
using Real = ExternalMagneticForceField::Real;
using FrameAxes = ElasticRodCompatCore::FrameAxes;

constexpr Real kEps = static_cast<Real>(1.0e-9);
constexpr Real kPi = static_cast<Real>(3.14159265358979323846);
constexpr Real kMmToM = static_cast<Real>(1.0e-3);
constexpr Real kMToMm = static_cast<Real>(1.0e3);
constexpr Real kAssistForceCapPerEdgeN = static_cast<Real>(0.15);
constexpr Real kAssistForceCapTotalN = static_cast<Real>(0.35);
const Vec3 kZAxis(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(1.0));

inline Vec3 safeNormalize(const Vec3& v, const Vec3& fallback)
{
    const Real n = v.norm();
    if (n <= kEps)
    {
        const Real nf = fallback.norm();
        return nf <= kEps ? kZAxis : fallback / nf;
    }
    return v / n;
}

inline Vec3 closestPointOnSegment(const Vec3& p, const Vec3& a, const Vec3& b, Real& u)
{
    const Vec3 ab = b - a;
    const Real ab2 = sofa::type::dot(ab, ab);
    u = ab2 <= kEps ? static_cast<Real>(0.0) : std::clamp(sofa::type::dot(p - a, ab) / ab2, static_cast<Real>(0.0), static_cast<Real>(1.0));
    return a + ab * u;
}

inline bool nearestTubeProjection(
    const ExternalMagneticForceField::VecVec3& tubeNodes,
    const std::vector<Real>& tubeCum,
    const Vec3& pointMm,
    Vec3& projectionMm,
    Real& projectionS,
    Real& distanceMm,
    Vec3* tangentMm = nullptr)
{
    if (tubeNodes.size() < 2 || tubeCum.size() != tubeNodes.size())
        return false;

    Real bestD2 = std::numeric_limits<Real>::max();
    std::size_t bestSeg = 0u;
    Real bestU = static_cast<Real>(0.0);
    Vec3 bestPoint = tubeNodes.front();
    for (std::size_t i = 0; i + 1 < tubeNodes.size(); ++i)
    {
        Real u = static_cast<Real>(0.0);
        const Vec3 proj = closestPointOnSegment(pointMm, tubeNodes[i], tubeNodes[i + 1], u);
        const Vec3 delta = pointMm - proj;
        const Real d2 = sofa::type::dot(delta, delta);
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestSeg = i;
            bestU = u;
            bestPoint = proj;
        }
    }

    if (bestD2 == std::numeric_limits<Real>::max())
        return false;

    projectionMm = bestPoint;
    projectionS = tubeCum[bestSeg] + bestU * (tubeCum[bestSeg + 1] - tubeCum[bestSeg]);
    distanceMm = std::sqrt(std::max(bestD2, static_cast<Real>(0.0)));
    if (tangentMm != nullptr)
        *tangentMm = safeNormalize(tubeNodes[bestSeg + 1] - tubeNodes[bestSeg], kZAxis);
    return true;
}

inline Vec3 orthogonalUnit(const Vec3& v)
{
    const Real ax = std::abs(v[0]);
    const Real ay = std::abs(v[1]);
    const Real az = std::abs(v[2]);
    const Vec3 ref = (ax <= ay && ax <= az)
        ? Vec3(static_cast<Real>(1.0), static_cast<Real>(0.0), static_cast<Real>(0.0))
        : ((ay <= az)
            ? Vec3(static_cast<Real>(0.0), static_cast<Real>(1.0), static_cast<Real>(0.0))
            : Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(1.0)));
    return safeNormalize(v.cross(ref), kZAxis);
}

inline Vec3 projectLateral(const Vec3& axis, const Vec3& value)
{
    return value - sofa::type::dot(value, axis) * axis;
}

inline Vec3 rotateToward(const Vec3& from, const Vec3& to, Real maxAngleRad)
{
    const Vec3 fromDir = safeNormalize(from, kZAxis);
    const Vec3 toDir = safeNormalize(to, fromDir);
    const Real cosAngle = std::clamp(sofa::type::dot(fromDir, toDir), static_cast<Real>(-1.0), static_cast<Real>(1.0));
    const Real angle = std::acos(cosAngle);
    if (angle <= std::max(maxAngleRad, static_cast<Real>(0.0)) + kEps)
        return toDir;

    Vec3 axis = fromDir.cross(toDir);
    if (axis.norm() <= kEps)
        axis = orthogonalUnit(fromDir);
    else
        axis /= axis.norm();

    const Real c = std::cos(maxAngleRad);
    const Real s = std::sin(maxAngleRad);
    return safeNormalize(fromDir * c + axis.cross(fromDir) * s, toDir);
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

inline Vec3 reconstructMagneticMoment(const FrameAxes& rest, const FrameAxes& current, const Vec3& brVector)
{
    return sofa::type::dot(rest.m1, brVector) * current.m1
        + sofa::type::dot(rest.m2, brVector) * current.m2
        + sofa::type::dot(rest.m3, brVector) * current.m3;
}

inline Vec3 buildTorqueAwareFieldDirection(const Vec3& momentDirection, const Vec3& targetDirection, Real minTorqueSin, Real* outTorqueSin)
{
    const Vec3 desired = safeNormalize(targetDirection, momentDirection);
    const Vec3 mDir = safeNormalize(momentDirection, desired);
    Vec3 tauDir = safeNormalize(mDir.cross(desired), orthogonalUnit(mDir));
    Vec3 bPerp = safeNormalize(tauDir.cross(mDir), orthogonalUnit(mDir));
    constexpr Real kParallelGain = static_cast<Real>(0.25);
    Vec3 bCandidate = safeNormalize(bPerp + kParallelGain * mDir, bPerp);

    Real torqueSin = std::clamp((mDir.cross(bCandidate)).norm(), static_cast<Real>(0.0), static_cast<Real>(1.0));
    const Real minSin = std::clamp(minTorqueSin, static_cast<Real>(0.0), static_cast<Real>(0.999));
    if (torqueSin < minSin)
    {
        const Vec3 perp2 = safeNormalize(mDir.cross(tauDir), bPerp);
        bCandidate = safeNormalize(bCandidate + (minSin - torqueSin) * perp2, bPerp);
        torqueSin = std::clamp((mDir.cross(bCandidate)).norm(), static_cast<Real>(0.0), static_cast<Real>(1.0));
    }

    if (outTorqueSin != nullptr)
        *outTorqueSin = torqueSin;
    return safeNormalize(bCandidate, desired);
}

inline Vec3 enforceGuidanceLateralSign(
    const Vec3& fieldDirection,
    const Vec3& forwardAxis,
    const Vec3& guidancePull,
    Real blendAlpha)
{
    const Vec3 forward = safeNormalize(forwardAxis, fieldDirection);
    Vec3 guidanceLateral = projectLateral(forward, guidancePull);
    const Real guidanceNorm = guidanceLateral.norm();
    if (guidanceNorm <= kEps)
        return safeNormalize(fieldDirection, forward);
    guidanceLateral /= guidanceNorm;

    const Vec3 currentField = safeNormalize(fieldDirection, forward);
    Vec3 fieldLateral = projectLateral(forward, currentField);
    Real fieldLateralNorm = fieldLateral.norm();
    if (fieldLateralNorm > kEps)
    {
        fieldLateral /= fieldLateralNorm;
        if (sofa::type::dot(fieldLateral, guidanceLateral) >= static_cast<Real>(0.0))
            return currentField;
    }

    const Real axial = sofa::type::dot(currentField, forward);
    const Real lateral = fieldLateralNorm > kEps
        ? fieldLateralNorm
        : std::sqrt(std::max(static_cast<Real>(1.0) - axial * axial, static_cast<Real>(0.0)));
    const Vec3 signCorrected = safeNormalize(
        axial * forward + std::max(lateral, static_cast<Real>(0.20)) * guidanceLateral,
        guidanceLateral);
    const Real alpha = std::clamp(blendAlpha, static_cast<Real>(0.0), static_cast<Real>(1.0));
    return safeNormalize(
        (static_cast<Real>(1.0) - alpha) * currentField + alpha * signCorrected,
        signCorrected);
}

}

ExternalMagneticForceField::ExternalMagneticForceField()
    : d_tubeNodes(initData(&d_tubeNodes, "tubeNodes", "Centerline points used by the native magnetic steering force field."))
    , d_brVector(initData(&d_brVector, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)), "brVector", "Reference remanent magnetic field vector."))
    , d_baVectorRef(initData(&d_baVectorRef, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(1.0)), "baVectorRef", "Fallback applied field direction when tubeNodes are unavailable."))
    , d_muZero(initData(&d_muZero, static_cast<Real>(1.0), "muZero", "Magnetic scaling denominator used by the original externalMagneticForce formulation."))
    , d_rodRadius(initData(&d_rodRadius, static_cast<Real>(0.35), "rodRadius", "Guidewire contact outer radius in scene length units (mm in this scene)."))
    , d_magneticCoreRadiusMm(initData(&d_magneticCoreRadiusMm, static_cast<Real>(0.20), "magneticCoreRadiusMm", "Magnetic/mechanical core radius used for magnetic moment area in mm."))
    , d_magneticEdgeCount(initData(&d_magneticEdgeCount, 5u, "magneticEdgeCount", "Number of distal magnetic edges; semantically matches elasticRod.cpp."))
    , d_lookAheadDistance(initData(&d_lookAheadDistance, static_cast<Real>(4.0), "lookAheadDistance", "Arc-length look-ahead distance on the centerline in scene units (mm)."))
    , d_recoveryLookAheadDistance(initData(&d_recoveryLookAheadDistance, static_cast<Real>(5.0), "recoveryLookAheadDistance", "Shorter strict recovery look-ahead distance in scene units (mm)."))
    , d_fieldSmoothingAlpha(initData(&d_fieldSmoothingAlpha, static_cast<Real>(0.0), "fieldSmoothingAlpha", "Compatibility field; strict path uses the nearest-tangent direction directly."))
    , d_maxFieldTurnAngleDeg(initData(&d_maxFieldTurnAngleDeg, static_cast<Real>(0.0), "maxFieldTurnAngleDeg", "Compatibility field; strict path uses the nearest-tangent direction directly."))
    , d_fieldRampTime(initData(&d_fieldRampTime, static_cast<Real>(0.20), "fieldRampTime", "Initial ramp duration in seconds to avoid start-up impulses."))
    , d_minTorqueSin(initData(&d_minTorqueSin, static_cast<Real>(0.22), "minTorqueSin", "Minimum |m x B| sine enforced when building the applied field direction."))
    , d_lateralForceScale(initData(&d_lateralForceScale, static_cast<Real>(0.0), "lateralForceScale", "Non-strict: bounded distal steering assist scale. Strict: small lateral tip-centering force cap in N."))
    , d_entryStraightDistance(initData(&d_entryStraightDistance, static_cast<Real>(0.0), "entryStraightDistance", "Distance from the vessel entry over which strict magnetic steering stays released along the entry axis before progressively engaging the target direction."))
    , d_entrySteeringReleaseDistance(initData(&d_entrySteeringReleaseDistance, static_cast<Real>(0.0), "entrySteeringReleaseDistance", "Additional blend distance after entryStraightDistance over which strict magnetic steering ramps from released to fully engaged."))
    , d_bendLookAheadDistance(initData(&d_bendLookAheadDistance, static_cast<Real>(10.0), "bendLookAheadDistance", "Far arc-length window in mm used by strict bend preview scheduling."))
    , d_bendNearWindowDistance(initData(&d_bendNearWindowDistance, static_cast<Real>(4.0), "bendNearWindowDistance", "Near arc-length window in mm used by strict bend preview scheduling."))
    , d_bendTurnMediumDeg(initData(&d_bendTurnMediumDeg, static_cast<Real>(12.0), "bendTurnMediumDeg", "Upcoming turn angle threshold in degrees above which strict bend preview starts ramping in."))
    , d_bendTurnHighDeg(initData(&d_bendTurnHighDeg, static_cast<Real>(25.0), "bendTurnHighDeg", "Upcoming turn angle threshold in degrees treated as a large bend for strict bend preview scheduling."))
    , d_fieldScaleStraight(initData(&d_fieldScaleStraight, static_cast<Real>(0.55), "fieldScaleStraight", "Strict straight-segment magnetic field scale multiplier."))
    , d_fieldScaleBend(initData(&d_fieldScaleBend, static_cast<Real>(1.15), "fieldScaleBend", "Strict large-bend magnetic field scale multiplier."))
    , d_recenterClearanceMm(initData(&d_recenterClearanceMm, static_cast<Real>(0.45), "recenterClearanceMm", "Strict clearance threshold in mm below which magnetic recentring can activate."))
    , d_recenterOffsetMm(initData(&d_recenterOffsetMm, static_cast<Real>(0.60), "recenterOffsetMm", "Strict tip offset threshold in mm above which magnetic recentring can activate."))
    , d_recenterBlend(initData(&d_recenterBlend, static_cast<Real>(0.70), "recenterBlend", "Blend weight toward the local forward tangent when strict magnetic recentring activates."))
    , d_headStretchReliefStart(initData(&d_headStretchReliefStart, static_cast<Real>(0.0035), "headStretchReliefStart", "Strict head-stretch ratio above which magnetic anti-kink relief starts attenuating field torque."))
    , d_headStretchReliefFull(initData(&d_headStretchReliefFull, static_cast<Real>(0.0085), "headStretchReliefFull", "Strict head-stretch ratio above which magnetic anti-kink relief reaches full strength."))
    , d_headStretchFieldScaleFloor(initData(&d_headStretchFieldScaleFloor, static_cast<Real>(0.35), "headStretchFieldScaleFloor", "Minimum strict magnetic field-scale multiplier retained under full head-stretch relief."))
    , d_strictPhysicalTorqueOnly(initData(&d_strictPhysicalTorqueOnly, false, "strictPhysicalTorqueOnly", "If true, use the commanded field direction directly and disable torque-floor steering helpers."))
    , d_externalFieldScale(initData(&d_externalFieldScale, static_cast<Real>(1.0), "externalFieldScale", "External magnetic scale gate used by strict release control; 0 disables the field and resets the native ramp."))
    , d_externalControlDt(initData(&d_externalControlDt, static_cast<Real>(0.0), "externalControlDt", "Optional external control dt in seconds. When > 0, GUI wall-clock control time overrides solver dt for magnetic ramp and smoothing."))
    , d_useExternalTargetDirection(initData(&d_useExternalTargetDirection, false, "useExternalTargetDirection", "Whether to override the internally tracked target field direction from Python/RL control."))
    , d_externalTargetDirection(initData(&d_externalTargetDirection, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(1.0)), "externalTargetDirection", "Optional externally commanded magnetic field direction used when useExternalTargetDirection is true."))
    , d_externalSurfaceClearanceMm(initData(&d_externalSurfaceClearanceMm, std::numeric_limits<Real>::infinity(), "externalSurfaceClearanceMm", "Exact surface clearance from the Python controller in mm; used to keep strict steering/contact recovery aligned with the real vessel wall."))
    , d_externalSurfaceContactActive(initData(&d_externalSurfaceContactActive, false, "externalSurfaceContactActive", "Whether the Python controller currently detects exact surface contact on the tip/head."))
    , d_debugTargetPoint(initData(&d_debugTargetPoint, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)), "debugTargetPoint", "Debug forward-nearest centerline target point currently tracked by the native magnetic component."))
    , d_debugLookAheadPoint(initData(&d_debugLookAheadPoint, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)), "debugLookAheadPoint", "Debug smoothed look-ahead point used internally to filter the applied field direction."))
    , d_debugBaVector(initData(&d_debugBaVector, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(1.0)), "debugBaVector", "Debug applied magnetic field direction selected from the nearest centerline tangent."))
    , d_debugForceVector(initData(&d_debugForceVector, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)), "debugForceVector", "Debug equivalent distal magnetic force direction for visualisation."))
    , d_debugTorqueVector(initData(&d_debugTorqueVector, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)), "debugTorqueVector", "Debug total magnetic torque vector in N.m for the active distal magnetic segment."))
    , d_debugMagneticMomentVector(initData(&d_debugMagneticMomentVector, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)), "debugMagneticMomentVector", "Debug aggregate distal magnetic moment reconstructed from the DER material frames."))
    , d_debugTorqueSin(initData(&d_debugTorqueSin, static_cast<Real>(0.0), "debugTorqueSin", "Debug sine of the angle between the distal magnetic moment and the applied field direction."))
    , d_debugAssistForceVector(initData(&d_debugAssistForceVector, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)), "debugAssistForceVector", "Debug bounded lateral magnetic centering force in N."))
    , d_strictInLumenMode(initData(&d_strictInLumenMode, false, "strictInLumenMode", "If true, clamp steering assist so it never points outward from the centerline-defined lumen."))
    , d_debugOutwardAssistComponentN(initData(&d_debugOutwardAssistComponentN, static_cast<Real>(0.0), "debugOutwardAssistComponentN", "Residual outward assist component in N after strict in-lumen clamping."))
    , d_debugDistalTangentFieldAngleDeg(initData(&d_debugDistalTangentFieldAngleDeg, static_cast<Real>(0.0), "debugDistalTangentFieldAngleDeg", "Angle between distal tangent and applied field in degrees."))
    , d_debugUpcomingTurnDeg(initData(&d_debugUpcomingTurnDeg, static_cast<Real>(0.0), "debugUpcomingTurnDeg", "Previewed upcoming centerline turn angle in degrees used by strict bend scheduling."))
    , d_debugBendSeverity(initData(&d_debugBendSeverity, static_cast<Real>(0.0), "debugBendSeverity", "Strict previewed bend severity in [0,1]."))
    , d_debugScheduledFieldScale(initData(&d_debugScheduledFieldScale, static_cast<Real>(1.0), "debugScheduledFieldScale", "Strict final magnetic field scale after all gates and ramping are applied."))
    , d_debugScheduledFieldScaleBase(initData(&d_debugScheduledFieldScaleBase, static_cast<Real>(1.0), "debugScheduledFieldScaleBase", "Strict scheduled magnetic field scale before strict steering gates are applied."))
    , d_debugStrictSteeringNeedAlpha(initData(&d_debugStrictSteeringNeedAlpha, static_cast<Real>(0.0), "debugStrictSteeringNeedAlpha", "Strict magnetic steering-need gate in [0,1] before field application."))
    , d_debugEntryReleaseAlpha(initData(&d_debugEntryReleaseAlpha, static_cast<Real>(1.0), "debugEntryReleaseAlpha", "Strict entry-release gate in [0,1] applied to magnetic steering after the straight entry corridor."))
    , d_debugRecenteringAlpha(initData(&d_debugRecenteringAlpha, static_cast<Real>(0.0), "debugRecenteringAlpha", "Strict magnetic recentering blend alpha in [0,1]."))
    , m_filteredBaVector(kZAxis)
    , m_appliedBaVector(kZAxis)
{
}

void ExternalMagneticForceField::init()
{
    Inherit::init();
    if (this->mstate == nullptr)
    {
        msg_error() << "ExternalMagneticForceField requires a Vec6d MechanicalObject in the same node.";
        return;
    }
    m_rodModel = this->getContext() != nullptr
        ? this->getContext()->template get<ElasticRodGuidewireModel>(sofa::core::objectmodel::BaseContext::Local)
        : nullptr;
    rebuildTubeArc();
    if (m_rodModel != nullptr)
        d_magneticEdgeCount.setValue(static_cast<unsigned int>(m_rodModel->magneticEdgeCount()));
    msg_info() << "ExternalMagneticForceField initialized: magneticEdgeCount=" << d_magneticEdgeCount.getValue()
               << ", contactRadiusMm=" << d_rodRadius.getValue()
               << ", magneticCoreRadiusMm=" << d_magneticCoreRadiusMm.getValue()
               << ", fieldRampTime=" << d_fieldRampTime.getValue();
}

void ExternalMagneticForceField::rebuildTubeArc()
{
    const VecVec3& tubeNodes = d_tubeNodes.getValue();
    m_tubeCum.assign(tubeNodes.size(), static_cast<Real>(0.0));
    for (std::size_t i = 1; i < tubeNodes.size(); ++i)
        m_tubeCum[i] = m_tubeCum[i - 1] + (tubeNodes[i] - tubeNodes[i - 1]).norm();
    m_hasLastTargetArcS = false;
    m_lastTargetArcS = static_cast<Real>(0.0);
    m_lastLocalForwardTangent = kZAxis;
    m_lastUpcomingTurnDeg = static_cast<Real>(0.0);
    m_lastBendSeverity = static_cast<Real>(0.0);
}

Vec3 ExternalMagneticForceField::interpolateTubePoint(Real s) const
{
    const VecVec3& tubeNodes = d_tubeNodes.getValue();
    if (tubeNodes.empty())
        return Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    if (tubeNodes.size() == 1 || m_tubeCum.size() != tubeNodes.size())
        return tubeNodes.front();

    const Real sClamped = std::clamp(s, static_cast<Real>(0.0), m_tubeCum.back());
    auto it = std::upper_bound(m_tubeCum.begin(), m_tubeCum.end(), sClamped);
    if (it == m_tubeCum.begin())
        return tubeNodes.front();
    if (it == m_tubeCum.end())
        return tubeNodes.back();

    const std::size_t i = static_cast<std::size_t>(std::distance(m_tubeCum.begin(), it) - 1);
    const Real ds = std::max(m_tubeCum[i + 1] - m_tubeCum[i], kEps);
    const Real alpha = std::clamp((sClamped - m_tubeCum[i]) / ds, static_cast<Real>(0.0), static_cast<Real>(1.0));
    return tubeNodes[i] * (static_cast<Real>(1.0) - alpha) + tubeNodes[i + 1] * alpha;
}

Vec3 ExternalMagneticForceField::interpolateTubeTangent(Real s, Real sampleDs, const Vec3& fallback) const
{
    const VecVec3& tubeNodes = d_tubeNodes.getValue();
    if (tubeNodes.size() < 2 || m_tubeCum.size() != tubeNodes.size())
        return fallback;

    const Real clampedDs = std::max(sampleDs, static_cast<Real>(0.25));
    const Real sBack = std::clamp(s - clampedDs, static_cast<Real>(0.0), m_tubeCum.back());
    const Real sFront = std::clamp(s + clampedDs, static_cast<Real>(0.0), m_tubeCum.back());
    if (sFront > sBack + kEps)
        return safeNormalize(interpolateTubePoint(sFront) - interpolateTubePoint(sBack), fallback);

    auto upperIt = std::upper_bound(m_tubeCum.begin(), m_tubeCum.end(), std::clamp(s, static_cast<Real>(0.0), m_tubeCum.back()));
    std::size_t idx = static_cast<std::size_t>(std::distance(m_tubeCum.begin(), upperIt));
    if (idx >= tubeNodes.size())
        idx = tubeNodes.size() - 1u;
    if (idx == 0u)
        idx = 1u;
    return safeNormalize(tubeNodes[idx] - tubeNodes[idx - 1u], fallback);
}

Vec3 ExternalMagneticForceField::computeLookAheadTargetDirection(const VecCoord& positions, Vec3& targetPoint, Vec3* lookAheadPoint) const
{
    const VecVec3& tubeNodes = d_tubeNodes.getValue();
    const bool strictPhysicalTorqueOnly = d_strictPhysicalTorqueOnly.getValue();
    m_lastStrictEntrySteeringAlpha = static_cast<Real>(1.0);
    m_lastLocalForwardTangent = safeNormalize(d_baVectorRef.getValue(), kZAxis);
    const Vec3 tipPos = kMToMm * coordCenter(positions.back());
    const Vec3 tipForward = positions.size() >= 2
        ? safeNormalize(coordCenter(positions.back()) - coordCenter(positions[positions.size() - 2]), safeNormalize(d_baVectorRef.getValue(), kZAxis))
        : safeNormalize(d_baVectorRef.getValue(), kZAxis);
    m_lastLocalForwardTangent = tipForward;

    if (tubeNodes.size() < 2 || m_tubeCum.size() != tubeNodes.size())
    {
        targetPoint = tipPos;
        if (lookAheadPoint != nullptr)
            *lookAheadPoint = targetPoint;
        return tipForward;
    }

    Vec3 nearestTangent = tipForward;
    Real bestS = static_cast<Real>(0.0);
    Real tipOffsetMm = static_cast<Real>(0.0);
    Vec3 nearestProjection = tubeNodes.front();
    if (!nearestTubeProjection(tubeNodes, m_tubeCum, tipPos, nearestProjection, bestS, tipOffsetMm, &nearestTangent))
    {
        targetPoint = tipPos;
        if (lookAheadPoint != nullptr)
            *lookAheadPoint = targetPoint;
        return tipForward;
    }
    targetPoint = nearestProjection;

    Real minLumenClearanceMm = std::numeric_limits<Real>::infinity();
    unsigned int barrierActiveNodes = 0u;
    if (m_rodModel != nullptr)
    {
        minLumenClearanceMm = m_rodModel->d_debugMinLumenClearanceMm.getValue();
        barrierActiveNodes = m_rodModel->d_debugBarrierActiveNodeCount.getValue();
    }
    const Real externalSurfaceClearanceMm = d_externalSurfaceClearanceMm.getValue();
    const bool externalSurfaceContactActive = d_externalSurfaceContactActive.getValue();
    if (std::isfinite(externalSurfaceClearanceMm))
        minLumenClearanceMm = std::min(minLumenClearanceMm, externalSurfaceClearanceMm);
    if (externalSurfaceContactActive)
        barrierActiveNodes = std::max(barrierActiveNodes, 1u);

    const Real baseLookAhead = std::max(d_lookAheadDistance.getValue(), static_cast<Real>(0.5));
    const Real recoveryLookAhead = std::clamp(
        d_recoveryLookAheadDistance.getValue(),
        static_cast<Real>(0.5),
        baseLookAhead);
    const Real steeringAngleDeg = std::acos(std::clamp(sofa::type::dot(tipForward, nearestTangent), static_cast<Real>(-1.0), static_cast<Real>(1.0))) * static_cast<Real>(180.0) / kPi;
    const Real steeringRecoveryAlpha = std::clamp(
        (steeringAngleDeg - static_cast<Real>(12.0)) / static_cast<Real>(28.0),
        static_cast<Real>(0.0),
        static_cast<Real>(1.0));
    const Real offsetRecoveryAlpha = std::clamp(
        (tipOffsetMm - static_cast<Real>(0.6)) / static_cast<Real>(1.4),
        static_cast<Real>(0.0),
        static_cast<Real>(1.0));
    const Real clearanceRecoveryAlpha = std::isfinite(minLumenClearanceMm)
        ? std::clamp(
            (static_cast<Real>(0.8) - minLumenClearanceMm) / static_cast<Real>(0.8),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0))
        : static_cast<Real>(0.0);
    Real recoveryAlpha = barrierActiveNodes > 0u ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
    recoveryAlpha = std::max(recoveryAlpha, clearanceRecoveryAlpha);
    recoveryAlpha = std::max(
        recoveryAlpha,
        std::max(
            std::min(steeringRecoveryAlpha, offsetRecoveryAlpha),
            static_cast<Real>(0.40) * steeringRecoveryAlpha));
    const Real minForwardMm = std::max(static_cast<Real>(0.05), static_cast<Real>(0.25) * d_rodRadius.getValue());

    if (strictPhysicalTorqueOnly)
    {
        Real steeringProgressWeightedSum = static_cast<Real>(0.0);
        Real steeringProgressWeightTotal = static_cast<Real>(0.0);
        Real steeringProgressFrontS = bestS;
        const std::size_t distalNodeCount = std::min<std::size_t>(
            positions.size(),
            std::max<std::size_t>(static_cast<std::size_t>(d_magneticEdgeCount.getValue()) + 1u, 2u));
        const std::size_t distalNodeStart = positions.size() - distalNodeCount;
        for (std::size_t nodeIndex = distalNodeStart; nodeIndex < positions.size(); ++nodeIndex)
        {
            Vec3 nodeProjection = tipPos;
            Vec3 nodeTangent = nearestTangent;
            Real nodeS = static_cast<Real>(0.0);
            Real nodeOffsetMm = static_cast<Real>(0.0);
            if (
                nearestTubeProjection(
                    tubeNodes,
                    m_tubeCum,
                    kMToMm * coordCenter(positions[nodeIndex]),
                    nodeProjection,
                    nodeS,
                    nodeOffsetMm,
                    &nodeTangent)
            )
            {
                const Real localAlpha = distalNodeCount > 1u
                    ? static_cast<Real>(nodeIndex - distalNodeStart) / static_cast<Real>(distalNodeCount - 1u)
                    : static_cast<Real>(1.0);
                const Real weight = static_cast<Real>(0.35) + static_cast<Real>(0.65) * localAlpha * localAlpha;
                steeringProgressWeightedSum += weight * nodeS;
                steeringProgressWeightTotal += weight;
                steeringProgressFrontS = std::max(steeringProgressFrontS, nodeS);
            }
        }
        const Real steeringProgressS = steeringProgressWeightTotal > kEps
            ? std::max(
                steeringProgressFrontS - static_cast<Real>(0.35) * minForwardMm,
                steeringProgressWeightedSum / steeringProgressWeightTotal)
            : bestS;
        const Real clampedSteeringProgressS = std::clamp(steeringProgressS, static_cast<Real>(0.0), m_tubeCum.back());
        const Real markerProgressS = std::clamp(
            std::max(bestS, steeringProgressFrontS),
            static_cast<Real>(0.0),
            m_tubeCum.back());

        const Real prevUpcomingTurnDeg = m_lastUpcomingTurnDeg;
        const Real prevBendSeverity = m_lastBendSeverity;

        const Real nextTargetS = std::min(markerProgressS + kEps, m_tubeCum.back());
        auto upperIt = std::upper_bound(m_tubeCum.begin(), m_tubeCum.end(), nextTargetS);
        std::size_t forwardIndex = static_cast<std::size_t>(std::distance(m_tubeCum.begin(), upperIt));
        if (forwardIndex >= tubeNodes.size())
            forwardIndex = tubeNodes.size() - 1;
        if (forwardIndex == 0u && tubeNodes.size() > 1u)
            forwardIndex = 1u;
        targetPoint = tubeNodes[forwardIndex];
        const Real targetArcS = m_tubeCum[forwardIndex];
        const Real steeringArcS = std::clamp(clampedSteeringProgressS + minForwardMm, static_cast<Real>(0.0), m_tubeCum.back());
        Vec3 localForwardTangent = interpolateTubeTangent(
            steeringArcS,
            std::max(static_cast<Real>(0.5), static_cast<Real>(0.35) * baseLookAhead),
            nearestTangent);

        const Real bendLookAheadMm = std::max(d_bendLookAheadDistance.getValue(), minForwardMm);
        const Real bendNearWindowMm = std::clamp(
            d_bendNearWindowDistance.getValue(),
            minForwardMm,
            bendLookAheadMm);
        const Real bendNearS = std::clamp(clampedSteeringProgressS + bendNearWindowMm, static_cast<Real>(0.0), m_tubeCum.back());
        const Real bendFarS = std::clamp(clampedSteeringProgressS + bendLookAheadMm, bendNearS, m_tubeCum.back());
        const Vec3 bendNearTangent = interpolateTubeTangent(
            bendNearS,
            std::max(static_cast<Real>(0.5), static_cast<Real>(0.25) * bendNearWindowMm),
            localForwardTangent);
        const Vec3 bendFarTangent = interpolateTubeTangent(
            bendFarS,
            std::max(static_cast<Real>(0.5), static_cast<Real>(0.25) * std::max(bendLookAheadMm - bendNearWindowMm, bendNearWindowMm)),
            bendNearTangent);

        const Real upcomingTurnDeg = std::acos(std::clamp(sofa::type::dot(bendNearTangent, bendFarTangent), static_cast<Real>(-1.0), static_cast<Real>(1.0))) * static_cast<Real>(180.0) / kPi;
        const Real bendTurnMediumDeg = std::max(d_bendTurnMediumDeg.getValue(), static_cast<Real>(0.0));
        const Real bendTurnHighDeg = std::max(d_bendTurnHighDeg.getValue(), bendTurnMediumDeg + static_cast<Real>(1.0e-3));
        const Real bendSeverityRaw = std::clamp(
            (upcomingTurnDeg - bendTurnMediumDeg) / (bendTurnHighDeg - bendTurnMediumDeg),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        // Keep some steering memory through the bend instead of immediately
        // relaxing back to the nearest local tangent as soon as the instantaneous
        // preview angle drops for one or two steps.
        const bool previewHoldActive = (
            barrierActiveNodes > 0u
            || externalSurfaceContactActive
            || tipOffsetMm > static_cast<Real>(0.75)
            || (std::isfinite(minLumenClearanceMm) && minLumenClearanceMm < static_cast<Real>(0.55)));
        const Real bendMemoryDecay = previewHoldActive
            ? static_cast<Real>(0.96)
            : static_cast<Real>(0.88);
        const Real bendSeverity = std::max(
            bendSeverityRaw,
            std::clamp(bendMemoryDecay * prevBendSeverity, static_cast<Real>(0.0), static_cast<Real>(1.0)));
        const Real heldUpcomingTurnDeg = std::max(
            upcomingTurnDeg,
            bendMemoryDecay * prevUpcomingTurnDeg);
        const Real previewAlpha = std::max(recoveryAlpha, bendSeverity);
        const Real lookAheadDistanceMm = (static_cast<Real>(1.0) - previewAlpha) * baseLookAhead + previewAlpha * recoveryLookAhead;
        const Real lookAheadArcS = std::min(steeringArcS + lookAheadDistanceMm, m_tubeCum.back());
        const Vec3 lookAheadPointValue = interpolateTubePoint(lookAheadArcS);
        if (lookAheadPoint != nullptr)
            *lookAheadPoint = lookAheadPointValue;
        m_lastTargetArcS = targetArcS;
        m_hasLastTargetArcS = true;
        m_lastLocalForwardTangent = localForwardTangent;
        m_lastUpcomingTurnDeg = heldUpcomingTurnDeg;
        m_lastBendSeverity = bendSeverity;

        // In strict mode the target marker remains the first unvisited centerline
        // point, but the applied field direction must stay forward-tangential.
        // Use the front-most tip progress for the marker, but gate steering with
        // the magnetic-head average progress so the head does not kink while most
        // of it is still trapped in the straight entry corridor.
        const Real straightDistanceMm = std::max(d_entryStraightDistance.getValue(), static_cast<Real>(0.0));
        const Real releaseSpanMm = std::max(d_entrySteeringReleaseDistance.getValue(), static_cast<Real>(0.0));
        Real entrySteeringAlpha = static_cast<Real>(1.0);
        if (clampedSteeringProgressS <= straightDistanceMm)
        {
            entrySteeringAlpha = static_cast<Real>(0.0);
        }
        else if (releaseSpanMm > kEps)
        {
            const Real u = std::clamp(
                (clampedSteeringProgressS - straightDistanceMm) / releaseSpanMm,
                static_cast<Real>(0.0),
                static_cast<Real>(1.0));
            entrySteeringAlpha = u * u * (static_cast<Real>(3.0) - static_cast<Real>(2.0) * u);
        }
        m_lastStrictEntrySteeringAlpha = entrySteeringAlpha;
        // In strict mode the applied field should eventually follow the local
        // lumen tangent, but not while the magnetic head is still emerging from
        // the straight entry corridor. Blend from the entry axis to the local
        // tangent using the physical release distance instead of snapping the
        // full steering torque on at t=0.
        const Vec3 entryAxis = safeNormalize(d_baVectorRef.getValue(), nearestTangent);
        const Real previewBlendGate = std::clamp(
            std::max(
                bendSeverity,
                static_cast<Real>(0.45) * recoveryAlpha),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        const Real previewBlend = entrySteeringAlpha
            * std::clamp(
                static_cast<Real>(0.40) + static_cast<Real>(0.52) * previewBlendGate,
                static_cast<Real>(0.0),
                static_cast<Real>(0.92))
            * previewBlendGate;
        // In stronger bends the earlier blend stayed too close to the local
        // tangent, so the field would start turning but then fail to keep the
        // distal head committed into the bend once contact began to build.
        // Bias the strict preview further toward the far tangent so the desired
        // field keeps looking into the bend rather than snapping back toward
        // the local straight-ahead direction.
        const Vec3 previewTargetTangent = safeNormalize(
            (static_cast<Real>(1.0) - static_cast<Real>(0.60) * previewBlendGate) * bendNearTangent
                + static_cast<Real>(0.60) * previewBlendGate * bendFarTangent,
            bendNearTangent);
        const Vec3 guidedTangent = previewBlend <= kEps
            ? nearestTangent
            : safeNormalize(
                (static_cast<Real>(1.0) - previewBlend) * nearestTangent
                    + previewBlend * previewTargetTangent,
                nearestTangent);
        const Vec3 appliedDirection = entrySteeringAlpha <= kEps
            ? entryAxis
            : safeNormalize(
                (static_cast<Real>(1.0) - entrySteeringAlpha) * entryAxis
                    + entrySteeringAlpha * guidedTangent,
                nearestTangent);
        m_lastLocalForwardTangent = guidedTangent;
        return appliedDirection;
    }

    const Real lookAheadDistanceMm = (static_cast<Real>(1.0) - recoveryAlpha) * baseLookAhead + recoveryAlpha * recoveryLookAhead;
    Real bestForwardDeltaS = std::numeric_limits<Real>::max();
    Real bestForwardD2 = std::numeric_limits<Real>::max();
    Real forwardNearestS = std::min(bestS + minForwardMm, m_tubeCum.back());
    for (std::size_t i = 0; i + 1 < tubeNodes.size(); ++i)
    {
        Real u = static_cast<Real>(0.0);
        const Vec3 projection = closestPointOnSegment(tipPos, tubeNodes[i], tubeNodes[i + 1], u);
        const Real projectionS = m_tubeCum[i] + u * (m_tubeCum[i + 1] - m_tubeCum[i]);
        if (projectionS + kEps < bestS)
            continue;

        const Vec3 delta = tipPos - projection;
        const Real d2 = sofa::type::dot(delta, delta);
        const Real forwardDeltaS = projectionS - bestS;
        if (forwardDeltaS + kEps < minForwardMm)
            continue;
        if (
            forwardDeltaS < bestForwardDeltaS - kEps
            || (std::abs(forwardDeltaS - bestForwardDeltaS) <= kEps && d2 < bestForwardD2)
        )
        {
            bestForwardDeltaS = forwardDeltaS;
            bestForwardD2 = d2;
            forwardNearestS = projectionS;
        }
    }

    const bool hasForwardTarget = bestForwardD2 < std::numeric_limits<Real>::max();
    Real targetS = hasForwardTarget ? forwardNearestS : std::min(bestS + minForwardMm, m_tubeCum.back());
    if (m_hasLastTargetArcS)
    {
        const Real maxForwardJumpMm = std::max(static_cast<Real>(1.0), static_cast<Real>(0.5) * lookAheadDistanceMm);
        targetS = std::min(targetS, std::min(m_lastTargetArcS + maxForwardJumpMm, m_tubeCum.back()));
    }
    targetS = std::clamp(targetS, std::min(bestS + minForwardMm, m_tubeCum.back()), m_tubeCum.back());
    targetPoint = interpolateTubePoint(targetS);
    const Vec3 lookAheadPointValue = interpolateTubePoint(std::min(targetS + lookAheadDistanceMm, m_tubeCum.back()));
    m_lastTargetArcS = targetS;
    m_hasLastTargetArcS = true;
    if (lookAheadPoint != nullptr)
        *lookAheadPoint = lookAheadPointValue;
    const Vec3 forwardDir = safeNormalize(targetPoint - tipPos, tipForward);
    const Vec3 lookAheadDir = safeNormalize(lookAheadPointValue - tipPos, forwardDir);
    const Real prevUpcomingTurnDeg = m_lastUpcomingTurnDeg;
    const Real prevBendSeverity = m_lastBendSeverity;
    const Real steeringArcS = std::clamp(bestS + minForwardMm, static_cast<Real>(0.0), m_tubeCum.back());
    const Vec3 localForwardTangent = interpolateTubeTangent(
        steeringArcS,
        std::max(static_cast<Real>(0.5), static_cast<Real>(0.35) * baseLookAhead),
        nearestTangent);
    const Real bendLookAheadMm = std::max(d_bendLookAheadDistance.getValue(), minForwardMm);
    const Real bendNearWindowMm = std::clamp(
        d_bendNearWindowDistance.getValue(),
        minForwardMm,
        bendLookAheadMm);
    const Real bendNearS = std::clamp(bestS + bendNearWindowMm, static_cast<Real>(0.0), m_tubeCum.back());
    const Real bendFarS = std::clamp(bestS + bendLookAheadMm, bendNearS, m_tubeCum.back());
    const Vec3 bendNearTangent = interpolateTubeTangent(
        bendNearS,
        std::max(static_cast<Real>(0.5), static_cast<Real>(0.25) * bendNearWindowMm),
        localForwardTangent);
    const Vec3 bendFarTangent = interpolateTubeTangent(
        bendFarS,
        std::max(static_cast<Real>(0.5), static_cast<Real>(0.25) * std::max(bendLookAheadMm - bendNearWindowMm, bendNearWindowMm)),
        bendNearTangent);
    const Real upcomingTurnDeg = std::acos(std::clamp(
        sofa::type::dot(bendNearTangent, bendFarTangent),
        static_cast<Real>(-1.0),
        static_cast<Real>(1.0))) * static_cast<Real>(180.0) / kPi;
    const Real bendTurnMediumDeg = std::max(d_bendTurnMediumDeg.getValue(), static_cast<Real>(0.0));
    const Real bendTurnHighDeg = std::max(d_bendTurnHighDeg.getValue(), bendTurnMediumDeg + static_cast<Real>(1.0e-3));
    const Real bendSeverityRaw = std::clamp(
        (upcomingTurnDeg - bendTurnMediumDeg) / (bendTurnHighDeg - bendTurnMediumDeg),
        static_cast<Real>(0.0),
        static_cast<Real>(1.0));
    const bool previewHoldActive = (
        barrierActiveNodes > 0u
        || externalSurfaceContactActive
        || tipOffsetMm > static_cast<Real>(0.60)
        || (std::isfinite(minLumenClearanceMm) && minLumenClearanceMm < static_cast<Real>(0.60)));
    const Real bendMemoryDecay = previewHoldActive
        ? static_cast<Real>(0.96)
        : static_cast<Real>(0.86);
    const Real bendSeverity = std::max(
        bendSeverityRaw,
        std::clamp(bendMemoryDecay * prevBendSeverity, static_cast<Real>(0.0), static_cast<Real>(1.0)));
    const Real heldUpcomingTurnDeg = std::max(
        upcomingTurnDeg,
        bendMemoryDecay * prevUpcomingTurnDeg);
    const Real previewBlend = std::clamp(
        static_cast<Real>(0.16)
            + static_cast<Real>(0.36) * bendSeverity
            + static_cast<Real>(0.20) * recoveryAlpha
            + static_cast<Real>(0.10) * smoothstepRange(
                tipOffsetMm,
                static_cast<Real>(0.50),
                static_cast<Real>(1.40)),
        static_cast<Real>(0.0),
        static_cast<Real>(0.62));
    const Vec3 previewTargetTangent = safeNormalize(
        (static_cast<Real>(1.0) - static_cast<Real>(0.55) * bendSeverity) * bendNearTangent
            + static_cast<Real>(0.55) * bendSeverity * bendFarTangent,
        bendNearTangent);
    const Vec3 guidedTangent = previewBlend <= kEps
        ? lookAheadDir
        : safeNormalize(
            (static_cast<Real>(1.0) - previewBlend) * lookAheadDir
                + previewBlend * previewTargetTangent,
            lookAheadDir);
    m_lastLocalForwardTangent = guidedTangent;
    m_lastUpcomingTurnDeg = heldUpcomingTurnDeg;
    m_lastBendSeverity = bendSeverity;
    if (!hasForwardTarget)
        return guidedTangent;
    return safeNormalize(static_cast<Real>(0.25) * forwardDir + static_cast<Real>(0.75) * guidedTangent, guidedTangent);
}

Vec3 ExternalMagneticForceField::computeNearestTubeTangentDirection(const VecCoord& positions) const
{
    const VecVec3& tubeNodes = d_tubeNodes.getValue();
    const Vec3 tipPos = kMToMm * coordCenter(positions.back());
    const Vec3 fallback = positions.size() >= 2
        ? safeNormalize(coordCenter(positions.back()) - coordCenter(positions[positions.size() - 2]), safeNormalize(d_baVectorRef.getValue(), kZAxis))
        : safeNormalize(d_baVectorRef.getValue(), kZAxis);

    if (tubeNodes.size() < 2)
        return fallback;

    Real minDistance = std::numeric_limits<Real>::max();
    std::size_t contactIndex = 0;
    for (std::size_t i = 0; i + 1 < tubeNodes.size(); ++i)
    {
        const Real d1 = (tipPos - tubeNodes[i]).norm();
        const Real d2 = (tipPos - tubeNodes[i + 1]).norm();
        const Real meanDistance = static_cast<Real>(0.5) * (d1 + d2);
        if (meanDistance < minDistance)
        {
            minDistance = meanDistance;
            contactIndex = i;
        }
    }

    return safeNormalize(tubeNodes[contactIndex + 1] - tubeNodes[contactIndex], fallback);
}

std::pair<std::size_t, std::size_t> ExternalMagneticForceField::activeNodeRange(std::size_t nodeCount) const
{
    if (nodeCount < 2)
        return {0, 0};
    const std::size_t edgeCount = nodeCount - 1;
    const std::size_t magneticEdgeCount = std::min<std::size_t>(d_magneticEdgeCount.getValue(), edgeCount);
    const std::size_t nodeStart = edgeCount - magneticEdgeCount;
    return {nodeStart, edgeCount};
}
void ExternalMagneticForceField::computeMagneticForces(
    const VecCoord& q,
    const ElasticRodCompatCore::State& state,
    const Vec3& targetDirection,
    const Vec3& guidancePointMm,
    const Vec3& appliedBaVector,
    Real torqueFieldScale,
    Real assistFieldScale,
    Real assistGain,
    VecDeriv& fq,
    Vec3* debugTipForce,
    Vec3* debugTotalTorque,
    Vec3* debugAssistForce,
    Real* debugOutwardAssistComponent) const
{
    if (q.size() < 2 || m_rodModel == nullptr)
        return;

    if (fq.size() < q.size())
        fq.resize(q.size());

    const auto& restFrames = m_rodModel->compatCore().restFrames();
    const Real radius = kMmToM * d_magneticCoreRadiusMm.getValue();
    const Real area = kPi * radius * radius;
    const Real muZeroSafe = std::max(std::abs(d_muZero.getValue()), kEps);
    const Vec3 brVector = torqueFieldScale * d_brVector.getValue();
    const std::size_t edgeCount = q.size() - 1;
    const std::size_t magneticEdgeCount = std::min<std::size_t>(d_magneticEdgeCount.getValue(), edgeCount);
    const std::size_t magneticStart = edgeCount - magneticEdgeCount;
    const Vec3 steeringDir = safeNormalize(targetDirection, appliedBaVector);
    const Real lateralForceScale = std::max(d_lateralForceScale.getValue(), static_cast<Real>(0.0));
    const bool strictInLumenMode = d_strictInLumenMode.getValue();
    const bool strictPhysicalTorqueOnly = d_strictPhysicalTorqueOnly.getValue();
    const VecVec3& tubeNodes = d_tubeNodes.getValue();
    const Vec3 tipPosMm = kMToMm * coordCenter(q.back());

    Vec3 tipForce(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Vec3 totalTorque(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Vec3 totalAssistForce(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Real residualOutwardAssistComponentN = static_cast<Real>(0.0);
    sofa::type::vector<Vec3> edgeAssistForces(edgeCount, Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
    for (std::size_t i = magneticStart; i < edgeCount; ++i)
    {
        const Vec3 edgeVecM = state.centersM[i + 1] - state.centersM[i];
        const Real edgeLenM = edgeVecM.norm();
        if (edgeLenM <= kEps || i >= state.materialFrames.size() || i >= restFrames.size())
            continue;

        const auto& current = state.materialFrames[i];
        const auto& rest = restFrames[i];
        const Vec3 magneticMoment = reconstructMagneticMoment(rest, current, brVector);
        const Real coeff1 = sofa::type::dot(rest.m1, brVector);
        const Real coeff2 = sofa::type::dot(rest.m2, brVector);
        const Real coeff3 = sofa::type::dot(rest.m3, brVector);
        const Vec3 edgeDir = edgeVecM / edgeLenM;
        const Real axialField = sofa::type::dot(edgeDir, appliedBaVector);
        const Vec3 lateralField = projectLateral(edgeDir, appliedBaVector);
        const Vec3 dEde = (
            coeff3 * lateralField
            - axialField * (coeff1 * current.m1 + coeff2 * current.m2)
        ) / std::max(edgeLenM, kEps);
        const Real dEdtheta = sofa::type::dot(coeff1 * current.m2 - coeff2 * current.m1, appliedBaVector);
        const Vec3 pairForceN = (area * edgeLenM / muZeroSafe) * dEde;
        // In the non-strict reduced safe path the translational pair-force and
        // lateral assist already capture the desired bend. Keeping even a
        // moderate axial-spin torque here still makes the short distal head show
        // visible node-wise wringing that looks far less coherent than a metal
        // guidewire tip. Leave only a very small residual twist authority in the
        // safe path so the head bends, but does not corkscrew segment by segment.
        const Real twistTorqueScale = strictPhysicalTorqueOnly ? static_cast<Real>(1.0) : static_cast<Real>(0.01);
        const Real twistTorqueNm = twistTorqueScale * (area * edgeLenM / muZeroSafe) * dEdtheta;
        const Vec3 torqueNm = (area * edgeLenM / muZeroSafe) * magneticMoment.cross(appliedBaVector);
        Vec3 torqueCoupleForceN(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if (strictPhysicalTorqueOnly)
        {
            // For the reduced translational state, pairForceN already represents the
            // magnetic bending couple as equal/opposite endpoint forces. Do not add
            // a second strict-only force couple on top of that physical response.
        }
        Vec3 assistForceN(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        if ((!strictPhysicalTorqueOnly) && lateralForceScale > kEps && assistFieldScale > kEps)
        {
            Vec3 lateralDir = projectLateral(edgeDir, steeringDir);
            const Real lateralNorm = lateralDir.norm();
            if (lateralNorm > kEps)
            {
                lateralDir /= lateralNorm;
                const Real distalAlpha = magneticEdgeCount > 1
                    ? static_cast<Real>(i - magneticStart + 1u) / static_cast<Real>(magneticEdgeCount)
                    : static_cast<Real>(1.0);
                const Real distalWeight = static_cast<Real>(0.20) + static_cast<Real>(1.80) * distalAlpha * distalAlpha;
                const Real assistMagnitudeN = std::min(
                    assistFieldScale * assistGain * distalWeight * lateralForceScale * lateralNorm * torqueNm.norm() / std::max(edgeLenM, kEps),
                    kAssistForceCapPerEdgeN
                );
                assistForceN = assistMagnitudeN * lateralDir;
                if (strictInLumenMode && tubeNodes.size() >= 2)
                {
                    const Vec3 midpointMm = static_cast<Real>(0.5) * kMToMm * (state.centersM[i] + state.centersM[i + 1]);
                    Real bestD2 = std::numeric_limits<Real>::max();
                    Vec3 closestMm(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
                    for (std::size_t seg = 0; seg + 1 < tubeNodes.size(); ++seg)
                    {
                        Real u = static_cast<Real>(0.0);
                        const Vec3 proj = closestPointOnSegment(midpointMm, tubeNodes[seg], tubeNodes[seg + 1], u);
                        const Vec3 delta = midpointMm - proj;
                        const Real d2 = sofa::type::dot(delta, delta);
                        if (d2 < bestD2)
                        {
                            bestD2 = d2;
                            closestMm = proj;
                        }
                    }
                    const Vec3 radialMm = midpointMm - closestMm;
                    const Real radialNormMm = radialMm.norm();
                    if (radialNormMm > kEps)
                    {
                        const Vec3 outwardNormal = radialMm / radialNormMm;
                        const Real outwardComponent = std::max(sofa::type::dot(assistForceN, outwardNormal), static_cast<Real>(0.0));
                        if (outwardComponent > static_cast<Real>(0.0))
                        {
                            assistForceN -= outwardComponent * outwardNormal;
                        }
                        residualOutwardAssistComponentN += std::max(sofa::type::dot(assistForceN, outwardNormal), static_cast<Real>(0.0));
                    }
                }
            }
        }
        totalTorque += torqueNm;
        totalAssistForce += assistForceN;
        edgeAssistForces[i] = assistForceN;

        fq[i][0] -= pairForceN[0];
        fq[i][1] -= pairForceN[1];
        fq[i][2] -= pairForceN[2];
        fq[i + 1][0] += pairForceN[0];
        fq[i + 1][1] += pairForceN[1];
        fq[i + 1][2] += pairForceN[2];
        fq[i][0] -= torqueCoupleForceN[0];
        fq[i][1] -= torqueCoupleForceN[1];
        fq[i][2] -= torqueCoupleForceN[2];
        fq[i + 1][0] += torqueCoupleForceN[0];
        fq[i + 1][1] += torqueCoupleForceN[1];
        fq[i + 1][2] += torqueCoupleForceN[2];
        fq[i][3] += twistTorqueNm;

        if (i + 1 == edgeCount)
            tipForce += pairForceN + torqueCoupleForceN;
    }

    if (strictPhysicalTorqueOnly && lateralForceScale > kEps && assistFieldScale > kEps && q.size() >= 2)
    {
        Real minLumenClearanceMm = std::numeric_limits<Real>::infinity();
        unsigned int barrierActiveNodes = 0u;
        if (m_rodModel != nullptr)
        {
            minLumenClearanceMm = m_rodModel->d_debugMinLumenClearanceMm.getValue();
            barrierActiveNodes = m_rodModel->d_debugBarrierActiveNodeCount.getValue();
        }
        const Real contactRecoveryAlpha = barrierActiveNodes > 0u
            ? static_cast<Real>(1.0)
            : (std::isfinite(minLumenClearanceMm)
                ? smoothstepRange(
                    static_cast<Real>(0.65) - minLumenClearanceMm,
                    static_cast<Real>(0.0),
                    static_cast<Real>(0.45))
                : static_cast<Real>(0.0));
        Vec3 nearestPointMm = tipPosMm;
        Vec3 nearestTubeTangent = safeNormalize(m_lastLocalForwardTangent, steeringDir);
        Real nearestS = static_cast<Real>(0.0);
        Real tipOffsetMm = static_cast<Real>(0.0);
        nearestTubeProjection(
            d_tubeNodes.getValue(),
            m_tubeCum,
            tipPosMm,
            nearestPointMm,
            nearestS,
            tipOffsetMm,
            &nearestTubeTangent);
        // Strict magnetic centering must stay purely lateral in the local
        // centerline frame. It may recenter the head toward the forward target
        // point, but it must not inject any artificial along-path traction.
        const Vec3 previewForwardAxis = safeNormalize(m_lastLocalForwardTangent, nearestTubeTangent);
        const Real previewTurnDeg = std::acos(std::clamp(
            sofa::type::dot(previewForwardAxis, nearestTubeTangent),
            static_cast<Real>(-1.0),
            static_cast<Real>(1.0))) * static_cast<Real>(180.0) / kPi;
        const Real previewTurnAlpha = smoothstepRange(
            previewTurnDeg,
            static_cast<Real>(6.0),
            static_cast<Real>(24.0));
        const Real previewAxisBlend = std::clamp(
            static_cast<Real>(0.18) * previewTurnAlpha
                + static_cast<Real>(0.30) * contactRecoveryAlpha
                + static_cast<Real>(0.18) * smoothstepRange(
                    tipOffsetMm,
                    static_cast<Real>(0.35),
                    static_cast<Real>(1.20)),
            static_cast<Real>(0.0),
            static_cast<Real>(0.72));
        const Vec3 localForwardAxis = safeNormalize(
            (static_cast<Real>(1.0) - previewAxisBlend) * nearestTubeTangent
                + previewAxisBlend * previewForwardAxis,
            nearestTubeTangent);
        Vec3 recenterTargetMm = nearestPointMm;
        const Real previewAlpha = std::clamp(
            static_cast<Real>(0.08)
                + static_cast<Real>(0.20) * smoothstepRange((guidancePointMm - nearestPointMm).norm(), static_cast<Real>(1.0), static_cast<Real>(8.0))
                + static_cast<Real>(0.22) * smoothstepRange(tipOffsetMm, static_cast<Real>(0.8), static_cast<Real>(2.0))
                + static_cast<Real>(0.18) * contactRecoveryAlpha,
            static_cast<Real>(0.0),
            static_cast<Real>(0.60));
        recenterTargetMm += previewAlpha * (guidancePointMm - nearestPointMm);
        const Vec3 tipToGuidanceMm = recenterTargetMm - tipPosMm;
        Vec3 tipPullMm = projectLateral(localForwardAxis, tipToGuidanceMm);
        Real tipPullNormMm = tipPullMm.norm();
        if (tipPullNormMm <= kEps && contactRecoveryAlpha > kEps)
        {
            tipPullMm = projectLateral(localForwardAxis, guidancePointMm - nearestPointMm);
            tipPullNormMm = tipPullMm.norm();
        }
        if (
            tipPullNormMm <= kEps
            && (contactRecoveryAlpha > kEps || tipOffsetMm > static_cast<Real>(0.25))
        )
        {
            tipPullMm = projectLateral(localForwardAxis, nearestPointMm - tipPosMm);
            tipPullNormMm = tipPullMm.norm();
        }
        if (
            tipPullNormMm <= static_cast<Real>(0.20)
            && q.size() >= 2
            && (contactRecoveryAlpha > static_cast<Real>(0.10) || previewTurnAlpha > static_cast<Real>(0.35))
        )
        {
            const Vec3 tipEdgeDir = safeNormalize(
                kMToMm * (coordCenter(q.back()) - coordCenter(q[q.size() - 2])),
                localForwardAxis);
            Vec3 steeringHoldPull = projectLateral(localForwardAxis, previewForwardAxis - tipEdgeDir);
            const Real steeringHoldNorm = steeringHoldPull.norm();
            if (steeringHoldNorm > kEps)
            {
                const Real steeringMismatchDeg = std::acos(std::clamp(
                    sofa::type::dot(tipEdgeDir, previewForwardAxis),
                    static_cast<Real>(-1.0),
                    static_cast<Real>(1.0))) * static_cast<Real>(180.0) / kPi;
                const Real steeringMismatchAlpha = smoothstepRange(
                    steeringMismatchDeg,
                    static_cast<Real>(8.0),
                    static_cast<Real>(35.0));
                const Real steeringHoldScaleMm = std::clamp(
                    static_cast<Real>(0.20)
                        + static_cast<Real>(0.55) * steeringMismatchAlpha
                        + static_cast<Real>(0.20) * contactRecoveryAlpha
                        + static_cast<Real>(0.15) * previewTurnAlpha,
                    static_cast<Real>(0.20),
                    static_cast<Real>(1.10));
                tipPullMm = (steeringHoldScaleMm / steeringHoldNorm) * steeringHoldPull;
                tipPullNormMm = tipPullMm.norm();
            }
        }
        if (
            tipPullNormMm <= static_cast<Real>(0.20)
            && m_hasLastStrictAssistDirection
            && (contactRecoveryAlpha > static_cast<Real>(0.10) || previewTurnAlpha > static_cast<Real>(0.35))
        )
        {
            Vec3 heldPull = projectLateral(localForwardAxis, m_lastStrictAssistDirection);
            const Real heldPullNorm = heldPull.norm();
            if (heldPullNorm > kEps)
            {
                bool useHeldPull = true;
                const Vec3 previewReference = projectLateral(localForwardAxis, previewForwardAxis);
                const Real previewReferenceNorm = previewReference.norm();
                if (previewReferenceNorm > kEps)
                {
                    const Real heldSign = sofa::type::dot(
                        heldPull / heldPullNorm,
                        previewReference / previewReferenceNorm);
                    useHeldPull = heldSign >= static_cast<Real>(0.15);
                }
                if (useHeldPull)
                {
                    const Real assistHoldScaleMm = std::clamp(
                        static_cast<Real>(0.25)
                            + static_cast<Real>(0.35) * contactRecoveryAlpha
                            + static_cast<Real>(0.25) * previewTurnAlpha,
                        static_cast<Real>(0.25),
                        static_cast<Real>(0.90));
                    tipPullMm = (assistHoldScaleMm / heldPullNorm) * heldPull;
                    tipPullNormMm = tipPullMm.norm();
                }
            }
        }
        if (tipPullNormMm > kEps)
        {
            m_lastStrictAssistDirection = tipPullMm / tipPullNormMm;
            m_hasLastStrictAssistDirection = true;
            const Real targetGate = smoothstepRange(
                tipPullNormMm,
                static_cast<Real>(0.20),
                static_cast<Real>(1.40));
            if (targetGate > kEps)
            {
                tipPullMm /= tipPullNormMm;
                const Real magneticHeadLengthScale = std::clamp(
                    static_cast<Real>(magneticEdgeCount) / static_cast<Real>(4.0),
                    static_cast<Real>(1.0),
                    static_cast<Real>(1.6));
                // In strict mode interpret lateralForceScale as the target total
                // lateral force that the soft composite magnetic head can
                // receive. The previous implementation multiplied it by a small
                // gain and only pushed the last node, which left the delivered
                // head-centering force far below the configured target and let
                // the tip pull once then self-straighten under shaft stiffness.
                const Real demandAlpha = std::clamp(
                    static_cast<Real>(0.35)
                        + static_cast<Real>(0.25) * std::min(assistGain, static_cast<Real>(1.6))
                        + static_cast<Real>(0.25) * targetGate
                        + static_cast<Real>(0.15) * std::clamp(tipOffsetMm / static_cast<Real>(1.5), static_cast<Real>(0.0), static_cast<Real>(1.0))
                        + static_cast<Real>(0.20) * contactRecoveryAlpha,
                    static_cast<Real>(0.0),
                    static_cast<Real>(1.0));
                // Keep the configured strict tip-target force as the baseline,
                // but allow a modest local boost when the head is already in a
                // meaningful bend/contact recovery regime. The first bend was
                // consistently saturating at the static cap, which made the tip
                // pull once and then re-straighten before it could commit into
                // the branch.
                const Real assistCapBoost = std::clamp(
                    static_cast<Real>(1.0)
                        + static_cast<Real>(0.15) * std::max(assistGain - static_cast<Real>(1.0), static_cast<Real>(0.0))
                        + static_cast<Real>(0.05) * targetGate
                        + static_cast<Real>(0.05) * std::clamp(tipOffsetMm / static_cast<Real>(1.2), static_cast<Real>(0.0), static_cast<Real>(1.0))
                        + static_cast<Real>(0.10) * contactRecoveryAlpha,
                    static_cast<Real>(1.0),
                    static_cast<Real>(1.35));
                const Real tipPullCapN = std::min(
                    kAssistForceCapTotalN,
                    lateralForceScale * assistCapBoost);
                const Real tipPullForceMagN = std::min(
                    assistFieldScale * magneticHeadLengthScale * lateralForceScale * demandAlpha,
                    tipPullCapN);
                const Vec3 tipPullForceN = tipPullForceMagN * tipPullMm;
                const std::size_t headNodeCount = std::min<std::size_t>(q.size(), magneticEdgeCount + 1u);
                const std::size_t headNodeStart = q.size() - headNodeCount;
                Real weightSum = static_cast<Real>(0.0);
                std::vector<Real> nodeWeights(headNodeCount, static_cast<Real>(0.0));
                for (std::size_t localIndex = 0; localIndex < headNodeCount; ++localIndex)
                {
                    const Real alpha = headNodeCount > 1u
                        ? static_cast<Real>(localIndex) / static_cast<Real>(headNodeCount - 1u)
                        : static_cast<Real>(1.0);
                    const Real weight = static_cast<Real>(0.35) + static_cast<Real>(0.65) * alpha * alpha;
                    nodeWeights[localIndex] = weight;
                    weightSum += weight;
                }
                if (weightSum > kEps)
                {
                    for (std::size_t localIndex = 0; localIndex < headNodeCount; ++localIndex)
                    {
                        const Real weight = nodeWeights[localIndex] / weightSum;
                        const std::size_t nodeIndex = headNodeStart + localIndex;
                        fq[nodeIndex][0] += weight * tipPullForceN[0];
                        fq[nodeIndex][1] += weight * tipPullForceN[1];
                        fq[nodeIndex][2] += weight * tipPullForceN[2];
                    }
                }
                tipForce += tipPullForceN;
                totalAssistForce += tipPullForceN;
            }
        }
        else if (contactRecoveryAlpha <= static_cast<Real>(0.05) && previewTurnAlpha <= static_cast<Real>(0.20))
        {
            m_hasLastStrictAssistDirection = false;
            m_lastStrictAssistDirection = Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        }
    }

    const Real totalAssistNorm = totalAssistForce.norm();
    Real assistScale = static_cast<Real>(1.0);
    if (totalAssistNorm > kAssistForceCapTotalN)
    {
        assistScale = kAssistForceCapTotalN / totalAssistNorm;
        totalAssistForce *= assistScale;
    }

    for (std::size_t i = magneticStart; i < edgeCount; ++i)
    {
        const Vec3 assistForceN = assistScale * edgeAssistForces[i];
        const Real distalAlpha = magneticEdgeCount > 1
            ? static_cast<Real>(i - magneticStart + 1u) / static_cast<Real>(magneticEdgeCount)
            : static_cast<Real>(1.0);
        const Real distalNodeWeight = std::clamp(
            static_cast<Real>(0.50) + static_cast<Real>(0.35) * distalAlpha,
            static_cast<Real>(0.50),
            static_cast<Real>(0.85));
        const Real proximalNodeWeight = static_cast<Real>(1.0) - distalNodeWeight;
        fq[i][0] += proximalNodeWeight * assistForceN[0];
        fq[i][1] += proximalNodeWeight * assistForceN[1];
        fq[i][2] += proximalNodeWeight * assistForceN[2];
        fq[i + 1][0] += distalNodeWeight * assistForceN[0];
        fq[i + 1][1] += distalNodeWeight * assistForceN[1];
        fq[i + 1][2] += distalNodeWeight * assistForceN[2];
    }

    tipForce += totalAssistForce;

    if (debugTipForce != nullptr)
        *debugTipForce = tipForce;
    if (debugTotalTorque != nullptr)
        *debugTotalTorque = totalTorque;
    if (debugAssistForce != nullptr)
        *debugAssistForce = totalAssistForce;
    if (debugOutwardAssistComponent != nullptr)
        *debugOutwardAssistComponent = residualOutwardAssistComponentN;
}

void ExternalMagneticForceField::addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv&)
{
    const VecCoord& q = x.getValue();
    VecDeriv& fq = *f.beginEdit();
    if (q.size() < 2)
    {
        f.endEdit();
        return;
    }
    if (m_tubeCum.size() != d_tubeNodes.getValue().size())
        rebuildTubeArc();

    Vec3 targetPoint = coordCenter(q.back());
    Vec3 lookAheadPoint = targetPoint;
    const Vec3 nearestTangent = computeNearestTubeTangentDirection(q);
    const Vec3 nativeTargetDirection = safeNormalize(computeLookAheadTargetDirection(q, targetPoint, &lookAheadPoint), nearestTangent);
    Vec3 targetDirection = nativeTargetDirection;
    if (d_useExternalTargetDirection.getValue())
    {
        targetDirection = safeNormalize(d_externalTargetDirection.getValue(), nativeTargetDirection);
        const Vec3 tipPosMm = kMToMm * coordCenter(q.back());
        const Real externalLookAheadMm = std::max(
            std::max(d_lookAheadDistance.getValue(), d_recoveryLookAheadDistance.getValue()),
            static_cast<Real>(1.0));
        targetPoint = tipPosMm + externalLookAheadMm * targetDirection;
        lookAheadPoint = targetPoint;
    }

    Vec3 aggregateMoment(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Vec3 distalTangent = nearestTangent;
    bool hasMoment = false;
    ElasticRodCompatCore::State currentState;
    Real currentMaxHeadStretch = static_cast<Real>(0.0);
    if (m_rodModel != nullptr)
    {
        currentState = m_rodModel->computeCurrentState(q);
        const auto& restFrames = m_rodModel->compatCore().restFrames();
        const auto& refLen = m_rodModel->compatCore().refLen();
        const std::size_t edgeCount = q.size() - 1;
        const std::size_t magneticEdgeCount = std::min<std::size_t>(d_magneticEdgeCount.getValue(), edgeCount);
        const std::size_t magneticStart = edgeCount - magneticEdgeCount;
        for (std::size_t i = magneticStart; i < edgeCount; ++i)
        {
            if (i + 1 >= currentState.centersM.size() || i >= currentState.materialFrames.size() || i >= restFrames.size())
                continue;
            const Vec3 edgeVecM = currentState.centersM[i + 1] - currentState.centersM[i];
            const Real edgeLenM = edgeVecM.norm();
            if (edgeLenM <= kEps)
                continue;

            distalTangent = safeNormalize(edgeVecM, distalTangent);
            aggregateMoment += edgeLenM * reconstructMagneticMoment(restFrames[i], currentState.materialFrames[i], d_brVector.getValue());
            if (i < refLen.size())
            {
                const Real refEdgeLenM = std::max(refLen[i], kEps);
                currentMaxHeadStretch = std::max(
                    currentMaxHeadStretch,
                    std::abs(edgeLenM - refEdgeLenM) / refEdgeLenM);
            }
        }
        hasMoment = aggregateMoment.norm() > kEps;
    }

    const Vec3 momentFallback = safeNormalize(distalTangent, targetDirection);
    const bool strictPhysicalTorqueOnly = d_strictPhysicalTorqueOnly.getValue();
    const Vec3 tipPosMm = kMToMm * coordCenter(q.back());
    Vec3 nearestPointMm = targetPoint;
    Vec3 nearestTubeTangent = nearestTangent;
    Real nearestS = static_cast<Real>(0.0);
    Real tipOffsetMm = static_cast<Real>(0.0);
    nearestTubeProjection(d_tubeNodes.getValue(), m_tubeCum, tipPosMm, nearestPointMm, nearestS, tipOffsetMm, &nearestTubeTangent);
    Real minLumenClearanceMm = std::numeric_limits<Real>::infinity();
    unsigned int barrierActiveNodes = 0u;
    if (m_rodModel != nullptr)
    {
        minLumenClearanceMm = m_rodModel->d_debugMinLumenClearanceMm.getValue();
        barrierActiveNodes = m_rodModel->d_debugBarrierActiveNodeCount.getValue();
    }
    const Real externalSurfaceClearanceMm = d_externalSurfaceClearanceMm.getValue();
    const bool externalSurfaceContactActive = d_externalSurfaceContactActive.getValue();
    if (std::isfinite(externalSurfaceClearanceMm))
        minLumenClearanceMm = std::min(minLumenClearanceMm, externalSurfaceClearanceMm);
    if (externalSurfaceContactActive)
        barrierActiveNodes = std::max(barrierActiveNodes, 1u);
    const Real bendSeverity = std::clamp(m_lastBendSeverity, static_cast<Real>(0.0), static_cast<Real>(1.0));
    const Real scheduledFieldScaleBase = strictPhysicalTorqueOnly
        ? ((static_cast<Real>(1.0) - bendSeverity) * d_fieldScaleStraight.getValue() + bendSeverity * d_fieldScaleBend.getValue())
        : static_cast<Real>(1.0);
    const Real headStretchReliefStart = std::max(d_headStretchReliefStart.getValue(), static_cast<Real>(0.0));
    const Real headStretchReliefFull = std::max(
        d_headStretchReliefFull.getValue(),
        headStretchReliefStart + static_cast<Real>(1.0e-6));
    const Real headStretchGate = std::clamp(
        (currentMaxHeadStretch - headStretchReliefStart) / (headStretchReliefFull - headStretchReliefStart),
        static_cast<Real>(0.0),
        static_cast<Real>(1.0));

    Vec3 desiredBa = safeNormalize(targetDirection, nearestTangent);
    Real recenteringAlpha = static_cast<Real>(0.0);
    Real antiKinkReleaseAlpha = static_cast<Real>(0.0);
    Real strictSteeringNeedAlpha = static_cast<Real>(1.0);
    Real strictOffsetNeed = static_cast<Real>(0.0);
    Real strictClearanceNeed = static_cast<Real>(0.0);
    Real strictCenterlinePullNeed = static_cast<Real>(0.0);
    Real strictTurnNeed = static_cast<Real>(0.0);
    Real strictTargetPullNeed = static_cast<Real>(0.0);
    Real strictBranchCommitFloor = static_cast<Real>(0.0);
    const Real entryReleaseAlpha = std::clamp(
        m_lastStrictEntrySteeringAlpha,
        static_cast<Real>(0.0),
        static_cast<Real>(1.0));
    if (hasMoment && !strictPhysicalTorqueOnly)
    {
        const Vec3 momentDirection = safeNormalize(aggregateMoment, momentFallback);
        desiredBa = buildTorqueAwareFieldDirection(momentDirection, targetDirection, d_minTorqueSin.getValue(), nullptr);
        const Vec3 safeForwardAxis = safeNormalize(m_lastLocalForwardTangent, nearestTangent);
        const Vec3 guidancePointMm = ((lookAheadPoint - tipPosMm).norm() > kEps) ? lookAheadPoint : targetPoint;
        const Vec3 guidancePullMm = projectLateral(safeForwardAxis, guidancePointMm - tipPosMm);
        const Vec3 centerlinePullMm = projectLateral(safeForwardAxis, nearestPointMm - tipPosMm);
        const Real guidancePullNormMm = guidancePullMm.norm();
        const Real centerlinePullNormMm = centerlinePullMm.norm();
        const Real safeOffsetNeed = smoothstepRange(
            tipOffsetMm,
            static_cast<Real>(0.18),
            static_cast<Real>(0.85));
        const Real safeClearanceNeed = std::isfinite(minLumenClearanceMm)
            ? smoothstepRange(
                static_cast<Real>(0.90) - minLumenClearanceMm,
                static_cast<Real>(0.0),
                static_cast<Real>(0.45))
            : static_cast<Real>(0.0);
        const Real safeBarrierNeed = smoothstepRange(
            static_cast<Real>(barrierActiveNodes),
            static_cast<Real>(1.0),
            static_cast<Real>(4.0));
        if (
            guidancePullNormMm > static_cast<Real>(0.20)
            && std::max(safeOffsetNeed, std::max(safeClearanceNeed, static_cast<Real>(0.35) * safeBarrierNeed)) > kEps
        )
        {
            const Real guidanceSignAlpha = std::clamp(
                static_cast<Real>(0.18)
                    + static_cast<Real>(0.54)
                        * std::max(safeOffsetNeed, std::max(safeClearanceNeed, static_cast<Real>(0.35) * safeBarrierNeed)),
                static_cast<Real>(0.0),
                static_cast<Real>(0.78));
            desiredBa = enforceGuidanceLateralSign(
                desiredBa,
                safeForwardAxis,
                guidancePullMm,
                guidanceSignAlpha);
        }
        if (
            centerlinePullNormMm > kEps
            && std::max(safeOffsetNeed, std::max(safeClearanceNeed, safeBarrierNeed)) > kEps
        )
        {
            const Real inwardSignAlpha = std::clamp(
                static_cast<Real>(0.22)
                    + static_cast<Real>(0.70)
                        * std::max(safeOffsetNeed, std::max(safeClearanceNeed, safeBarrierNeed)),
                static_cast<Real>(0.0),
                static_cast<Real>(0.88));
            desiredBa = enforceGuidanceLateralSign(
                desiredBa,
                safeForwardAxis,
                centerlinePullMm,
                inwardSignAlpha);
        }
        const Real safeSteeringNeed = std::clamp(
            std::max(
                std::max(safeOffsetNeed, safeClearanceNeed),
                std::max(
                    safeBarrierNeed,
                    std::max(
                        smoothstepRange(guidancePullNormMm, static_cast<Real>(0.20), static_cast<Real>(1.60)),
                        static_cast<Real>(0.85) * smoothstepRange(centerlinePullNormMm, static_cast<Real>(0.15), static_cast<Real>(1.10))))),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        if (m_hasAppliedBaVector && safeSteeringNeed > kEps)
        {
            const Vec3 heldBa = safeNormalize(m_appliedBaVector, desiredBa);
            const Vec3 heldLateral = projectLateral(safeForwardAxis, heldBa);
            const Real heldLateralNorm = heldLateral.norm();
            Real sameSideAlpha = static_cast<Real>(1.0);
            if (guidancePullNormMm > kEps && heldLateralNorm > kEps)
            {
                sameSideAlpha = std::clamp(
                    static_cast<Real>(0.5)
                        + static_cast<Real>(0.5)
                            * sofa::type::dot(heldLateral / heldLateralNorm, guidancePullMm / guidancePullNormMm),
                    static_cast<Real>(0.0),
                    static_cast<Real>(1.0));
            }
            const Real currentAngleToTangent = std::acos(std::clamp(
                sofa::type::dot(safeNormalize(desiredBa, nearestTangent), nearestTangent),
                static_cast<Real>(-1.0),
                static_cast<Real>(1.0)));
            const Real heldAngleToTangent = std::acos(std::clamp(
                sofa::type::dot(safeNormalize(heldBa, nearestTangent), nearestTangent),
                static_cast<Real>(-1.0),
                static_cast<Real>(1.0)));
            const Real relaxMarginRad = static_cast<Real>(2.0) * kPi / static_cast<Real>(180.0);
            if (sameSideAlpha > static_cast<Real>(0.35) && heldAngleToTangent > currentAngleToTangent + relaxMarginRad)
            {
                const Real holdAlpha = std::clamp(
                    sameSideAlpha
                        * (static_cast<Real>(0.22)
                            + static_cast<Real>(0.48) * safeSteeringNeed
                            + static_cast<Real>(0.08) * safeBarrierNeed),
                    static_cast<Real>(0.0),
                    static_cast<Real>(0.68));
                desiredBa = safeNormalize(
                    (static_cast<Real>(1.0) - holdAlpha) * desiredBa + holdAlpha * heldBa,
                    desiredBa);
            }
        }
    }
    else if (hasMoment && strictPhysicalTorqueOnly)
    {
        // In strict mode we do not want the magnetic field to inject a large
        // steering torque while the head is still nearly centered and the
        // upcoming centerline remains gentle. Let the field stay close to the
        // current magnetic-moment direction until there is an actual need to
        // recenter or pre-steer, then blend toward the forward centerline
        // guidance.
        const Vec3 momentDirection = safeNormalize(aggregateMoment, momentFallback);
        const Vec3 strictForwardAxis = safeNormalize(m_lastLocalForwardTangent, nearestTangent);
        const Vec3 guidancePointMm = ((lookAheadPoint - tipPosMm).norm() > kEps) ? lookAheadPoint : targetPoint;
        const Vec3 centerlinePointMm = nearestPointMm;
        const Vec3 guidancePullMm = projectLateral(strictForwardAxis, guidancePointMm - tipPosMm);
        const Vec3 centerlinePullMm = projectLateral(strictForwardAxis, centerlinePointMm - tipPosMm);
        const Real guidancePullNormMm = guidancePullMm.norm();
        const Real centerlinePullNormMm = centerlinePullMm.norm();
        strictCenterlinePullNeed = smoothstepRange(
            centerlinePullNormMm,
            static_cast<Real>(0.20),
            static_cast<Real>(1.20));
        strictTargetPullNeed = smoothstepRange(
            guidancePullNormMm,
            static_cast<Real>(0.25),
            static_cast<Real>(1.80));
        // The earlier thresholds waited until the head was already visibly
        // eccentric or nearly touching the wall before strict steering started
        // helping. That lets the very soft distal magnetic segment form a
        // "hook" first and only then asks the field to recover it. Trigger the
        // need signal earlier so magnetic recentering starts while the head is
        // still only mildly offset.
        strictOffsetNeed = smoothstepRange(
            tipOffsetMm,
            static_cast<Real>(0.20),
            static_cast<Real>(0.85));
        strictClearanceNeed = std::isfinite(minLumenClearanceMm)
            ? smoothstepRange(
                static_cast<Real>(0.95) - minLumenClearanceMm,
                static_cast<Real>(0.0),
                static_cast<Real>(0.45))
            : static_cast<Real>(0.0);
        const Real turnEntryDeg = std::max(
            static_cast<Real>(5.0),
            static_cast<Real>(0.60) * std::max(d_bendTurnMediumDeg.getValue(), static_cast<Real>(0.0)));
        const Real turnFullDeg = std::max(
            turnEntryDeg + static_cast<Real>(1.0e-3),
            static_cast<Real>(0.82) * std::max(d_bendTurnHighDeg.getValue(), turnEntryDeg + static_cast<Real>(1.0e-3)));
        const Real turnNeed = smoothstepRange(
            m_lastUpcomingTurnDeg,
            turnEntryDeg,
            turnFullDeg);
        strictTurnNeed = turnNeed;
        const Real bendNeed = smoothstepRange(
            bendSeverity,
            static_cast<Real>(0.03),
            static_cast<Real>(0.35));
        const Real previewNeed = std::clamp(
            std::max(
                std::max(strictOffsetNeed, strictClearanceNeed),
                std::max(
                    static_cast<Real>(0.85) * strictTargetPullNeed,
                    static_cast<Real>(0.80) * strictCenterlinePullNeed))
                * static_cast<Real>(0.18)
                + turnNeed * static_cast<Real>(0.40)
                + bendNeed * static_cast<Real>(0.28)
                + strictTargetPullNeed * static_cast<Real>(0.34)
                + strictCenterlinePullNeed * static_cast<Real>(0.20),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        strictSteeringNeedAlpha = smoothstep01(previewNeed);
        strictSteeringNeedAlpha = std::max(
            strictSteeringNeedAlpha,
            std::clamp(
                static_cast<Real>(0.46) * strictTargetPullNeed
                    + static_cast<Real>(0.18) * strictCenterlinePullNeed
                    + static_cast<Real>(0.32) * std::max(strictTurnNeed, bendNeed),
                static_cast<Real>(0.0),
                static_cast<Real>(1.0)));
        if (barrierActiveNodes > 0u)
        {
            strictSteeringNeedAlpha = std::max(
                strictSteeringNeedAlpha,
                static_cast<Real>(0.60) + static_cast<Real>(0.25) * std::max(strictClearanceNeed, strictCenterlinePullNeed));
        }
        const Real branchCommitNeed = std::clamp(
            std::max(
                std::max(strictTurnNeed, bendNeed),
                std::max(
                    std::max(strictTargetPullNeed, strictCenterlinePullNeed),
                    std::max(strictClearanceNeed, static_cast<Real>(0.70) * strictOffsetNeed))),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        const Real recenterClearanceThresholdMm = std::max(
            d_recenterClearanceMm.getValue(),
            static_cast<Real>(1.0e-3));
        const Real recenterOffsetThresholdMm = std::max(
            d_recenterOffsetMm.getValue(),
            static_cast<Real>(0.0));
        const Real recenterOffsetNeed = smoothstepRange(
            tipOffsetMm,
            recenterOffsetThresholdMm,
            recenterOffsetThresholdMm + static_cast<Real>(0.90));
        const Real recenterClearanceNeed = std::isfinite(minLumenClearanceMm)
            ? smoothstepRange(
                recenterClearanceThresholdMm - minLumenClearanceMm,
                static_cast<Real>(0.0),
                recenterClearanceThresholdMm)
            : static_cast<Real>(0.0);
        const Real recenterBarrierNeed = smoothstepRange(
            static_cast<Real>(barrierActiveNodes),
            static_cast<Real>(1.0),
            static_cast<Real>(5.0));
        Real recenterSideConflictNeed = static_cast<Real>(0.0);
        if (m_hasAppliedBaVector && guidancePullNormMm > kEps)
        {
            const Vec3 heldLateral = projectLateral(strictForwardAxis, m_appliedBaVector);
            const Real heldLateralNorm = heldLateral.norm();
            if (heldLateralNorm > kEps)
            {
                recenterSideConflictNeed = smoothstepRange(
                    -sofa::type::dot(heldLateral / heldLateralNorm, guidancePullMm / guidancePullNormMm),
                    static_cast<Real>(0.05),
                    static_cast<Real>(0.65));
            }
        }
        recenteringAlpha = std::clamp(
            d_recenterBlend.getValue()
                * std::max(
                    std::min(
                        recenterClearanceNeed,
                        std::max(recenterOffsetNeed, headStretchGate)),
                    recenterSideConflictNeed * std::max(recenterClearanceNeed, recenterBarrierNeed)),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        antiKinkReleaseAlpha = std::clamp(
            std::max(recenteringAlpha, static_cast<Real>(0.85) * headStretchGate),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        if (entryReleaseAlpha > kEps && branchCommitNeed > kEps)
        {
            strictBranchCommitFloor = std::clamp(
                entryReleaseAlpha
                    * (static_cast<Real>(0.18) * std::max(strictTargetPullNeed, strictCenterlinePullNeed)
                        + static_cast<Real>(0.34) * branchCommitNeed
                        + static_cast<Real>(0.10) * std::max(strictClearanceNeed, strictOffsetNeed)),
                static_cast<Real>(0.0),
                static_cast<Real>(0.55));
            if (m_hasAppliedBaVector && guidancePullNormMm > kEps)
            {
                const Vec3 heldLateral = projectLateral(strictForwardAxis, m_appliedBaVector);
                const Real heldLateralNorm = heldLateral.norm();
                if (heldLateralNorm > kEps)
                {
                    const Real signCommit = std::clamp(
                        sofa::type::dot(heldLateral / heldLateralNorm, guidancePullMm / guidancePullNormMm),
                        static_cast<Real>(0.0),
                        static_cast<Real>(1.0));
                    strictBranchCommitFloor *= static_cast<Real>(0.55) + static_cast<Real>(0.45) * signCommit;
                }
            }
        }
        strictBranchCommitFloor *= std::clamp(
            static_cast<Real>(1.0) - static_cast<Real>(0.80) * antiKinkReleaseAlpha,
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        strictSteeringNeedAlpha *= std::clamp(
            static_cast<Real>(1.0) - static_cast<Real>(0.45) * antiKinkReleaseAlpha,
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        const Real steeringBlend = std::clamp(
            std::max(entryReleaseAlpha * strictSteeringNeedAlpha, strictBranchCommitFloor)
                * std::clamp(
                    static_cast<Real>(1.0) - static_cast<Real>(0.55) * antiKinkReleaseAlpha,
                    static_cast<Real>(0.0),
                    static_cast<Real>(1.0)),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        Vec3 torqueTargetDirection = targetDirection;
        if (centerlinePullNormMm > kEps)
        {
            Vec3 inwardLateralDir = centerlinePullMm / centerlinePullNormMm;
            if (guidancePullNormMm > kEps)
            {
                const Vec3 guidanceLateralDir = guidancePullMm / guidancePullNormMm;
                const Real previewWeight = std::clamp(
                    static_cast<Real>(0.16)
                        + static_cast<Real>(0.24) * turnNeed
                        + static_cast<Real>(0.14) * bendNeed
                        + static_cast<Real>(0.08) * strictTargetPullNeed,
                    static_cast<Real>(0.0),
                    static_cast<Real>(0.50));
                inwardLateralDir = safeNormalize(
                    (static_cast<Real>(1.0) - previewWeight) * inwardLateralDir
                        + previewWeight * guidanceLateralDir,
                    inwardLateralDir);
            }
            const Real centerlineBiasAlpha = std::clamp(
                static_cast<Real>(0.16) * strictTargetPullNeed
                    + static_cast<Real>(0.10) * strictCenterlinePullNeed
                    + static_cast<Real>(0.22) * turnNeed
                    + static_cast<Real>(0.12) * bendNeed
                    + static_cast<Real>(0.08) * strictClearanceNeed,
                static_cast<Real>(0.0),
                static_cast<Real>(0.40));
            if (centerlineBiasAlpha > kEps)
            {
                const Real lateralBias = std::clamp(
                    static_cast<Real>(0.30)
                        + static_cast<Real>(0.18) * strictTargetPullNeed
                        + static_cast<Real>(0.10) * strictCenterlinePullNeed
                        + static_cast<Real>(0.10) * turnNeed,
                    static_cast<Real>(0.24),
                    static_cast<Real>(0.54));
                const Vec3 inwardTarget = safeNormalize(
                    strictForwardAxis + lateralBias * inwardLateralDir,
                    strictForwardAxis);
                torqueTargetDirection = safeNormalize(
                    (static_cast<Real>(1.0) - centerlineBiasAlpha) * targetDirection
                        + centerlineBiasAlpha * inwardTarget,
                    targetDirection);
            }
        }
        const Vec3 torqueAwareTarget = buildTorqueAwareFieldDirection(
            momentDirection,
            torqueTargetDirection,
            static_cast<Real>(0.0),
            nullptr);
        desiredBa = steeringBlend <= kEps
            ? momentDirection
            : safeNormalize(
                (static_cast<Real>(1.0) - steeringBlend) * momentDirection
                    + steeringBlend * torqueAwareTarget,
                torqueAwareTarget);
        if (guidancePullNormMm > kEps && strictTargetPullNeed > kEps && antiKinkReleaseAlpha < static_cast<Real>(0.35))
        {
            const Real signHoldAlpha = std::clamp(
                static_cast<Real>(0.70)
                    + static_cast<Real>(0.15) * strictTargetPullNeed
                    + static_cast<Real>(0.10) * std::max(strictClearanceNeed, strictOffsetNeed),
                static_cast<Real>(0.0),
                static_cast<Real>(0.80));
            desiredBa = enforceGuidanceLateralSign(
                desiredBa,
                strictForwardAxis,
                guidancePullMm,
                signHoldAlpha);
        }
        if (recenteringAlpha > kEps)
        {
            const Vec3 releaseTarget = buildTorqueAwareFieldDirection(
                momentDirection,
                strictForwardAxis,
                static_cast<Real>(0.0),
                nullptr);
            desiredBa = safeNormalize(
                (static_cast<Real>(1.0) - recenteringAlpha) * desiredBa + recenteringAlpha * releaseTarget,
                releaseTarget);
        }
        // When the distal head is already being steered into a bend, the raw
        // preview target can relax toward the local tangent for one or two
        // steps and make the field "snap back" visually. Keep a short memory
        // of the previously applied steering direction whenever the bend /
        // contact need is still active, so the strict path turns continuously
        // through the corner instead of pulling once and self-straightening.
        const Real steeringHoldNeed = std::clamp(
            std::max(
                bendSeverity,
                std::max(
                    strictTurnNeed,
                    std::max(
                        strictClearanceNeed,
                        std::max(
                            static_cast<Real>(0.70) * strictOffsetNeed,
                            static_cast<Real>(0.90) * strictTargetPullNeed)))),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        if (m_hasAppliedBaVector && steeringHoldNeed > kEps && antiKinkReleaseAlpha < static_cast<Real>(0.95))
        {
            const Vec3 heldBa = safeNormalize(m_appliedBaVector, desiredBa);
            const Real currentAngleToTangent = std::acos(std::clamp(
                sofa::type::dot(safeNormalize(desiredBa, nearestTangent), nearestTangent),
                static_cast<Real>(-1.0),
                static_cast<Real>(1.0)));
            const Real heldAngleToTangent = std::acos(std::clamp(
                sofa::type::dot(safeNormalize(heldBa, nearestTangent), nearestTangent),
                static_cast<Real>(-1.0),
                static_cast<Real>(1.0)));
            const Real relaxMarginRad = static_cast<Real>(2.0) * kPi / static_cast<Real>(180.0);
            if (heldAngleToTangent > currentAngleToTangent + relaxMarginRad)
            {
                const Real holdAlpha = std::clamp(
                    static_cast<Real>(0.35)
                        + static_cast<Real>(0.45) * steeringHoldNeed
                        + static_cast<Real>(0.15) * strictBranchCommitFloor,
                    static_cast<Real>(0.0),
                    static_cast<Real>(0.72))
                    * std::clamp(
                        static_cast<Real>(1.0) - static_cast<Real>(0.90) * antiKinkReleaseAlpha,
                        static_cast<Real>(0.0),
                        static_cast<Real>(1.0));
                if (holdAlpha > kEps)
                {
                    desiredBa = safeNormalize(
                        (static_cast<Real>(1.0) - holdAlpha) * desiredBa + holdAlpha * heldBa,
                        desiredBa);
                }
            }
        }
        // The strict steering stack above blends preview, torque-awareness,
        // recenter release, and short-term memory. In some bend states that can
        // still leave the final field with a small outward lateral sign even
        // though the head is visibly eccentric and should be pulled back toward
        // the centerline. Clamp the final lateral sign once more against the
        // true centerline pull so the rendered / applied field never keeps
        // steering outward when the tip offset already demands inward recovery.
        if (centerlinePullNormMm > kEps)
        {
            const Real inwardSignNeed = std::clamp(
                std::max(
                    std::max(strictOffsetNeed, strictClearanceNeed),
                    std::max(
                        static_cast<Real>(0.85) * strictCenterlinePullNeed,
                        std::max(
                            static_cast<Real>(0.70) * recenteringAlpha,
                            static_cast<Real>(0.35) * steeringHoldNeed)))
                    + static_cast<Real>(0.18) * smoothstepRange(
                        tipOffsetMm,
                        static_cast<Real>(0.45),
                        static_cast<Real>(1.25))
                    + (barrierActiveNodes > 0u ? static_cast<Real>(0.12) : static_cast<Real>(0.0)),
                static_cast<Real>(0.0),
                static_cast<Real>(1.0));
            if (inwardSignNeed > kEps)
            {
                const Real inwardSignAlpha = std::clamp(
                    static_cast<Real>(0.20) + static_cast<Real>(0.72) * inwardSignNeed,
                    static_cast<Real>(0.0),
                    static_cast<Real>(0.92));
                desiredBa = enforceGuidanceLateralSign(
                    desiredBa,
                    strictForwardAxis,
                    centerlinePullMm,
                    inwardSignAlpha);
            }
        }
    }

    const Real externalFieldScale = std::clamp(d_externalFieldScale.getValue(), static_cast<Real>(0.0), static_cast<Real>(1.0));
    d_debugUpcomingTurnDeg.setValue(m_lastUpcomingTurnDeg);
    d_debugBendSeverity.setValue(bendSeverity);
    d_debugScheduledFieldScaleBase.setValue(scheduledFieldScaleBase);
    d_debugScheduledFieldScale.setValue(scheduledFieldScaleBase);
    d_debugStrictSteeringNeedAlpha.setValue(strictSteeringNeedAlpha);
    d_debugEntryReleaseAlpha.setValue(entryReleaseAlpha);
    d_debugRecenteringAlpha.setValue(recenteringAlpha);
    if (externalFieldScale <= kEps)
    {
        m_elapsedTime = static_cast<Real>(0.0);
        m_lastExternalFieldScale = externalFieldScale;
        m_hasFilteredBaVector = false;
        m_hasAppliedBaVector = false;
        m_hasLastStrictAssistDirection = false;
        m_lastStrictAssistDirection = Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
        d_debugTargetPoint.setValue(targetPoint);
        d_debugLookAheadPoint.setValue(lookAheadPoint);
        d_debugBaVector.setValue(safeNormalize(d_baVectorRef.getValue(), kZAxis));
        d_debugForceVector.setValue(Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
        d_debugTorqueVector.setValue(Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
        d_debugMagneticMomentVector.setValue(aggregateMoment);
        d_debugTorqueSin.setValue(static_cast<Real>(0.0));
        d_debugAssistForceVector.setValue(Vec3(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)));
        d_debugOutwardAssistComponentN.setValue(static_cast<Real>(0.0));
        d_debugDistalTangentFieldAngleDeg.setValue(static_cast<Real>(0.0));
        f.endEdit();
        return;
    }
    if (m_lastExternalFieldScale <= kEps && externalFieldScale > kEps)
        m_elapsedTime = static_cast<Real>(0.0);
    m_lastExternalFieldScale = externalFieldScale;

    Real solverDt = mparams != nullptr ? static_cast<Real>(mparams->dt()) : static_cast<Real>(0.0);
    if (solverDt <= kEps && m_rodModel != nullptr)
        solverDt = std::max(m_rodModel->d_dt.getValue(), static_cast<Real>(0.0));
    const Real externalControlDt = std::max(d_externalControlDt.getValue(), static_cast<Real>(0.0));
    const Real controlDt = externalControlDt > kEps ? externalControlDt : solverDt;
    const Real timeScale = (solverDt > kEps && controlDt > kEps)
        ? std::max(controlDt / solverDt, static_cast<Real>(1.0))
        : static_cast<Real>(1.0);

    if (strictPhysicalTorqueOnly)
    {
        Real maxTurnRad = std::max(d_maxFieldTurnAngleDeg.getValue(), static_cast<Real>(0.0)) * kPi / static_cast<Real>(180.0);
        const bool contactLike = (
            barrierActiveNodes > 0u
            || (std::isfinite(minLumenClearanceMm) && minLumenClearanceMm < static_cast<Real>(0.30))
        );
        // Keep strict GUI steering wall-clock aware through the field ramp, but
        // do not let a long render frame multiply the per-step turn cap into an
        // effectively unbounded snap. That was showing up as head jitter and
        // node-wise twist once the first bend/contact region lowered solver dt.
        if (timeScale > static_cast<Real>(1.0) + kEps)
        {
            const Real turnTimeScale = contactLike
                ? std::clamp(
                    static_cast<Real>(1.0) + static_cast<Real>(0.10) * (timeScale - static_cast<Real>(1.0)),
                    static_cast<Real>(1.0),
                    static_cast<Real>(1.30))
                : std::clamp(
                    static_cast<Real>(1.0) + static_cast<Real>(0.16) * (timeScale - static_cast<Real>(1.0)),
                    static_cast<Real>(1.0),
                    static_cast<Real>(1.55));
            maxTurnRad *= turnTimeScale;
        }
        if (contactLike)
        {
            const Real contactTurnScale = std::clamp(
                static_cast<Real>(0.72)
                    - static_cast<Real>(0.08) * bendSeverity
                    - static_cast<Real>(0.06) * recenteringAlpha,
                static_cast<Real>(0.50),
                static_cast<Real>(0.78));
            const Real branchCommitTurnBoost = std::clamp(
                static_cast<Real>(0.18) * strictBranchCommitFloor
                    + static_cast<Real>(0.10) * std::max(strictTurnNeed, strictTargetPullNeed)
                    + static_cast<Real>(0.08) * std::max(strictClearanceNeed, static_cast<Real>(0.5) * strictCenterlinePullNeed),
                static_cast<Real>(0.0),
                static_cast<Real>(0.16));
            maxTurnRad *= std::clamp(
                contactTurnScale + branchCommitTurnBoost,
                static_cast<Real>(0.50),
                static_cast<Real>(0.92));
        }
        else if (maxTurnRad > kEps)
        {
            maxTurnRad *= static_cast<Real>(1.0) + static_cast<Real>(0.10) * bendSeverity;
        }
        if (!m_hasAppliedBaVector || maxTurnRad <= kEps)
            m_appliedBaVector = desiredBa;
        else
            m_appliedBaVector = rotateToward(m_appliedBaVector, desiredBa, maxTurnRad);
        m_filteredBaVector = m_appliedBaVector;
    }
    else
    {
        const bool contactLike = (
            externalSurfaceContactActive
            || barrierActiveNodes > 0u
            || (std::isfinite(minLumenClearanceMm) && minLumenClearanceMm < static_cast<Real>(0.45))
            || tipOffsetMm > static_cast<Real>(0.85));
        const Real smoothingAlpha = std::clamp(d_fieldSmoothingAlpha.getValue(), static_cast<Real>(0.0), static_cast<Real>(1.0));
        Real effectiveSmoothingAlpha = smoothingAlpha;
        if (smoothingAlpha > kEps && smoothingAlpha < static_cast<Real>(1.0) && timeScale > static_cast<Real>(1.0) + kEps)
        {
            // In GUI / wall-clock mode the control dt can be an order of
            // magnitude larger than the solver dt. Using the raw timeScale here
            // effectively disables safe-path smoothing exactly when the head
            // first reaches the bend, which shows up as jitter and local twist.
            const Real smoothingTimeScale = contactLike
                ? std::clamp(
                    static_cast<Real>(1.0) + static_cast<Real>(0.08) * (timeScale - static_cast<Real>(1.0)),
                    static_cast<Real>(1.0),
                    static_cast<Real>(1.22))
                : std::clamp(
                    static_cast<Real>(1.0) + static_cast<Real>(0.16) * (timeScale - static_cast<Real>(1.0)),
                    static_cast<Real>(1.0),
                    static_cast<Real>(1.45));
            effectiveSmoothingAlpha = static_cast<Real>(1.0) - std::pow(
                static_cast<Real>(1.0) - smoothingAlpha,
                smoothingTimeScale);
        }
        if (!m_hasFilteredBaVector || effectiveSmoothingAlpha <= kEps)
            m_filteredBaVector = desiredBa;
        else
            m_filteredBaVector = safeNormalize(
                (static_cast<Real>(1.0) - effectiveSmoothingAlpha) * m_filteredBaVector + effectiveSmoothingAlpha * desiredBa,
                desiredBa
            );

        Real maxTurnRad = std::max(d_maxFieldTurnAngleDeg.getValue(), static_cast<Real>(0.0)) * kPi / static_cast<Real>(180.0);
        if (timeScale > static_cast<Real>(1.0) + kEps)
        {
            // The safe path previously multiplied the turn cap by the full
            // wall-clock/solver ratio, so a single slow GUI frame could let the
            // applied field snap almost directly to the new target. That abrupt
            // steering change is a major source of the distal-head wring near the
            // first bend.
            const Real turnTimeScale = contactLike
                ? std::clamp(
                    static_cast<Real>(1.0) + static_cast<Real>(0.07) * (timeScale - static_cast<Real>(1.0)),
                    static_cast<Real>(1.0),
                    static_cast<Real>(1.20))
                : std::clamp(
                    static_cast<Real>(1.0) + static_cast<Real>(0.14) * (timeScale - static_cast<Real>(1.0)),
                    static_cast<Real>(1.0),
                    static_cast<Real>(1.45));
            maxTurnRad *= turnTimeScale;
        }
        if (contactLike)
        {
            const Real contactOffsetGate = smoothstepRange(
                tipOffsetMm,
                static_cast<Real>(0.45),
                static_cast<Real>(1.20));
            const Real contactClearanceGate = std::isfinite(minLumenClearanceMm)
                ? smoothstepRange(
                    static_cast<Real>(0.65) - minLumenClearanceMm,
                    static_cast<Real>(0.0),
                    static_cast<Real>(0.40))
                : static_cast<Real>(0.0);
            const Real contactTurnScale = std::clamp(
                static_cast<Real>(0.70)
                    + static_cast<Real>(0.08) * contactClearanceGate
                    + static_cast<Real>(0.08) * contactOffsetGate,
                static_cast<Real>(0.68),
                static_cast<Real>(0.82));
            maxTurnRad *= contactTurnScale;
        }
        if (!m_hasAppliedBaVector || maxTurnRad <= kEps)
            m_appliedBaVector = m_filteredBaVector;
        else
            m_appliedBaVector = rotateToward(m_appliedBaVector, m_filteredBaVector, maxTurnRad);
    }

    Real appliedTorqueSin = static_cast<Real>(0.0);
    if (hasMoment)
    {
        const Vec3 momentDirection = safeNormalize(aggregateMoment, momentFallback);
        appliedTorqueSin = std::clamp((momentDirection.cross(m_appliedBaVector)).norm(), static_cast<Real>(0.0), static_cast<Real>(1.0));
        if (
            (!strictPhysicalTorqueOnly)
            && (appliedTorqueSin + static_cast<Real>(1.0e-6) < std::clamp(d_minTorqueSin.getValue(), static_cast<Real>(0.0), static_cast<Real>(0.999)))
        )
        {
            m_filteredBaVector = desiredBa;
            m_appliedBaVector = desiredBa;
            appliedTorqueSin = std::clamp((momentDirection.cross(m_appliedBaVector)).norm(), static_cast<Real>(0.0), static_cast<Real>(1.0));
        }
    }

    m_hasFilteredBaVector = true;
    m_hasAppliedBaVector = true;

    d_debugTargetPoint.setValue(targetPoint);
    d_debugLookAheadPoint.setValue(lookAheadPoint);
    d_debugBaVector.setValue(m_appliedBaVector);
    d_debugMagneticMomentVector.setValue(aggregateMoment);
    d_debugTorqueSin.setValue(appliedTorqueSin);
    const Real distalTangentFieldAngleDeg = std::acos(std::clamp(sofa::type::dot(safeNormalize(distalTangent, nearestTangent), safeNormalize(m_appliedBaVector, nearestTangent)), static_cast<Real>(-1.0), static_cast<Real>(1.0))) * static_cast<Real>(180.0) / kPi;
    d_debugDistalTangentFieldAngleDeg.setValue(distalTangentFieldAngleDeg);
    const Real contactFieldAngleGate = std::clamp(
        (distalTangentFieldAngleDeg - static_cast<Real>(55.0)) / static_cast<Real>(25.0),
        static_cast<Real>(0.0),
        static_cast<Real>(1.0));
    const bool recoveryActive = (
        barrierActiveNodes > 0u
        || minLumenClearanceMm < static_cast<Real>(0.8)
        || (distalTangentFieldAngleDeg > static_cast<Real>(12.0) && tipOffsetMm > static_cast<Real>(0.6))
    );
    const Real misalignmentGain = std::clamp(distalTangentFieldAngleDeg / static_cast<Real>(35.0), static_cast<Real>(0.0), static_cast<Real>(1.25));
    const Real offsetGain = std::clamp((tipOffsetMm - static_cast<Real>(0.6)) / static_cast<Real>(1.4), static_cast<Real>(0.0), static_cast<Real>(1.0));
    const Real clearanceGain = std::isfinite(minLumenClearanceMm)
        ? std::clamp((static_cast<Real>(0.8) - minLumenClearanceMm) / static_cast<Real>(0.8), static_cast<Real>(0.0), static_cast<Real>(1.0))
        : static_cast<Real>(0.0);
    Real assistGain = static_cast<Real>(1.0) + static_cast<Real>(0.35) * misalignmentGain;
    if (recoveryActive)
    {
        assistGain += static_cast<Real>(0.65) * misalignmentGain
            + static_cast<Real>(0.85) * offsetGain
            + static_cast<Real>(0.85) * clearanceGain;
        assistGain = std::max(assistGain, static_cast<Real>(1.5));
    }
    if (recenteringAlpha > kEps)
    {
        assistGain = (static_cast<Real>(1.0) - recenteringAlpha) * assistGain + recenteringAlpha;
    }
    Real safeContactReliefAlpha = static_cast<Real>(0.0);
    if (!strictPhysicalTorqueOnly)
    {
        const Real safeHeadStretchReliefStart = std::max(
            static_cast<Real>(0.20) * headStretchReliefStart,
            static_cast<Real>(0.0010));
        const Real safeHeadStretchReliefFull = std::max(
            static_cast<Real>(0.50) * headStretchReliefFull,
            safeHeadStretchReliefStart + static_cast<Real>(1.0e-6));
        const Real safeHeadStretchReliefAlpha = std::clamp(
            (currentMaxHeadStretch - safeHeadStretchReliefStart)
                / (safeHeadStretchReliefFull - safeHeadStretchReliefStart),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        const Real safeBarrierReliefAlpha = barrierActiveNodes > 0u
            ? smoothstepRange(
                static_cast<Real>(barrierActiveNodes),
                static_cast<Real>(1.0),
                static_cast<Real>(3.0))
            : static_cast<Real>(0.0);
        safeContactReliefAlpha = std::clamp(
            std::max(
                safeHeadStretchReliefAlpha,
                std::max(
                    static_cast<Real>(1.10) * clearanceGain,
                    std::max(contactFieldAngleGate, safeBarrierReliefAlpha))),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        if (safeContactReliefAlpha > kEps)
        {
            const Real safeAssistGainFloor = static_cast<Real>(1.15);
            assistGain = (static_cast<Real>(1.0) - safeContactReliefAlpha) * assistGain
                + safeContactReliefAlpha * safeAssistGainFloor;
        }
    }

    m_elapsedTime += std::max(controlDt, static_cast<Real>(0.0));
    Real fieldScale = externalFieldScale * scheduledFieldScaleBase;
    const Real rampTime = std::max(d_fieldRampTime.getValue(), static_cast<Real>(0.0));
    if (fieldScale > kEps && rampTime > kEps)
    {
        const Real alpha = std::clamp(m_elapsedTime / rampTime, static_cast<Real>(0.0), static_cast<Real>(1.0));
        fieldScale *= static_cast<Real>(0.5) - static_cast<Real>(0.5) * std::cos(kPi * alpha);
    }
    Real assistFieldScale = fieldScale;
    if (strictPhysicalTorqueOnly)
    {
        const Real steeringAuthorityAlpha = std::clamp(
            std::max(entryReleaseAlpha * strictSteeringNeedAlpha, strictBranchCommitFloor),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        fieldScale *= steeringAuthorityAlpha;
        // Keep the actual torque steering conservative, but allow a small
        // purely lateral tip-centering action to start earlier once the head is
        // measurably offset or approaching the wall. This targets the user's
        // desired "pull back to centerline" behavior without injecting forward
        // traction or altering the push/insertion pipeline.
        const Real strictCenteringGate = std::clamp(
            std::max(
                static_cast<Real>(0.20) * entryReleaseAlpha,
                static_cast<Real>(0.55) * std::max(
                    std::max(strictOffsetNeed, strictClearanceNeed),
                    static_cast<Real>(0.85) * strictCenterlinePullNeed)),
            static_cast<Real>(0.0),
            static_cast<Real>(1.0));
        assistFieldScale *= std::max(steeringAuthorityAlpha, strictCenteringGate);
        if (headStretchGate > kEps)
        {
            const Real floorScale = std::clamp(
                d_headStretchFieldScaleFloor.getValue(),
                static_cast<Real>(0.0),
                static_cast<Real>(1.0));
            fieldScale *= (static_cast<Real>(1.0) - headStretchGate) + headStretchGate * floorScale;
            assistFieldScale *= (static_cast<Real>(1.0) - headStretchGate) + headStretchGate * floorScale;
        }
    }
    else if (safeContactReliefAlpha > kEps)
    {
        // Safe mode still needs meaningful steering authority once the head is
        // grazing the wall; collapsing the field almost to zero makes the tip
        // keep pushing straight into the first bend instead of turning inward.
        const Real safeTorqueFloor = static_cast<Real>(0.18);
        const Real safeAssistFloor = static_cast<Real>(0.52);
        fieldScale *= (static_cast<Real>(1.0) - safeContactReliefAlpha) + safeContactReliefAlpha * safeTorqueFloor;
        assistFieldScale *= (static_cast<Real>(1.0) - safeContactReliefAlpha) + safeContactReliefAlpha * safeAssistFloor;
    }
    d_debugScheduledFieldScale.setValue(fieldScale);
    Vec3 debugTipForce(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Vec3 debugTotalTorque(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Vec3 debugAssistForce(static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0));
    Real debugOutwardAssistComponent = static_cast<Real>(0.0);
    const Vec3 guidancePointMm = ((lookAheadPoint - tipPosMm).norm() > kEps) ? lookAheadPoint : targetPoint;
    computeMagneticForces(q, currentState, targetDirection, guidancePointMm, m_appliedBaVector, fieldScale, assistFieldScale, assistGain, fq, &debugTipForce, &debugTotalTorque, &debugAssistForce, &debugOutwardAssistComponent);
    d_debugForceVector.setValue(debugTipForce);
    d_debugTorqueVector.setValue(debugTotalTorque);
    d_debugAssistForceVector.setValue(debugAssistForce);
    d_debugOutwardAssistComponentN.setValue(debugOutwardAssistComponent);
    f.endEdit();
}

void ExternalMagneticForceField::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(dx);
    if (df.isSet())
    {
        VecDeriv& dforce = *df.beginEdit();
        df.endEdit();
        SOFA_UNUSED(dforce);
    }
}

void ExternalMagneticForceField::addKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset)
{
    SOFA_UNUSED(matrix);
    SOFA_UNUSED(kFact);
    SOFA_UNUSED(offset);
}

SReal ExternalMagneticForceField::getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord&) const
{
    return static_cast<SReal>(0.0);
}

int ExternalMagneticForceFieldClass = sofa::core::RegisterObject(
    "Reduced-state native magnetic steering force field aligned with externalMagneticForce semantics."
).add<ExternalMagneticForceField>();

} // namespace elastic_rod_guidewire
