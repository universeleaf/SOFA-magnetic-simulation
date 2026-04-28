#pragma once

#include <ElasticRodGuidewire/ElasticRodCompatCore.h>
#include <ElasticRodGuidewire/ElasticRodTypes.h>
#include <ElasticRodGuidewire/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#include <utility>
#include <vector>

namespace elastic_rod_guidewire
{

class ElasticRodGuidewireModel;

class SOFA_ELASTICRODGUIDEWIRE_API ExternalMagneticForceField
    : public sofa::core::behavior::ForceField<CarrierTypes>
{
public:
    using DataTypes = CarrierTypes;
    using Inherit = sofa::core::behavior::ForceField<DataTypes>;
    using Real = typename DataTypes::Real;
    using Coord = CarrierCoord;
    using Deriv = CarrierDeriv;
    using VecCoord = CarrierVecCoord;
    using VecDeriv = CarrierVecDeriv;
    using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;
    using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
    using Vec3 = sofa::type::Vec<3, Real>;
    using VecVec3 = sofa::type::vector<Vec3>;

    SOFA_CLASS(ExternalMagneticForceField, Inherit);

    ExternalMagneticForceField();
    ~ExternalMagneticForceField() override = default;

    void init() override;
    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    void addKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset) override;
    SReal getPotentialEnergy(const sofa::core::MechanicalParams* mparams, const DataVecCoord& x) const override;

    sofa::core::objectmodel::Data<VecVec3> d_tubeNodes;
    sofa::core::objectmodel::Data<Vec3> d_brVector;
    sofa::core::objectmodel::Data<Vec3> d_baVectorRef;
    sofa::core::objectmodel::Data<Real> d_muZero;
    sofa::core::objectmodel::Data<Real> d_rodRadius;
    sofa::core::objectmodel::Data<Real> d_magneticCoreRadiusMm;
    sofa::core::objectmodel::Data<unsigned int> d_magneticEdgeCount;
    sofa::core::objectmodel::Data<Real> d_lookAheadDistance;
    sofa::core::objectmodel::Data<Real> d_recoveryLookAheadDistance;
    sofa::core::objectmodel::Data<Real> d_fieldSmoothingAlpha;
    sofa::core::objectmodel::Data<Real> d_maxFieldTurnAngleDeg;
    sofa::core::objectmodel::Data<Real> d_fieldRampTime;
    sofa::core::objectmodel::Data<Real> d_minTorqueSin;
    sofa::core::objectmodel::Data<Real> d_lateralForceScale;
    sofa::core::objectmodel::Data<Real> d_entryStraightDistance;
    sofa::core::objectmodel::Data<Real> d_entrySteeringReleaseDistance;
    sofa::core::objectmodel::Data<Real> d_bendLookAheadDistance;
    sofa::core::objectmodel::Data<Real> d_bendNearWindowDistance;
    sofa::core::objectmodel::Data<Real> d_bendTurnMediumDeg;
    sofa::core::objectmodel::Data<Real> d_bendTurnHighDeg;
    sofa::core::objectmodel::Data<Real> d_fieldScaleStraight;
    sofa::core::objectmodel::Data<Real> d_fieldScaleBend;
    sofa::core::objectmodel::Data<Real> d_recenterClearanceMm;
    sofa::core::objectmodel::Data<Real> d_recenterOffsetMm;
    sofa::core::objectmodel::Data<Real> d_recenterBlend;
    sofa::core::objectmodel::Data<Real> d_headStretchReliefStart;
    sofa::core::objectmodel::Data<Real> d_headStretchReliefFull;
    sofa::core::objectmodel::Data<Real> d_headStretchFieldScaleFloor;
    sofa::core::objectmodel::Data<bool> d_strictPhysicalTorqueOnly;
    sofa::core::objectmodel::Data<Real> d_externalFieldScale;
    sofa::core::objectmodel::Data<Real> d_externalControlDt;
    sofa::core::objectmodel::Data<bool> d_useExternalTargetDirection;
    sofa::core::objectmodel::Data<Vec3> d_externalTargetDirection;
    sofa::core::objectmodel::Data<Real> d_externalSurfaceClearanceMm;
    sofa::core::objectmodel::Data<bool> d_externalSurfaceContactActive;
    sofa::core::objectmodel::Data<Vec3> d_debugTargetPoint;
    sofa::core::objectmodel::Data<Vec3> d_debugLookAheadPoint;
    sofa::core::objectmodel::Data<Vec3> d_debugBaVector;
    sofa::core::objectmodel::Data<Vec3> d_debugForceVector;
    sofa::core::objectmodel::Data<Vec3> d_debugTorqueVector;
    sofa::core::objectmodel::Data<Vec3> d_debugMagneticMomentVector;
    sofa::core::objectmodel::Data<Real> d_debugTorqueSin;
    sofa::core::objectmodel::Data<Vec3> d_debugAssistForceVector;
    sofa::core::objectmodel::Data<bool> d_strictInLumenMode;
    sofa::core::objectmodel::Data<Real> d_debugOutwardAssistComponentN;
    sofa::core::objectmodel::Data<Real> d_debugDistalTangentFieldAngleDeg;
    sofa::core::objectmodel::Data<Real> d_debugUpcomingTurnDeg;
    sofa::core::objectmodel::Data<Real> d_debugBendSeverity;
    sofa::core::objectmodel::Data<Real> d_debugScheduledFieldScale;
    sofa::core::objectmodel::Data<Real> d_debugScheduledFieldScaleBase;
    sofa::core::objectmodel::Data<Real> d_debugStrictSteeringNeedAlpha;
    sofa::core::objectmodel::Data<Real> d_debugEntryReleaseAlpha;
    sofa::core::objectmodel::Data<Real> d_debugRecenteringAlpha;

private:
    void rebuildTubeArc();
    Vec3 interpolateTubePoint(Real s) const;
    Vec3 interpolateTubeTangent(Real s, Real sampleDs, const Vec3& fallback) const;
    Vec3 computeLookAheadTargetDirection(const VecCoord& positions, Vec3& targetPoint, Vec3* lookAheadPoint = nullptr) const;
    Vec3 computeNearestTubeTangentDirection(const VecCoord& positions) const;
    void computeMagneticForces(
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
        Real* debugOutwardAssistComponent) const;
    std::pair<std::size_t, std::size_t> activeNodeRange(std::size_t nodeCount) const;

    std::vector<Real> m_tubeCum;
    Vec3 m_filteredBaVector;
    Vec3 m_appliedBaVector;
    bool m_hasFilteredBaVector {false};
    bool m_hasAppliedBaVector {false};
    mutable bool m_hasLastTargetArcS {false};
    mutable Real m_lastTargetArcS {static_cast<Real>(0.0)};
    mutable Real m_lastStrictEntrySteeringAlpha {static_cast<Real>(1.0)};
    mutable Vec3 m_lastLocalForwardTangent {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(1.0)};
    mutable Real m_lastUpcomingTurnDeg {static_cast<Real>(0.0)};
    mutable Real m_lastBendSeverity {static_cast<Real>(0.0)};
    mutable Vec3 m_lastStrictAssistDirection {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
    mutable bool m_hasLastStrictAssistDirection {false};
    Real m_elapsedTime {static_cast<Real>(0.0)};
    Real m_lastExternalFieldScale {static_cast<Real>(1.0)};
    ElasticRodGuidewireModel* m_rodModel {nullptr};
};

} // namespace elastic_rod_guidewire
