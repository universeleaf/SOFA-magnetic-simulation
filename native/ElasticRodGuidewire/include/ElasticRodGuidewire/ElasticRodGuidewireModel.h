#pragma once

#include <ElasticRodGuidewire/ElasticRodCompatCore.h>
#include <ElasticRodGuidewire/ElasticRodTypes.h>
#include <ElasticRodGuidewire/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/vector.h>

#include <array>
#include <vector>

namespace elastic_rod_guidewire
{

class SOFA_ELASTICRODGUIDEWIRE_API ElasticRodGuidewireModel
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
    using Vec3 = elastic_rod_guidewire::Vec3;
    using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;
    using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
    using VecReal = sofa::type::vector<Real>;
    using VecVec3 = sofa::type::vector<Vec3>;
    using VecUInt = sofa::type::vector<unsigned int>;
    using RigidTypes = sofa::defaulttype::Rigid3Types;
    using RigidCoord = typename RigidTypes::Coord;
    using RigidDeriv = typename RigidTypes::Deriv;
    using RigidVecCoord = typename RigidTypes::VecCoord;
    using RigidVecDeriv = typename RigidTypes::VecDeriv;

    SOFA_CLASS(ElasticRodGuidewireModel, Inherit);

    ElasticRodGuidewireModel();
    ~ElasticRodGuidewireModel() override = default;

    void init() override;
    void reinit() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;
    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    void addKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset) override;
    SReal getPotentialEnergy(const sofa::core::MechanicalParams* mparams, const DataVecCoord& x) const override;

    std::size_t magneticEdgeCount() const;
    const ElasticRodCompatCore& compatCore() const { return m_core; }
    ElasticRodCompatCore::State computeCurrentState(const VecCoord& q) const;
    void buildRigidState(const VecCoord& q, RigidVecCoord& out) const;
    void buildRigidVelocity(const VecCoord& q, const VecDeriv& v, RigidVecDeriv& out) const;
    bool useDynamicStrictWindows() const;
    Real supportWindowWeight(std::size_t nodeIndex) const;

    sofa::core::objectmodel::Data<VecVec3> d_initialNodes;
    sofa::core::objectmodel::Data<VecVec3> d_undeformedNodes;
    sofa::core::objectmodel::Data<Real> d_rho;
    sofa::core::objectmodel::Data<Real> d_rodRadius;
    sofa::core::objectmodel::Data<Real> d_mechanicalCoreRadiusMm;
    sofa::core::objectmodel::Data<Real> d_dt;
    sofa::core::objectmodel::Data<Real> d_youngHead;
    sofa::core::objectmodel::Data<Real> d_youngBody;
    sofa::core::objectmodel::Data<Real> d_shearHead;
    sofa::core::objectmodel::Data<Real> d_shearBody;
    sofa::core::objectmodel::Data<VecReal> d_edgeEAProfile;
    sofa::core::objectmodel::Data<VecReal> d_edgeEIProfile;
    sofa::core::objectmodel::Data<VecReal> d_edgeGJProfile;
    sofa::core::objectmodel::Data<Real> d_rodLength;
    sofa::core::objectmodel::Data<unsigned int> d_magneticEdgeCount;
    sofa::core::objectmodel::Data<unsigned int> d_softTipEdgeCount;
    sofa::core::objectmodel::Data<unsigned int> d_pushNodeCount;
    sofa::core::objectmodel::Data<unsigned int> d_axialDriveNodeCount;
    sofa::core::objectmodel::Data<bool> d_useDynamicSupportWindows;
    sofa::core::objectmodel::Data<VecUInt> d_supportNodeIndices;
    sofa::core::objectmodel::Data<VecUInt> d_driveNodeIndices;
    sofa::core::objectmodel::Data<Real> d_supportWindowLengthMm;
    sofa::core::objectmodel::Data<Real> d_supportReleaseDistanceMm;
    sofa::core::objectmodel::Data<Real> d_driveWindowLengthMm;
    sofa::core::objectmodel::Data<Real> d_driveWindowOutsideOffsetMm;
    sofa::core::objectmodel::Data<unsigned int> d_driveWindowMinNodeCount;
    sofa::core::objectmodel::Data<Real> d_commandedInsertion;
    sofa::core::objectmodel::Data<Real> d_commandedTwist;
    sofa::core::objectmodel::Data<Vec3> d_insertionDirection;
    sofa::core::objectmodel::Data<VecVec3> d_tubeNodes;
    sofa::core::objectmodel::Data<VecReal> d_tubeRadiiMm;
    sofa::core::objectmodel::Data<VecReal> d_nodeInitialPathSmm;
    sofa::core::objectmodel::Data<Real> d_proximalAxialStiffness;
    sofa::core::objectmodel::Data<Real> d_proximalLateralStiffness;
    sofa::core::objectmodel::Data<Real> d_proximalAngularStiffness;
    sofa::core::objectmodel::Data<Real> d_proximalLinearDamping;
    sofa::core::objectmodel::Data<Real> d_proximalAngularDamping;
    sofa::core::objectmodel::Data<Real> d_edgeAxialDamping;
    sofa::core::objectmodel::Data<Real> d_axialStretchStiffnessScale;
    sofa::core::objectmodel::Data<bool> d_axialStretchUseBodyFloor;
    sofa::core::objectmodel::Data<bool> d_useImplicitStretch;
    sofa::core::objectmodel::Data<bool> d_useImplicitBendTwist;
    sofa::core::objectmodel::Data<bool> d_useKinematicSupportBlock;
    sofa::core::objectmodel::Data<bool> d_strictLumenBarrierEnabled;
    sofa::core::objectmodel::Data<Real> d_strictLumenActivationMarginMm;
    sofa::core::objectmodel::Data<Real> d_strictLumenSafetyMarginMm;
    sofa::core::objectmodel::Data<Real> d_strictLumenBarrierStiffness;
    sofa::core::objectmodel::Data<Real> d_strictLumenBarrierDamping;
    sofa::core::objectmodel::Data<Real> d_strictLumenBarrierMaxForcePerNodeN;
    sofa::core::objectmodel::Data<Real> d_strictLumenEntryExtensionMm;
    sofa::core::objectmodel::Data<Real> d_strictLumenEntrySupportRadiusMm;
    sofa::core::objectmodel::Data<bool> d_commitReferenceStateEachStep;
    sofa::core::objectmodel::Data<VecReal> d_debugRefLen;
    sofa::core::objectmodel::Data<VecReal> d_debugVoronoiLen;
    sofa::core::objectmodel::Data<VecReal> d_debugEA;
    sofa::core::objectmodel::Data<VecReal> d_debugEI;
    sofa::core::objectmodel::Data<VecReal> d_debugGJ;
    sofa::core::objectmodel::Data<VecReal> d_debugEdgeLengthMm;
    sofa::core::objectmodel::Data<VecReal> d_debugStretch;
    sofa::core::objectmodel::Data<VecVec3> d_debugKappa;
    sofa::core::objectmodel::Data<VecReal> d_debugTwist;
    sofa::core::objectmodel::Data<Real> d_debugTipProgress;
    sofa::core::objectmodel::Data<Real> d_debugTotalMass;
    sofa::core::objectmodel::Data<int> d_debugAbnormalEdgeIndex;
    sofa::core::objectmodel::Data<Real> d_debugAbnormalEdgeLengthMm;
    sofa::core::objectmodel::Data<Real> d_debugAbnormalEdgeRefLengthMm;
    sofa::core::objectmodel::Data<Real> d_debugMaxAxialBoundaryErrorMm;
    sofa::core::objectmodel::Data<Real> d_debugMaxLateralBoundaryErrorMm;
    sofa::core::objectmodel::Data<Real> d_debugMaxInternalForceN;
    sofa::core::objectmodel::Data<Real> d_debugMaxStretchForceN;
    sofa::core::objectmodel::Data<Real> d_debugMaxBoundaryForceN;
    sofa::core::objectmodel::Data<Real> d_debugMaxBoundaryTorqueNm;
    sofa::core::objectmodel::Data<Real> d_debugDriveReactionN;
    sofa::core::objectmodel::Data<Real> d_debugMaxBendResidual;
    sofa::core::objectmodel::Data<Real> d_debugMinLumenClearanceMm;
    sofa::core::objectmodel::Data<Vec3> d_debugBarrierForceVector;
    sofa::core::objectmodel::Data<unsigned int> d_debugBarrierActiveNodeCount;
    sofa::core::objectmodel::Data<Real> d_debugMaxHeadStretch;

private:
    static constexpr Real kMmToM = static_cast<Real>(1.0e-3);
    static constexpr unsigned int kLocalNodeCount = 3;
    static constexpr unsigned int kActiveNodeDofCount = 4;
    static constexpr unsigned int kLocalDofCount = kLocalNodeCount * kActiveNodeDofCount;
    static constexpr unsigned int kResidualCount = 3;

    struct LocalBendTwistBlock
    {
        bool active {false};
        std::array<std::size_t, kLocalNodeCount> nodes {0, 0, 0};
        std::array<Real, kResidualCount> residual {static_cast<Real>(0.0), static_cast<Real>(0.0), static_cast<Real>(0.0)};
        std::array<Real, kLocalDofCount> forceSI {};
        std::array<Real, kLocalDofCount * kLocalDofCount> stiffnessSI {};
    };

    struct LocalBendTwistEvaluation
    {
        bool valid {false};
        Vec2 deltaKappa {static_cast<Real>(0.0), static_cast<Real>(0.0)};
        Real twistError {static_cast<Real>(0.0)};
        Real bendCoeff {static_cast<Real>(0.0)};
        Real twistCoeff {static_cast<Real>(0.0)};
        Real energySI {static_cast<Real>(0.0)};
    };

    void configureCoreFromData(const VecCoord& positions);
    void refreshBendTwistCache(const VecCoord& q);
    void updateDebugState(const VecCoord& q);

    Real stretchEnergySI(const Coord& a, const Coord& b, std::size_t edgeIndex) const;
    Real effectiveAxialEA(std::size_t edgeIndex) const;
    Real bendTwistEnergySI(const VecCoord& q, std::size_t interiorIndex) const;
    Real boundaryEnergySI(const VecCoord& q) const;

    Real accumulateStretchForces(const VecCoord& q, const VecDeriv& v, VecDeriv& f) const;
    void accumulateBendTwistForces(const VecCoord& q, VecDeriv& f) const;
    void accumulateBoundaryForces(const VecCoord& q, const VecDeriv& v, VecDeriv& f, Real& maxForceN, Real& maxTorqueNm, Real& driveReactionN) const;

    void applyStretchDForce(const VecCoord& q, const VecDeriv& dx, VecDeriv& df, Real kFactor, Real bFactor) const;
    void applyBendTwistDForce(const VecDeriv& dx, VecDeriv& df, Real kFactor) const;
    void applyBoundaryDForce(const VecCoord& q, const VecDeriv& dx, VecDeriv& df, Real kFactor, Real bFactor) const;

    void addBoundaryKToMatrix(const VecCoord& q, sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset) const;
    void addStretchKToMatrix(const VecCoord& q, sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset) const;
    void addBendTwistKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned int& offset) const;
    void refreshLumenProfile();
    bool hasLumenProfile() const;
    Real tubeRadiusMm(Real s) const;
    bool tubePointAtS(Real s, Vec3& pointMm, Real& radiusMm) const;
    bool projectToTube(const Vec3& point, Vec3& closestPoint, Real& outS) const;
    bool sampleStrictLumenCandidateAtS(
        const Vec3& pointMm,
        Real s,
        Real safetyMarginMm,
        Vec3& projectionMm,
        Real& clearanceMm,
        Vec3& outwardNormalM) const;
    bool sampleStrictEntrySupportConstraint(
        const Vec3& pointMm,
        Real safetyMarginMm,
        Vec3& projectionMm,
        Real& projS,
        Real& clearanceMm,
        Vec3& outwardNormalM) const;
    bool sampleStrictLumenConstraint(
        std::size_t nodeIndex,
        const Vec3& pointMm,
        Real safetyMarginMm,
        Vec3& projectionMm,
        Real& projS,
        Real& clearanceMm,
        Vec3& outwardNormalM) const;
    bool sampleStrictLumenConstraintForEdge(
        std::size_t edgeIndex,
        const Vec3& pointMm,
        Real safetyMarginMm,
        Vec3& projectionMm,
        Real& projS,
        Real& clearanceMm,
        Vec3& outwardNormalM) const;
    bool strictBarrierPointEligible(const Vec3& pointMm) const;
    Real strictBarrierNodeWeight(std::size_t nodeIndex, std::size_t nodeCount, const Vec3& pointMm) const;
    Real strictBarrierEdgeWeight(std::size_t edgeIndex, std::size_t nodeCount, const Vec3& midpointMm) const;
    void applyStrictLumenBarrier(
        const VecCoord& q,
        const VecDeriv& v,
        VecDeriv& f,
        Real& minClearanceMm,
        Vec3& totalBarrierForce,
        unsigned int& activeNodeCount) const;

    void applyPerturbationScene(Coord& coord, unsigned int dof, Real delta) const;
    void applyPerturbationSI(Coord& coord, unsigned int dof, Real delta) const;
    void applyLocalPerturbationSI(VecCoord& q, const LocalBendTwistBlock& block, unsigned int dofIndex, Real delta) const;
    void computeLocalBendTwistEvaluation(const VecCoord& q, std::size_t interiorIndex, LocalBendTwistEvaluation& evaluation) const;
    Real localBendTwistEnergySI(const VecCoord& q, std::size_t interiorIndex) const;
    void computeLocalBendTwistResidual(const VecCoord& q, std::size_t interiorIndex, std::array<Real, kResidualCount>& residual) const;
    void computeLocalBendTwistBlock(const VecCoord& q, std::size_t interiorIndex, LocalBendTwistBlock& block) const;
    void computeBoundaryTargets(std::size_t nodeCount, std::vector<Vec3>& targetCentersMm, std::vector<Real>& targetTheta) const;
    bool hasBoundaryDriver() const;
    bool nodeListedIn(const VecUInt& indices, std::size_t nodeIndex) const;
    Real nodeNominalPathSmm(std::size_t nodeIndex) const;
    bool useFullExternalSupportZone() const;
    bool useFullExternalDriveZone() const;
    Real rigidSupportWeight(std::size_t nodeIndex) const;
    bool driveWindowNodeSelected(std::size_t nodeIndex) const;
    std::size_t primaryBoundaryNodeIndex() const;
    std::size_t supportNodeCount() const;
    std::size_t axialDriveNodeCount() const;
    std::size_t rigidSupportNodeCount() const;
    std::size_t supportReleaseNodeCount() const;
    bool useKinematicSupportBlock() const;
    bool isSupportInteriorEdge(std::size_t edgeIndex) const;
    bool isSupportInteriorBlock(std::size_t interiorIndex) const;
    void projectSupportBlockState(bool updateVelocity);
    Real boundaryWeight(std::size_t idx, std::size_t count) const;
    Real axialDriveWeight(std::size_t idx, std::size_t count) const;
    Real boundaryPenaltyWeight(std::size_t idx, std::size_t count) const;
    Real supportConstitutiveEdgeWeight(std::size_t edgeIndex) const;
    Real supportConstitutiveBlockWeight(std::size_t interiorIndex) const;
    Real blockMatrixValue(const LocalBendTwistBlock& block, unsigned int row, unsigned int col) const;

    ElasticRodCompatCore m_core;
    std::size_t m_cachedNodeCount {0};
    std::vector<LocalBendTwistBlock> m_bendTwistBlocks;
    std::vector<Vec3> m_restSecondDiffM;
    std::vector<Real> m_restTwistDiff;
    VecVec3 m_tubeNodesCached;
    VecReal m_tubeRadiiCachedMm;
    std::vector<Real> m_tubeCum;
    bool m_haveLumenProfile {false};
    Real m_lastProjectedInsertionMm {static_cast<Real>(0.0)};
    Real m_lastProjectedTwistRad {static_cast<Real>(0.0)};
    bool m_haveProjectedBoundaryState {false};
};

} // namespace elastic_rod_guidewire
