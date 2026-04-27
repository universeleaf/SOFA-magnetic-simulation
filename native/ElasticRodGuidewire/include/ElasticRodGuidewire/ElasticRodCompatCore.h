#pragma once

#include <ElasticRodGuidewire/ElasticRodTypes.h>
#include <ElasticRodGuidewire/config.h>

#include <cstddef>
#include <vector>

namespace elastic_rod_guidewire
{

class SOFA_ELASTICRODGUIDEWIRE_API ElasticRodCompatCore
{
public:
    using DataTypes = CarrierTypes;
    using Real = typename DataTypes::Real;
    using Coord = CarrierCoord;
    using Deriv = CarrierDeriv;
    using Vec3 = elastic_rod_guidewire::Vec3;
    using Vec2 = elastic_rod_guidewire::Vec2;
    using Quat = elastic_rod_guidewire::Quat;

    struct FrameAxes
    {
        Vec3 m1;
        Vec3 m2;
        Vec3 m3;
    };

    struct State
    {
        std::vector<Vec3> centersM;
        std::vector<Real> theta;
        std::vector<Real> edgeLenM;
        std::vector<Vec3> tangents;
        std::vector<FrameAxes> referenceFrames;
        std::vector<FrameAxes> materialFrames;
        std::vector<Vec3> kb;
        std::vector<Vec2> kappa;
        std::vector<Real> refTwist;
        std::vector<Real> twist;
    };

    void configure(
        Real rho,
        Real mechanicalCoreRadiusMm,
        Real dt,
        Real youngHead,
        Real youngBody,
        Real shearHead,
        Real shearBody,
        Real rodLengthMm,
        std::size_t magneticEdgeCount,
        std::size_t softTipEdgeCount = 0u,
        const std::vector<Real>& edgeEAProfile = {},
        const std::vector<Real>& edgeEIProfile = {},
        const std::vector<Real>& edgeGJProfile = {});

    void initialize(const std::vector<Coord>& initialCoords, const std::vector<Vec3>& undeformedNodesMm);
    State computeState(const std::vector<Coord>& coords) const;
    void commitState(const State& state);

    void computeBoundaryTargets(
        Real commandedInsertionMm,
        Real commandedTwistRad,
        const Vec3& insertionDirection,
        std::size_t pushNodeCount,
        std::vector<Vec3>& targetCentersMm,
        std::vector<Real>& targetTheta) const;
    Real supportArcLengthMm(std::size_t nodeIndex) const;

    static Vec3 safeNormalize(const Vec3& v, const Vec3& fallback);
    static Vec3 parallelTransport(const Vec3& d1, const Vec3& t1, const Vec3& t2);
    static Real signedAngle(const Vec3& u, const Vec3& v, const Vec3& n);
    static FrameAxes buildFrameFromTangent(const Vec3& tangentM, const Vec3& preferredM1);
    static FrameAxes rotateFrameAboutAxis(const FrameAxes& frame, Real theta);
    static Quat quatFromFrame(const FrameAxes& frame);
    static Quat quatFromZTo(const Vec3& direction);

    std::size_t nodeCount() const { return m_initialCentersMm.size(); }
    std::size_t edgeCount() const { return m_refLen.size(); }
    std::size_t magneticEdgeCount() const { return m_magneticEdgeCount; }
    std::size_t magneticStartEdge() const;

    Real rodRadiusM() const { return m_mechanicalCoreRadiusM; }
    Real mechanicalCoreRadiusM() const { return m_mechanicalCoreRadiusM; }
    Real rodLengthM() const { return m_rodLengthM; }
    Real totalMassKg() const { return m_totalMassKg; }

    const std::vector<Real>& refLen() const { return m_refLen; }
    const std::vector<Real>& voronoiLen() const { return m_voronoiLen; }
    const std::vector<Real>& EA() const { return m_EA; }
    const std::vector<Real>& EI() const { return m_EI; }
    const std::vector<Real>& GJ() const { return m_GJ; }
    const std::vector<Vec2>& kappaBar() const { return m_kappaBar; }
    const std::vector<Real>& undeformedTwist() const { return m_undeformedTwist; }
    const std::vector<Real>& lumpedMassKg() const { return m_lumpedMassKg; }
    const std::vector<Real>& lumpedThetaInertiaKgM2() const { return m_lumpedThetaInertiaKgM2; }
    const std::vector<FrameAxes>& restFrames() const { return m_restFrames; }
    const std::vector<FrameAxes>& committedReferenceFrames() const { return m_committedReferenceFrames; }
    const std::vector<Vec3>& committedTangents() const { return m_committedTangents; }
    const std::vector<Real>& committedRefTwist() const { return m_committedRefTwist; }
    const std::vector<Vec3>& initialCentersMm() const { return m_initialCentersMm; }
    const std::vector<Vec3>& undeformedCentersM() const { return m_undeformedCentersM; }

private:
    std::vector<Vec3> buildUndeformedCentersM(const std::vector<Coord>& initialCoords, const std::vector<Vec3>& undeformedNodesMm) const;
    void initializeCommittedReference(const std::vector<Coord>& initialCoords);
    void computeReferenceGeometry();
    void computeMassDistribution();
    void computeStiffness();
    void computeRestCurvature();

    Real m_rho {7800.0};
    Real m_mechanicalCoreRadiusM {0.20e-3};
    Real m_dt {1.0e-4};
    Real m_youngHead {1.8e10};
    Real m_youngBody {5.5e10};
    Real m_shearHead {6.766917293233083e9};
    Real m_shearBody {2.067669172932331e10};
    Real m_rodLengthM {400.0e-3};
    std::size_t m_magneticEdgeCount {5};
    std::size_t m_softTipEdgeCount {0};
    std::vector<Real> m_edgeEAProfile;
    std::vector<Real> m_edgeEIProfile;
    std::vector<Real> m_edgeGJProfile;

    std::vector<Vec3> m_initialCentersMm;
    std::vector<Vec3> m_undeformedCentersM;
    std::vector<Real> m_refLen;
    std::vector<Real> m_voronoiLen;
    std::vector<Real> m_EA;
    std::vector<Real> m_EI;
    std::vector<Real> m_GJ;
    std::vector<Real> m_lumpedMassKg;
    std::vector<Real> m_lumpedThetaInertiaKgM2;
    std::vector<FrameAxes> m_restFrames;
    std::vector<Vec2> m_kappaBar;
    std::vector<Real> m_undeformedTwist;
    std::vector<FrameAxes> m_committedReferenceFrames;
    std::vector<Vec3> m_committedTangents;
    std::vector<Real> m_committedRefTwist;
    Real m_totalMassKg {0.0};
};

} // namespace elastic_rod_guidewire
