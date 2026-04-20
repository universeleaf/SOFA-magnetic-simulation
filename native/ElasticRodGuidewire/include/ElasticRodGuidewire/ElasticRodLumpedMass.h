#pragma once

#include <ElasticRodGuidewire/ElasticRodCompatCore.h>
#include <ElasticRodGuidewire/ElasticRodTypes.h>
#include <ElasticRodGuidewire/config.h>

#include <sofa/core/behavior/Mass.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/type/vector.h>

namespace elastic_rod_guidewire
{

class SOFA_ELASTICRODGUIDEWIRE_API ElasticRodLumpedMass
    : public sofa::core::behavior::Mass<CarrierTypes>
{
public:
    using DataTypes = CarrierTypes;
    using Inherit = sofa::core::behavior::Mass<DataTypes>;
    using Real = typename DataTypes::Real;
    using Coord = CarrierCoord;
    using Deriv = CarrierDeriv;
    using VecCoord = CarrierVecCoord;
    using VecDeriv = CarrierVecDeriv;
    using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;
    using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
    using Vec3 = sofa::type::Vec<3, Real>;
    using Vec6 = sofa::type::Vec<6, Real>;
    using VecReal = sofa::type::vector<Real>;
    using VecVec3 = sofa::type::vector<Vec3>;

    SOFA_CLASS(ElasticRodLumpedMass, Inherit);

    ElasticRodLumpedMass();
    ~ElasticRodLumpedMass() override = default;

    void init() override;
    void reinit() override;
    void addMDx(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;
    void accFromF(const sofa::core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f) override;
    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    SReal getKineticEnergy(const sofa::core::MechanicalParams* mparams, const DataVecDeriv& v) const override;
    SReal getPotentialEnergy(const sofa::core::MechanicalParams* mparams, const DataVecCoord& x) const override;
    Vec6 getMomentum(const sofa::core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const override;
    void addGravityToV(const sofa::core::MechanicalParams* mparams, DataVecDeriv& d_v) override;
    void addMToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal mFact, unsigned int& offset) override;
    SReal getElementMass(sofa::Index index) const override;
    void getElementMass(sofa::Index index, sofa::linearalgebra::BaseMatrix* m) const override;
    bool isDiagonal() const override { return true; }

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
    sofa::core::objectmodel::Data<Real> d_rodLength;
    sofa::core::objectmodel::Data<unsigned int> d_magneticEdgeCount;
    sofa::core::objectmodel::Data<unsigned int> d_softTipEdgeCount;
    sofa::core::objectmodel::Data<VecReal> d_debugLumpedMass;
    sofa::core::objectmodel::Data<VecReal> d_debugLumpedRotInertia;
    sofa::core::objectmodel::Data<Real> d_debugTotalMass;

private:
    static constexpr Real kMmToM = static_cast<Real>(1.0e-3);

    void configureCoreFromData(const VecCoord& positions);

    ElasticRodCompatCore m_core;
    std::size_t m_cachedNodeCount {0};
    std::vector<Real> m_effectiveThetaInertia;
    std::vector<Real> m_dummyInertia;
};

} // namespace elastic_rod_guidewire
