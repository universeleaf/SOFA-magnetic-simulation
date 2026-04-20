#pragma once

#include <ElasticRodGuidewire/ElasticRodTypes.h>
#include <ElasticRodGuidewire/config.h>

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace elastic_rod_guidewire
{

class ElasticRodGuidewireModel;

class SOFA_ELASTICRODGUIDEWIRE_API ElasticRodRigidStateAdapter
    : public sofa::core::Mapping<CarrierTypes, sofa::defaulttype::Rigid3Types>
{
public:
    using In = CarrierTypes;
    using Out = sofa::defaulttype::Rigid3Types;
    using Inherit = sofa::core::Mapping<In, Out>;
    using InVecCoord = typename In::VecCoord;
    using InVecDeriv = typename In::VecDeriv;
    using OutVecCoord = typename Out::VecCoord;
    using OutVecDeriv = typename Out::VecDeriv;
    using InDataVecCoord = sofa::core::objectmodel::Data<InVecCoord>;
    using InDataVecDeriv = sofa::core::objectmodel::Data<InVecDeriv>;
    using OutDataVecCoord = sofa::core::objectmodel::Data<OutVecCoord>;
    using OutDataVecDeriv = sofa::core::objectmodel::Data<OutVecDeriv>;
    using InMatrixDeriv = typename In::MatrixDeriv;
    using OutMatrixDeriv = typename Out::MatrixDeriv;
    using InDataMatrixDeriv = sofa::core::objectmodel::Data<InMatrixDeriv>;
    using OutDataMatrixDeriv = sofa::core::objectmodel::Data<OutMatrixDeriv>;
    using Real = typename In::Real;
    using Vec3 = elastic_rod_guidewire::Vec3;

    SOFA_CLASS(ElasticRodRigidStateAdapter, Inherit);

    ElasticRodRigidStateAdapter();
    ~ElasticRodRigidStateAdapter() override = default;

    void init() override;
    bool isMechanical() const override { return false; }
    bool sameTopology() const override { return true; }
    bool isLinear() const override { return false; }

    void apply(const sofa::core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in) override;
    void applyJ(const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in) override;
    void applyJT(const sofa::core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in) override;
    void applyJT(const sofa::core::ConstraintParams* cparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in) override;
    void computeAccFromMapping(const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& accOut, const InDataVecDeriv& vIn, const InDataVecDeriv& accIn) override;

    const sofa::type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override { return nullptr; }
    void disable() override {}

private:
    ElasticRodGuidewireModel* m_rodModel {nullptr};
};

} // namespace elastic_rod_guidewire
