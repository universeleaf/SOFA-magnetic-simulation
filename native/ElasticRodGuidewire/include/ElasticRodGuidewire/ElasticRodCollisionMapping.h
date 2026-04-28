#pragma once

#include <ElasticRodGuidewire/ElasticRodTypes.h>
#include <ElasticRodGuidewire/config.h>

#include <sofa/core/Mapping.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/type/vector.h>

#include <vector>

namespace elastic_rod_guidewire
{

class SOFA_ELASTICRODGUIDEWIRE_API ElasticRodCollisionMapping
    : public sofa::core::Mapping<CarrierTypes, sofa::defaulttype::Vec3dTypes>
{
public:
    using In = CarrierTypes;
    using Out = sofa::defaulttype::Vec3dTypes;
    using Inherit = sofa::core::Mapping<In, Out>;
    using Real = typename In::Real;
    using InVecCoord = typename In::VecCoord;
    using InVecDeriv = typename In::VecDeriv;
    using InDeriv = typename In::Deriv;
    using InMatrixDeriv = typename In::MatrixDeriv;
    using OutVecCoord = typename Out::VecCoord;
    using OutVecDeriv = typename Out::VecDeriv;
    using OutDeriv = typename Out::Deriv;
    using OutMatrixDeriv = typename Out::MatrixDeriv;
    using VecUInt = sofa::type::vector<unsigned int>;
    using InDataVecCoord = sofa::core::objectmodel::Data<InVecCoord>;
    using InDataVecDeriv = sofa::core::objectmodel::Data<InVecDeriv>;
    using InDataMatrixDeriv = sofa::core::objectmodel::Data<InMatrixDeriv>;
    using OutDataVecCoord = sofa::core::objectmodel::Data<OutVecCoord>;
    using OutDataVecDeriv = sofa::core::objectmodel::Data<OutVecDeriv>;
    using OutDataMatrixDeriv = sofa::core::objectmodel::Data<OutMatrixDeriv>;

    SOFA_CLASS(ElasticRodCollisionMapping, Inherit);

    ElasticRodCollisionMapping();
    ~ElasticRodCollisionMapping() override = default;

    void init() override;
    bool isMechanical() const override { return true; }
    bool sameTopology() const override { return true; }
    bool isLinear() const override { return true; }

    sofa::core::objectmodel::Data<VecUInt> d_selectedIndices;

    void apply(const sofa::core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in) override;
    void applyJ(const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in) override;
    void applyJT(const sofa::core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in) override;
    void applyJT(const sofa::core::ConstraintParams* cparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in) override;
    void computeAccFromMapping(const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& accOut, const InDataVecDeriv& vIn, const InDataVecDeriv& accIn) override;
    const sofa::linearalgebra::BaseMatrix* getJ() override;
    const sofa::type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

private:
    static constexpr Real kMToMm = static_cast<Real>(1.0e3);
    static constexpr Real kSceneForceToNewton = static_cast<Real>(1.0e-3);

    void refreshSelectedIndices(std::size_t parentNodeCount);

    sofa::linearalgebra::EigenSparseMatrix<In, Out> m_jacobian;
    sofa::type::vector<sofa::linearalgebra::BaseMatrix*> m_jacobians;
    std::vector<std::size_t> m_selectedIndices;
};

} // namespace elastic_rod_guidewire
