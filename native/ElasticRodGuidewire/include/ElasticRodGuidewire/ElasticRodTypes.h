#pragma once

#include <ElasticRodGuidewire/config.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Quat.h>
#include <sofa/type/Vec.h>

namespace elastic_rod_guidewire
{

using CarrierTypes = sofa::defaulttype::Vec6dTypes;
using CarrierCoord = CarrierTypes::Coord;
using CarrierDeriv = CarrierTypes::Deriv;
using CarrierVecCoord = CarrierTypes::VecCoord;
using CarrierVecDeriv = CarrierTypes::VecDeriv;
using Vec3 = sofa::type::Vec<3, double>;
using Vec2 = sofa::type::Vec<2, double>;
using Quat = sofa::type::Quat<double>;

inline Vec3 coordCenter(const CarrierCoord& c)
{
    return Vec3(c[0], c[1], c[2]);
}

inline void setCoordCenter(CarrierCoord& c, const Vec3& v)
{
    c[0] = v[0];
    c[1] = v[1];
    c[2] = v[2];
}

inline double coordTheta(const CarrierCoord& c)
{
    return c[3];
}

inline void setCoordTheta(CarrierCoord& c, double theta)
{
    c[3] = theta;
    c[4] = 0.0;
    c[5] = 0.0;
}

inline Vec3 derivCenter(const CarrierDeriv& d)
{
    return Vec3(d[0], d[1], d[2]);
}

inline void setDerivCenter(CarrierDeriv& d, const Vec3& v)
{
    d[0] = v[0];
    d[1] = v[1];
    d[2] = v[2];
}

inline double derivTheta(const CarrierDeriv& d)
{
    return d[3];
}

inline void setDerivTheta(CarrierDeriv& d, double thetaRate)
{
    d[3] = thetaRate;
    d[4] = 0.0;
    d[5] = 0.0;
}

template<class TVec6>
inline void clearUnused(TVec6& c)
{
    c[4] = 0.0;
    c[5] = 0.0;
}

} // namespace elastic_rod_guidewire
