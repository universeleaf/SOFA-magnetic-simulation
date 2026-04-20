#pragma once

#include <sofa/config.h>

#ifdef SOFA_BUILD_ELASTICRODGUIDEWIRE
#  define SOFA_TARGET ElasticRodGuidewire
#  define SOFA_ELASTICRODGUIDEWIRE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_ELASTICRODGUIDEWIRE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace elastic_rod_guidewire
{
constexpr const char* MODULE_NAME = "ElasticRodGuidewire";
constexpr const char* MODULE_VERSION = "1.0";
} // namespace elastic_rod_guidewire
