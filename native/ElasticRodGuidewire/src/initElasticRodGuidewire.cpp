#include <ElasticRodGuidewire/config.h>
#include <ElasticRodGuidewire/ExternalMagneticForceField.h>

#include <sofa/config/sharedlibrary_defines.h>

namespace
{
bool g_initialized = false;
}

extern "C"
{
SOFA_ELASTICRODGUIDEWIRE_API void initExternalModule()
{
    if (g_initialized)
        return;
    g_initialized = true;
}

SOFA_ELASTICRODGUIDEWIRE_API const char* getModuleName()
{
    return elastic_rod_guidewire::MODULE_NAME;
}

SOFA_ELASTICRODGUIDEWIRE_API const char* getModuleVersion()
{
    return elastic_rod_guidewire::MODULE_VERSION;
}

SOFA_ELASTICRODGUIDEWIRE_API const char* getModuleLicense()
{
    return "LGPL";
}

SOFA_ELASTICRODGUIDEWIRE_API const char* getModuleDescription()
{
    return "Native guidewire backend and magnetic coupling components aligned with the legacy elasticRod / externalMagneticForce semantics.";
}
}
