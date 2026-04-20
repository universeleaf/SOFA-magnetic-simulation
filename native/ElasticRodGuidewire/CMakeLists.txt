cmake_minimum_required(VERSION 3.21)
project(ElasticRodGuidewire VERSION 1.0 LANGUAGES CXX)

set(ELASTICROD_OUTPUT_NAME "ElasticRodGuidewire_hotfix_turnfix" CACHE STRING "Output DLL base name for the ElasticRodGuidewire plugin")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Sofa.Config REQUIRED)
find_package(Sofa.Core REQUIRED)
find_package(Sofa.DefaultType REQUIRED)
find_package(Sofa.Helper REQUIRED)
find_package(Sofa.Type REQUIRED)
find_package(Sofa.LinearAlgebra REQUIRED)
find_package(Sofa.Simulation.Core REQUIRED)

add_library(ElasticRodGuidewire SHARED
    src/ElasticRodCompatCore.cpp
    src/ElasticRodCollisionMapping.cpp
    src/ElasticRodLumpedMass.cpp
    src/ElasticRodGuidewireModel.cpp
    src/ElasticRodRigidStateAdapter.cpp
    src/ExternalMagneticForceField.cpp
    src/initElasticRodGuidewire.cpp
)

target_compile_definitions(ElasticRodGuidewire PRIVATE SOFA_BUILD_ELASTICRODGUIDEWIRE)

target_include_directories(ElasticRodGuidewire
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(ElasticRodGuidewire
    PUBLIC
        Sofa.Core
        Sofa.DefaultType
        Sofa.Helper
        Sofa.Type
        Sofa.LinearAlgebra
        Sofa.Simulation.Core
)

set_target_properties(ElasticRodGuidewire PROPERTIES
    OUTPUT_NAME ${ELASTICROD_OUTPUT_NAME}
    DEBUG_POSTFIX ""
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${cfg} cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/bin)
endforeach()
