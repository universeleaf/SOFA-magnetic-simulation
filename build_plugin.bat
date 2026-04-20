@echo off
setlocal

set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
set "SOFA_ROOT=D:\SOFA_v25.06.00_Win64\SOFA_v25.06.00_Win64"
set "PLUGIN_SRC=%~dp0native\ElasticRodGuidewire"
set "PLUGIN_BUILD=%~dp0build\ElasticRodGuidewire"
set "LOCAL_VCPKG_BOOST=%~dp0tools\vcpkg\installed\x64-windows\include"
set "LOCAL_EIGEN_DIR=%~dp0..\third_party\eigen-3.4.0"
if "%ELASTICROD_OUTPUT_NAME%"=="" set "ELASTICROD_OUTPUT_NAME=%~1"
if "%ELASTICROD_OUTPUT_NAME%"=="" set "ELASTICROD_OUTPUT_NAME=ElasticRodGuidewire_hotfix_turnfix"
set "BOOST_ARG="
set "EIGEN_ARG="

rem This plugin no longer depends on Boost in its current CMakeLists.
rem But SOFA's exported CMake files still call FindBoost, so feed it a local include tree when available.

if not exist "%VSDEVCMD%" goto VSDEV_MISSING
if not exist "%SOFA_ROOT%\lib\cmake\Sofa.Core\Sofa.CoreConfig.cmake" goto SOFA_ROOT_INVALID
if not exist "%PLUGIN_SRC%\CMakeLists.txt" goto PLUGIN_SRC_MISSING
if "%BOOST_INCLUDE_DIR%"=="" if exist "%LOCAL_VCPKG_BOOST%\boost\version.hpp" set "BOOST_INCLUDE_DIR=%LOCAL_VCPKG_BOOST%"
if "%EIGEN3_INCLUDE_DIR%"=="" if exist "%LOCAL_EIGEN_DIR%\signature_of_eigen3_matrix_library" set "EIGEN3_INCLUDE_DIR=%LOCAL_EIGEN_DIR%"
if not "%BOOST_INCLUDE_DIR%"=="" set "BOOST_ARG=-DBoost_INCLUDE_DIR=%BOOST_INCLUDE_DIR%"
if not "%EIGEN3_INCLUDE_DIR%"=="" set "EIGEN_ARG=-DEIGEN3_INCLUDE_DIR=%EIGEN3_INCLUDE_DIR%"

call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1

cmake --fresh ^
    -S "%PLUGIN_SRC%" ^
    -B "%PLUGIN_BUILD%" ^
    -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH=%SOFA_ROOT% ^
    -DELASTICROD_OUTPUT_NAME=%ELASTICROD_OUTPUT_NAME% ^
    %BOOST_ARG% ^
    %EIGEN_ARG%
if errorlevel 1 exit /b 1

cmake --build "%PLUGIN_BUILD%" --config Release
if errorlevel 1 exit /b 1

echo [INFO] Plugin build finished for %ELASTICROD_OUTPUT_NAME%. DLL should be under "%~dp0build\ElasticRodGuidewire"
exit /b 0

:VSDEV_MISSING
echo [ERROR] VsDevCmd.bat not found: %VSDEVCMD%
exit /b 1

:SOFA_ROOT_INVALID
echo [ERROR] SOFA_ROOT invalid: %SOFA_ROOT%
exit /b 1

:PLUGIN_SRC_MISSING
echo [ERROR] Plugin source not found: %PLUGIN_SRC%
exit /b 1
