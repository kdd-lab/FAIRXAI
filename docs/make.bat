@ECHO OFF
REM ================================================
REM  Sphinx documentation build script for Windows
REM  Cross-compatible with the Python script setup
REM ================================================

pushd %~dp0

REM If SPHINXBUILD is not defined, use default
if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)

REM Match the folder structure used in generate_docs.py
set SOURCEDIR=source
set BUILDDIR=build

REM Check if sphinx-build is available
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo The 'sphinx-build' command was not found.
    echo Make sure you have Sphinx installed and available in PATH,
    echo or set the SPHINXBUILD environment variable to point to it.
    echo.
    echo To install Sphinx, run:
    echo     pip install sphinx
    echo.
    exit /b 1
)

REM If no argument is given, print help
if "%1" == "" goto help

REM Run sphinx-build with the requested target
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
