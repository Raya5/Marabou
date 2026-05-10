#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[install_marabou] $*"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_repo_root() {
    # Must be run from the Marabou_IV repository root.
    if [[ ! -f "CMakeLists.txt" || ! -f "setup.py" ]]; then
        die "run this script from the Marabou_IV repository root."
    fi
}

require_gurobi() {
    if [[ -z "${GUROBI_HOME:-}" ]]; then
        cat >&2 <<EOF
Error: GUROBI_HOME is not set.

Please set GUROBI_HOME to your Gurobi installation directory.

Example:
  export GUROBI_HOME=/opt/gurobi1203/armlinux64
EOF
        exit 1
    fi

    [[ -f "$GUROBI_HOME/include/gurobi_c++.h" ]] ||
        die "could not find $GUROBI_HOME/include/gurobi_c++.h"

    [[ -f "$GUROBI_HOME/lib/libgurobi_c++.a" ]] ||
        die "could not find $GUROBI_HOME/lib/libgurobi_c++.a"
}

configure_gurobi_env() {
    # Expose Gurobi to CMake, compiler, linker, and runtime.
    export PATH="$GUROBI_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$GUROBI_HOME/lib:${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="$GUROBI_HOME/lib:${DYLD_LIBRARY_PATH:-}"
    export CPATH="$GUROBI_HOME/include:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="$GUROBI_HOME/include:${CPLUS_INCLUDE_PATH:-}"
    export LIBRARY_PATH="$GUROBI_HOME/lib:${LIBRARY_PATH:-}"
}

detect_jobs() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2
}

print_system_info() {
    log "system info"
    echo "  architecture: $(uname -m)"
    echo "  gcc target:   $(gcc -dumpmachine)"
    echo "  GUROBI_HOME:  $GUROBI_HOME"
    echo "  LIBDIR:       $LIBDIR"

    if [[ -f "$LIBDIR/libprotobuf.so" ]]; then
        echo "  protobuf:     $LIBDIR/libprotobuf.so"
    else
        echo "  warning: protobuf library not found at $LIBDIR/libprotobuf.so"
    fi

    if [[ -f "$LIBDIR/libopenblas.so" ]]; then
        echo "  OpenBLAS:     $LIBDIR/libopenblas.so"
    else
        echo "  warning: OpenBLAS library not found at $LIBDIR/libopenblas.so"
    fi
}

clean_previous_build() {
    log "clean previous build"

    rm -rf build
    rm -f maraboupy/MarabouCore*.so

    rm -rf tools/boost-*/bin.v2
    rm -rf tools/boost-*/installed
    rm -rf tools/boost-*/installed32

    rm -rf tools/protobuf-*/installed
    rm -rf tools/protobuf-*/build

    rm -rf tools/OpenBLAS-*/installed
    rm -rf tools/OpenBLAS-*/build

    rm -rf tools/cadical/build
}

configure_build() {
    log "configure"

    mkdir -p build
    cd build

    cmake .. \
        -DBUILD_PYTHON=ON \
        -DENABLE_GUROBI=ON \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBLAS_LIBRARIES="$LIBDIR/libopenblas.so" \
        -DProtobuf_LIBRARY="$LIBDIR/libprotobuf.so" \
        -DProtobuf_INCLUDE_DIR="/usr/include" \
        -DCMAKE_CXX_STANDARD_LIBRARIES="-L${LIBDIR} -lprotobuf -lopenblas" \
        -DRUN_UNIT_TEST=OFF
}

build_project() {
    log "build"
    cmake --build . -j"$(detect_jobs)"
}

smoke_test_python_import() {
    log "smoke test Python import"

    cd ..
    python -c "import maraboupy.MarabouCore; print('MarabouCore import OK')"
}

main() {
    log "start"

    require_repo_root
    require_gurobi
    configure_gurobi_env

    LIBDIR="/usr/lib/$(gcc -dumpmachine)"
    export LIBDIR

    print_system_info
    clean_previous_build
    configure_build
    build_project
    smoke_test_python_import

    log "done"
}

main "$@"