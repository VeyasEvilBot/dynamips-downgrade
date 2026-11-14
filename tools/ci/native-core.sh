#!/usr/bin/env bash
# Native rewrite checks only; public Go/legacy checks run in their own workflow.
set -euo pipefail

usage() {
    echo 'Usage: tools/ci/native-core.sh check debug|release | sanitize | clangd | tidy'
}
command=${1:-}
case "$command" in
    check)
        mode=${2:-debug}
        case "$mode" in debug|release) ;; *) usage >&2; exit 2 ;; esac
        ;;
    sanitize|clangd|tidy) mode=debug ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
jobs=${DYNAMIPS_BUILD_JOBS:-4}
mkdir -p build/ci-logs
exec > >(tee "build/ci-logs/${command}-${mode}.log") 2>&1

cxflags=${DYNAMIPS_CI_CXFLAGS:-}
ldflags=${DYNAMIPS_CI_LDFLAGS:-}
if [[ "$command" == sanitize ]]; then
    cxflags+=" -fsanitize=address,undefined -fno-omit-frame-pointer"
    ldflags+=" -fsanitize=address,undefined"
    export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1}"
    export UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}"
fi
xmake f -m "$mode" --system_packages="${DYNAMIPS_CI_SYSTEM_PACKAGES:-y}" --dynamips_arch=amd64 \
    --cxflags="$cxflags" --ldflags="$ldflags" -y

# Build the module consumers together. A legacy test success does not validate these.
xmake build -r -j "$jobs" -y \
    jit-test interpreter-test mmu-test cpu-fault-test physical-test exceptions-test \
    memory-test ownership-test count-batch-test count-dispatch-test device-test host-test console-test rommon-test bootflash-test \
    interrupt-test pcmcia-test temperature-test machine-test c7200-boot \
    dynamips-downgrade terminal-console-test jit-playground

if [[ "$command" == clangd || "$command" == tidy ]]; then
    stage=$(mktemp -d "$PWD/build/ci-compiledb.XXXXXX")
    trap 'rm -rf "$stage"' EXIT
    xmake project -k compile_commands --lsp=clangd "$stage"
    mv "$stage/compile_commands.json" compile_commands.json
    # Avoid compiledb's legacy binding/benchmark dependencies: all consumers above
    # have already been built with this job's compiler and configuration.
    if [[ "$command" == clangd ]]; then
        mise run --skip-deps lint:clangd
    else
        mise run --skip-deps lint:cpp:jit
    fi
else
    mise run test:core ::: test:machine ::: test:console ::: test:host ::: test:cli
    xmake run -y boot-console-test
fi
