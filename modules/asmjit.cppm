module;

#include <asmjit/x86.h>

export module asmjit;

export namespace asmjit {
using ::asmjit::Arch;
using ::asmjit::BaseEmitter;
using ::asmjit::CodeHolder;
using ::asmjit::Error;
using ::asmjit::ErrorHandler;
using ::asmjit::FuncSignature;
using ::asmjit::Imm;
using ::asmjit::JitRuntime;
using ::asmjit::kErrorOk;
using ::asmjit::kErrorOutOfMemory;

namespace DebugUtils {
using ::asmjit::DebugUtils::errorAsString;
}

namespace x86 {
using ::asmjit::x86::Compiler;
using ::asmjit::x86::Gp;
using ::asmjit::x86::Mem;
} // namespace x86
} // namespace asmjit
