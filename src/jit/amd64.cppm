module;

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <dynamips/macros.h>

export module jit.amd64;

import asmjit;
import jit.mips64;
import jit.ir;
import jit.memory;
import jit.tlb;

export namespace jit::amd64 {

enum class error_code {
  invalid_ir,
  out_of_memory,
  code_generation,
  invalid_state,
  unavailable,
  memory_required,
  cop0_required,
  cop1_required,
  hilo_required,
  memory_fault,
  invalid_tlb_state,
  unsupported_performance_counter,
};

struct error {
  error_code code;
  std::size_t node_index = 0;
  std::uint32_t native_code = 0;
};

struct execution_error {
  error_code code;
  std::uint64_t pc;
  memory::fault access{};
  std::optional<std::uint64_t> branch_pc{};
  std::uint32_t completed_instructions = 0;
  std::uint32_t instructions = 0;
  std::uint32_t blocks = 1;
};

[[nodiscard]] fn message(error value) noexcept -> std::string_view {
  switch (value.code) {
  case error_code::invalid_ir:
    return "invalid IR";
  case error_code::out_of_memory:
    return "compilation scratch or native allocation exhausted";
  case error_code::code_generation:
    return value.native_code
               ? asmjit::DebugUtils::errorAsString(value.native_code)
               : "code generation failed";
  case error_code::invalid_state:
    return "state has too few slots";
  case error_code::unavailable:
    return "native AMD64 backend is unavailable";
  case error_code::hilo_required:
    return "block requires HI/LO state";
  case error_code::cop1_required:
    return "block requires COP1 state";
  case error_code::cop0_required:
    return "block requires CP0 state";
  case error_code::memory_required:
    return "block requires guest memory";
  case error_code::invalid_tlb_state:
    return "invalid TLB register state";
  case error_code::memory_fault:
    return "guest memory access failed";
  case error_code::unsupported_performance_counter:
    return "enabled RM7000 performance counters are unsupported";
  }
  return "native compilation failed";
}

enum class exit_reason { completed, guest_exception };

struct execution_result {
  std::uint64_t next_pc;
  std::uint32_t instructions;
  exit_reason reason;
  std::uint32_t blocks = 1;
  std::uint32_t accounted_instructions = 0;
};

class compiler;

struct trace_region {
  std::uint64_t pc;
  std::optional<std::uint64_t> branch_pc;
};

class compiled_block {
public:
  compiled_block(const compiled_block &) = delete;
  fn operator=(const compiled_block &) -> compiled_block & = delete;

  compiled_block(compiled_block &&other) noexcept
      : runtime_(std::move(other.runtime_)),
        entry_(std::exchange(other.entry_, nullptr)), slots_(other.slots_),
        uses_memory_(other.uses_memory_), uses_cop0_(other.uses_cop0_),
        uses_hilo_(other.uses_hilo_), uses_cop1_(other.uses_cop1_),
        branch_pc_(other.branch_pc_), start_pc_(other.start_pc_),
        instructions_(other.instructions_),
        guards_writes_(other.guards_writes_), simple_(other.simple_),
        regions_(std::move(other.regions_)) {}

  fn operator=(compiled_block &&other) noexcept -> compiled_block & {
    if (this != &other) {
      reset();
      runtime_ = std::move(other.runtime_);
      entry_ = std::exchange(other.entry_, nullptr);
      slots_ = other.slots_;
      uses_memory_ = other.uses_memory_;
      uses_cop0_ = other.uses_cop0_;
      uses_hilo_ = other.uses_hilo_;
      uses_cop1_ = other.uses_cop1_;
      branch_pc_ = other.branch_pc_;
      start_pc_ = other.start_pc_;
      instructions_ = other.instructions_;
      guards_writes_ = other.guards_writes_;
      simple_ = other.simple_;
      regions_ = std::move(other.regions_);
    }
    return *this;
  }

  ~compiled_block() { reset(); }

  [[nodiscard]] fn run(std::span<std::uint64_t> state) const noexcept
      -> std::expected<std::uint64_t, error> {
    if (!entry_)
      return std::unexpected(error{error_code::unavailable});
    if (state.size() < slots_)
      return std::unexpected(error{error_code::invalid_state});
    if (uses_cop1_)
      return std::unexpected(error{error_code::cop1_required});
    if (uses_hilo_)
      return std::unexpected(error{error_code::hilo_required});
    if (uses_cop0_)
      return std::unexpected(error{error_code::cop0_required});
    if (uses_memory_)
      return std::unexpected(error{error_code::memory_required});
    return entry_(state.data(), nullptr);
  }

  [[nodiscard, gnu::always_inline]] inline fn
  execute(std::span<std::uint64_t> state, memory::address_space &memory,
          mips64::cop0_state *cop0 = nullptr,
          mips64::hilo_state *hilo = nullptr,
          mips64::cop1_state *cop1 = nullptr,
          std::span<const std::span<const std::byte>> code = {},
          std::uint64_t budget = UINT64_MAX, std::uint32_t blocks = 1,
          std::uint64_t cycle_limit = UINT64_MAX) const noexcept
      -> std::expected<execution_result, execution_error> {
    if (!entry_)
      return std::unexpected(execution_error{error_code::unavailable, 0});
    if (state.size() < slots_)
      return std::unexpected(execution_error{error_code::invalid_state, 0});
    if (uses_cop0_ && !cop0)
      return std::unexpected(execution_error{error_code::cop0_required, 0});
    if (uses_hilo_ && !hilo)
      return std::unexpected(execution_error{error_code::hilo_required, 0});
    if (uses_cop1_ && !cop1)
      return std::unexpected(execution_error{error_code::cop1_required, 0});
    if (guards_writes_ && code.empty())
      return std::unexpected(execution_error{error_code::invalid_state, 0});
    if (!budget || !blocks || (regions_.empty() && budget < instructions_))
      return execution_result{start_pc_, 0, exit_reason::completed, 0, 0};
    if (simple_)
      return execution_result{entry_(state.data(), nullptr), instructions_,
                              exit_reason::completed, 1, 0};
    execution_context context{&memory, cop0, hilo, cop1};
    context.writes.ranges = code;
    context.read_cache = memory.read_cache();
    context.budget = budget;
    context.block_limit = std::min<std::uint32_t>(64, blocks);
    context.cycle_limit = cycle_limit;
    if (guards_writes_ && cop0) {
      context.writes.context = &context;
      context.writes.before_external = &account_before_external;
    }
    const let monitor =
        memory.monitor_writes(guards_writes_ ? &context.writes : nullptr);
    const let pc = entry_(state.data(), &context);
    const trace_region region = regions_.empty()
                                    ? trace_region{start_pc_, branch_pc_}
                                    : regions_[context.region];
    if (context.failed)
      return std::unexpected(execution_error{
          context.issue, pc, context.fault,
          region.branch_pc && pc == *region.branch_pc + 4 ? region.branch_pc
                                                          : std::nullopt,
          context.instruction_base +
              static_cast<std::uint32_t>((pc - region.pc) / 4) -
              context.accounted,
          context.instruction_base +
              static_cast<std::uint32_t>((pc - region.pc) / 4),
          regions_.empty() ? 1u : context.executed_blocks});
    return execution_result{
        pc,
        context.trap_pc ? context.instruction_base +
                              static_cast<std::uint32_t>(
                                  (*context.trap_pc - region.pc) / 4 + 1)
        : (!regions_.empty() || context.executed_slots)
            ? context.executed_slots
            : instructions_ - (context.annulled_slot ? 1u : 0u),
        context.trap_pc ? exit_reason::guest_exception : exit_reason::completed,
        regions_.empty() ? 1u : context.executed_blocks, context.accounted};
  }

  [[nodiscard]] fn
  run(std::span<std::uint64_t> state, memory::address_space &memory,
      mips64::cop0_state *cop0 = nullptr, mips64::hilo_state *hilo = nullptr,
      mips64::cop1_state *cop1 = nullptr) const noexcept
      -> std::expected<std::uint64_t, execution_error> {
    const let result = execute(state, memory, cop0, hilo, cop1);
    if (!result)
      return std::unexpected(result.error());
    return result->next_pc;
  }

private:
  friend class compiler;

  struct execution_context {
    memory::address_space *memory;
    mips64::cop0_state *cop0;
    mips64::hilo_state *hilo;
    mips64::cop1_state *cop1;
    memory::write_monitor writes{};
    const memory::cache_entry *read_cache = nullptr;
    std::uint32_t executed_slots = 0;
    std::uint32_t instruction_base = 0;
    std::uint32_t effect_prefix = 0;
    std::uint32_t accounted = 0;
    std::uint32_t executed_blocks = 0;
    std::uint32_t block_limit = 1;
    std::uint32_t region = 0;
    std::uint64_t budget = UINT64_MAX;
    std::uint64_t cycle_limit = UINT64_MAX;
    std::optional<std::uint64_t> trap_pc{};
    memory::fault fault{};
    bool annulled_slot = false;
    bool failed = false;
    error_code issue = error_code::memory_fault;
  };

private:
  using entry_point = std::uint64_t (*)(std::uint64_t *, execution_context *);

  static fn account_before_external(void *opaque) noexcept -> void {
    let &context = *static_cast<execution_context *>(opaque);
    context.cop0->advance_cycles(context.effect_prefix - context.accounted);
    context.accounted = context.effect_prefix;
  }

  template <std::unsigned_integral T, bool sign_extend = true,
            bool Linked = false>
  static fn load(execution_context *context, std::uint64_t address) noexcept
      -> std::uint64_t {
    let result = [&] {
      if constexpr (Linked)
        return context->memory->load_linked<T>({address});
      else
        return context->memory->load<T>({address});
    }();
    if (!result) {
      context->fault = result.error();
      context->failed = true;
      return 0;
    }
    if constexpr (sizeof(T) == 1 && sign_extend)
      return static_cast<std::uint64_t>(
          std::int64_t{std::bit_cast<std::int8_t>(*result)});
    if constexpr (sizeof(T) == 2 && sign_extend)
      return static_cast<std::uint64_t>(
          std::int64_t{std::bit_cast<std::int16_t>(*result)});
    if constexpr (sizeof(T) == 4 && sign_extend)
      return static_cast<std::uint64_t>(
          std::int64_t{std::bit_cast<std::int32_t>(*result)});
    return *result;
  }

  template <std::unsigned_integral T, bool Right>
  static fn load_merge(execution_context *context, std::uint64_t address,
                        std::uint64_t previous) noexcept -> std::uint64_t {
    const T loaded = static_cast<T>(
        load<T>(context, address & ~std::uint64_t{sizeof(T) - 1}));
    if (context->failed) {
      context->fault.address.value = address;
      return 0;
    }
    const let byte = static_cast<unsigned>(address & (sizeof(T) - 1));
    const let shift = (Right ? sizeof(T) - 1 - byte : byte) * 8;
    constexpr T mask = std::numeric_limits<T>::max();
    const T result =
        Right
            ? (static_cast<T>(previous) & ~(mask >> shift)) | (loaded >> shift)
            : (static_cast<T>(previous) & ~(mask << shift)) | (loaded << shift);
    if constexpr (sizeof(T) == 4)
      return static_cast<std::uint64_t>(
          std::int64_t{std::bit_cast<std::int32_t>(result)});
    return result;
  }

  template <std::unsigned_integral T>
  static fn store(execution_context *context, std::uint64_t address,
                   std::uint64_t value) noexcept -> void {
    let result = context->memory->store<T>({address}, static_cast<T>(value));
    if (!result) {
      context->fault = result.error();
      context->failed = true;
    }
  }

  template <std::unsigned_integral T, bool Right>
  static fn store_merge(execution_context *context, std::uint64_t address,
                         std::uint64_t value) noexcept -> void {
    const let offset = static_cast<unsigned>(address & (sizeof(T) - 1));
    const let begin = Right ? 0u : offset;
    const let count =
        Right ? offset + 1 : static_cast<unsigned>(sizeof(T)) - offset;
    std::array<std::byte, sizeof(T)> bytes;
    for (unsigned i = 0; i < count; i++) {
      const let shift =
          (Right ? count - 1 - i : static_cast<unsigned>(sizeof(T)) - 1 - i) *
          8;
      bytes[i] = static_cast<std::byte>((value >> shift) & 255);
    }
    const let stored = context->memory->store_bytes(
        {(address & ~std::uint64_t{sizeof(T) - 1}) + begin},
        std::span{bytes}.first(count));
    if (!stored) {
      context->fault = stored.error();
      context->fault.address.value = address;
      context->fault.size = static_cast<memory::access_size>(sizeof(T));
      context->failed = true;
    }
  }

  static fn store_conditional32(execution_context *context,
                                 std::uint64_t address,
                                 std::uint64_t value) noexcept
      -> std::uint64_t {
    const let result = context->memory->store_conditional<std::uint32_t>(
        {address}, static_cast<std::uint32_t>(value));
    if (!result) {
      context->fault = result.error();
      context->failed = true;
      return 0;
    }
    return *result ? 1 : 0;
  }

  static fn tlb_operation(execution_context *context,
                           std::uint32_t operation) noexcept -> void {
    let &cp0 = *context->cop0;
    const let index = static_cast<std::size_t>(cp0.registers[0] & 63);
    if (operation == 0) {
      const let value = cp0.translations.read(index);
      if (value) {
        cp0.registers[5] = value->mask;
        cp0.registers[10] = value->high;
        cp0.registers[2] = value->low[0];
        cp0.registers[3] = value->low[1];
      }
      context->memory->invalidate();
    } else if (operation == 1) {
      const let written =
          cp0.translations.write(index, {cp0.registers[5],
                                         cp0.registers[10],
                                         {cp0.registers[2], cp0.registers[3]}});
      if (!written) {
        context->failed = true;
        context->issue = error_code::invalid_tlb_state;
        return;
      }
      context->memory->invalidate();
    } else {
      const let found = cp0.translations.probe(
          {cp0.registers[10]}, static_cast<std::uint8_t>(cp0.registers[10]));
      cp0.registers[0] = found ? *found : 0x80000000ULL;
    }
  }

  static fn cache_index(execution_context *context,
                         std::uint64_t address) noexcept -> void {
    const let result = context->memory->cache_index({address});
    if (!result) {
      context->fault = result.error();
      context->failed = true;
    }
  }

  static fn multiply32(execution_context *context, std::uint64_t left,
                        std::uint64_t right) noexcept -> void {
    context->hilo->multiply32(left, right);
  }

  static fn multiply_unsigned32(execution_context *context, std::uint64_t left,
                                 std::uint64_t right) noexcept -> void {
    context->hilo->multiply_unsigned32(left, right);
  }

  static fn divide_unsigned32(execution_context *context, std::uint64_t left,
                               std::uint64_t right) noexcept -> void {
    context->hilo->divide_unsigned32(left, right);
  }

  static fn divide_signed32(execution_context *context, std::uint64_t left,
                             std::uint64_t right) noexcept -> void {
    context->hilo->divide_signed32(left, right);
  }

  static fn exception_return(execution_context *context) noexcept
      -> std::uint64_t {
    context->memory->invalidate();
    return context->cop0->exception_return().value;
  }

  static fn overflow(execution_context *context, std::uint64_t pc,
                      std::uint32_t delay_slot) noexcept -> std::uint64_t {
    context->memory->clear_reservation();
    if (!(context->cop0->registers[12] & 2))
      context->memory->invalidate();
    context->trap_pc = pc + (delay_slot ? 4 : 0);
    return context->cop0->exception(12, {pc}, delay_slot != 0).value;
  }

  static fn software_exception(execution_context *context, std::uint64_t pc,
                                std::uint32_t delay_slot,
                                std::uint32_t code) noexcept -> std::uint64_t {
    context->memory->clear_reservation();
    if (!(context->cop0->registers[12] & 2))
      context->memory->invalidate();
    context->trap_pc = pc + (delay_slot ? 4 : 0);
    return context->cop0->exception(code, {pc}, delay_slot != 0).value;
  }

  static fn write_cop0(execution_context *context, std::uint32_t slot,
                        std::uint64_t value, std::uint32_t wide) noexcept
      -> void {
    account_before_external(context);
    if (slot == 22 &&
        !mips64::cop0_state::performance_control_supported(value)) {
      context->failed = true;
      context->issue = error_code::unsupported_performance_counter;
      return;
    }
    const let old_status = context->cop0->registers[12];
    const let old_asid = context->cop0->registers[10] & 255;
    if (wide && (slot == 4 || slot == 20)) {
      const std::uint64_t preserve = slot == 4 ? 0x7fffffULL : 0x1ffffffffULL;
      context->cop0->registers[slot] =
          (context->cop0->registers[slot] & preserve) | (value & ~preserve);
    } else if (wide && (slot == 10 || slot == 14 || slot == 30))
      context->cop0->registers[slot] =
          slot == 10 ? value & 0xc00000ffffffe0ffULL : value;
    else
      context->cop0->write(static_cast<std::uint16_t>(slot),
                           static_cast<std::uint32_t>(value));
    if (context->memory &&
        ((slot == 10 && old_asid != (context->cop0->registers[10] & 255)) ||
         (slot == 12 && ((old_status ^ context->cop0->registers[12]) & 0xfe))))
      context->memory->invalidate();
  }

  compiled_block(std::shared_ptr<asmjit::JitRuntime> runtime, entry_point entry,
                 std::uint16_t slots, bool uses_memory, bool uses_cop0,
                 bool uses_hilo, bool uses_cop1,
                 std::optional<std::uint64_t> branch_pc, std::uint64_t start_pc,
                 std::uint32_t instructions, bool guards_writes,
                 std::vector<trace_region> regions, bool simple) noexcept
      : runtime_(std::move(runtime)), entry_(entry), slots_(slots),
        uses_memory_(uses_memory), uses_cop0_(uses_cop0), uses_hilo_(uses_hilo),
        uses_cop1_(uses_cop1), branch_pc_(branch_pc), start_pc_(start_pc),
        instructions_(instructions), guards_writes_(guards_writes),
        simple_(simple), regions_(std::move(regions)) {}

  std::shared_ptr<asmjit::JitRuntime> runtime_;
  entry_point entry_;
  std::uint16_t slots_;
  bool uses_memory_;
  bool uses_cop0_;
  bool uses_hilo_;
  bool uses_cop1_;
  std::optional<std::uint64_t> branch_pc_;
  std::uint64_t start_pc_;
  std::uint32_t instructions_;
  bool guards_writes_;
  bool simple_;
  std::vector<trace_region> regions_;

  fn reset() noexcept -> void {
    // Only this owner can release the installed entry point.
    if (entry_)
      static_cast<void>(runtime_->release(std::exchange(entry_, nullptr)));
  }
};

class compiler {
public:
  [[nodiscard]] static fn create() noexcept -> std::expected<compiler, error> {
    try {
      return compiler{std::make_shared<asmjit::JitRuntime>()};
    } catch (const std::bad_alloc &) {
      return std::unexpected(error{error_code::out_of_memory});
    }
  }

  [[nodiscard]] fn compile(const ir::block &block,
                            std::pmr::memory_resource &scratch,
                            bool guard_writes = false) const noexcept
      -> std::expected<compiled_block, error> {
    return compile_trace(std::span{&block, 1}, scratch, guard_writes, false);
  }

  [[nodiscard]] fn compile_trace(std::span<const ir::block> blocks,
                                  std::pmr::memory_resource &scratch,
                                  bool guard_writes = true,
                                  bool fuse = true) const noexcept
      -> std::expected<compiled_block, error> {
    if (!runtime_ || runtime_->environment().arch() != asmjit::Arch::kX64)
      return std::unexpected(error{error_code::unavailable});
    if (blocks.empty() || blocks.size() > 64 || (!fuse && blocks.size() != 1))
      return std::unexpected(error{error_code::invalid_ir});
    std::uint32_t trace_words = 0;
    for (const let &block : blocks) {
      if (let valid = block.verify(); !valid)
        return std::unexpected(
            error{error_code::invalid_ir, valid.error().node_index});
      if (fuse && (block.info().instruction_count > 65 - trace_words ||
                   block.info().start_pc !=
                       blocks.front().info().start_pc + trace_words * 4))
        return std::unexpected(error{error_code::invalid_ir});
      trace_words += block.info().instruction_count;
    }

    const let failure = [](asmjit::Error code, std::size_t index) {
      return std::unexpected(error{code == asmjit::kErrorOutOfMemory
                                       ? error_code::out_of_memory
                                       : error_code::code_generation,
                                   index, code});
    };
    try {
      error_handler errors;
      asmjit::CodeHolder code;
      code.setErrorHandler(&errors);
      if (const let status =
              code.init(runtime_->environment(), runtime_->cpuFeatures()))
        return failure(status, 0);
      asmjit::x86::Compiler output{&code};
      const let function = output.addFunc(
          asmjit::FuncSignature::build<std::uint64_t, std::uint64_t *,
                                       compiled_block::execution_context *>());
      if (!function)
        return failure(errors.code, 0);
      const let state = output.newUIntPtr();
      if (errors.code)
        return failure(errors.code, 0);
      function->setArg(0, state);
      const let context = output.newUIntPtr();
      if (errors.code)
        return failure(errors.code, 0);
      function->setArg(1, context);
      bool uses_cop1 = false;
      bool uses_memory = fuse;
      bool uses_cop0 = false;
      bool uses_hilo = false;

      std::pmr::vector<decltype(output.newLabel())> labels{&scratch};
      std::vector<trace_region> regions;
      std::uint16_t slots = 0;
      std::uint32_t source_instructions = 0;
      if (fuse)
        regions.reserve(blocks.size());
      for (const let &block : blocks) {
        labels.push_back(output.newLabel());
        slots = std::max(slots, block.info().state_slots);
        source_instructions += block.info().instruction_count;
        if (fuse)
          regions.push_back({block.info().start_pc, block.branch_pc()});
      }
      const let finish = [&](asmjit::x86::Gp pc) -> asmjit::Error {
        return output.ret(pc);
      };
      for (std::size_t region_index = 0; region_index < blocks.size();
           region_index++) {
        const let &block = blocks[region_index];
        std::pmr::vector<asmjit::x86::Gp> values{&scratch};
        values.resize(block.nodes().size());
        std::pmr::vector<std::uint8_t> keep_store(block.nodes().size(), 0,
                                                  &scratch);
        std::pmr::vector<std::uint8_t> observed(block.info().state_slots, 1,
                                                &scratch);
        const let *conditional = block.branch();
        const bool likely = conditional && conditional->annul_not_taken;
        std::pmr::vector<std::uint8_t> guarded_sources(
            block.info().instruction_count, 0, &scratch);
        if (guard_writes)
          for (const let &node : block.nodes())
            if ((node.op >= ir::opcode::load8_signed &&
                 node.op <= ir::opcode::store64) ||
                node.op == ir::opcode::tlb_operation ||
                node.op == ir::opcode::cache_index)
              guarded_sources[node.source_index] = 1;
        // An exclusive merge pair may read once after proving both accesses
        // use one cached page. Every fallback retains the first partial write.
        std::pmr::vector<std::uint32_t> pair_first(block.nodes().size(),
                                                   UINT32_MAX, &scratch);
        std::pmr::vector<std::uint32_t> pair_second(block.nodes().size(),
                                                    UINT32_MAX, &scratch);
        std::pmr::vector<asmjit::x86::Gp> pair_flags(block.nodes().size(),
                                                     &scratch);
        if (fuse) {
          const let resolve_alias = [&](ir::value_id id) {
            while (block.nodes()[id.index].op == ir::opcode::alias)
              id = block.nodes()[id.index].data.inputs.left;
            return id;
          };
          const let address_parts = [&](ir::value_id id) {
            id = resolve_alias(id);
            const let &node = block.nodes()[id.index];
            if (node.op == ir::opcode::add_wrap) {
              const let left = resolve_alias(node.data.inputs.left);
              const let right = resolve_alias(node.data.inputs.right);
              if (block.nodes()[left.index].op == ir::opcode::constant)
                return std::pair{right.index,
                                 block.nodes()[left.index].data.immediate};
              if (block.nodes()[right.index].op == ir::opcode::constant)
                return std::pair{left.index,
                                 block.nodes()[right.index].data.immediate};
            }
            return std::pair{id.index, std::uint64_t{0}};
          };
          const let references = [&](const ir::node &node, std::uint32_t id) {
            switch (node.op) {
            case ir::opcode::constant:
            case ir::opcode::read_state:
            case ir::opcode::read_hilo:
            case ir::opcode::read_cop0:
            case ir::opcode::read_cop0_64:
            case ir::opcode::read_cop1_64:
            case ir::opcode::syscall:
            case ir::opcode::breakpoint:
            case ir::opcode::exception_return:
            case ir::opcode::memory_barrier:
            case ir::opcode::tlb_operation:
              return false;
            case ir::opcode::shift_left64:
            case ir::opcode::shift_right64:
            case ir::opcode::shift_right_arithmetic64:
            case ir::opcode::shift_left32:
            case ir::opcode::shift_right32:
            case ir::opcode::shift_right_arithmetic32:
              return node.data.shift.value.index == id;
            default:
              return node.data.inputs.left.index == id ||
                     node.data.inputs.right.index == id;
            }
          };
          for (std::size_t first = 0; first < block.nodes().size(); first++) {
            const let &left = block.nodes()[first];
            if (left.op != ir::opcode::load32_left &&
                left.op != ir::opcode::load64_left)
              continue;
            const std::uint32_t bytes =
                left.op == ir::opcode::load32_left ? 4u : 8u;
            if (block.branch() &&
                block.branch()->source_index <= left.source_index + 1)
              continue;
            const let base = address_parts(left.data.inputs.left);
            std::optional<std::size_t> first_store;
            for (std::size_t second = first + 1; second < block.nodes().size();
                 second++) {
              const let &right = block.nodes()[second];
              if (right.source_index > left.source_index + 1)
                break;
              if (right.op == ir::opcode::write_state &&
                  right.source_index == left.source_index &&
                  right.data.inputs.left.index == first) {
                if (first_store)
                  break;
                first_store = second;
                continue;
              }
              if (right.op == (bytes == 4 ? ir::opcode::load32_right
                                          : ir::opcode::load64_right)) {
                if (!first_store ||
                    right.source_index != left.source_index + 1 ||
                    resolve_alias(right.data.inputs.right).index != first)
                  break;
                const let other = address_parts(right.data.inputs.left);
                if (base.first != other.first ||
                    other.second - base.second != bytes - 1)
                  break;
                std::optional<std::size_t> second_store;
                for (std::size_t j = second + 1;
                     j < block.nodes().size() &&
                     block.nodes()[j].source_index == right.source_index;
                     j++) {
                  const let &store = block.nodes()[j];
                  if (store.op == ir::opcode::write_state &&
                      store.data.inputs.left.index == second &&
                      store.slot.index ==
                          block.nodes()[*first_store].slot.index)
                    second_store = j;
                }
                if (!second_store || *second_store != second + 1)
                  break;
                bool exclusive = !(block.indirect() &&
                                   block.indirect()->target.index == first);
                if (block.branch())
                  exclusive = exclusive &&
                              block.branch()->left.index != first &&
                              block.branch()->right.index != first;
                for (std::size_t j = first + 1;
                     exclusive && j < block.nodes().size(); j++)
                  if (j != *first_store && j != second &&
                      references(block.nodes()[j],
                                 static_cast<std::uint32_t>(first)))
                    exclusive = false;
                if (exclusive) {
                  pair_second[first] = static_cast<std::uint32_t>(second);
                  pair_first[second] = static_cast<std::uint32_t>(first);
                  pair_flags[first] = output.newUInt32();
                }
                break;
              }
              if (!ir::is_pure(right.op) &&
                  (right.op != ir::opcode::read_state ||
                   (first_store && right.slot.index ==
                                       block.nodes()[*first_store].slot.index)))
                break;
            }
          }
        }
        bool crossed_slot = false;
        std::uint32_t previous_store_source = block.info().instruction_count;
        for (std::size_t i = block.nodes().size(); i > 0; i--) {
          const let &node = block.nodes()[i - 1];
          if (previous_store_source != node.source_index) {
            if (guarded_sources[node.source_index])
              std::ranges::fill(observed, 1);
            previous_store_source = node.source_index;
          }
          if (likely && !crossed_slot &&
              node.source_index <= conditional->source_index) {
            // An annulled slot cannot replace preceding architectural writes.
            std::ranges::fill(observed, 1);
            crossed_slot = true;
          }
          switch (node.op) {
          case ir::opcode::read_state:
            observed[node.slot.index] = 1;
            break;
          case ir::opcode::write_state:
            keep_store[i - 1] = observed[node.slot.index];
            observed[node.slot.index] = 0;
            break;
          default:
            if (!ir::is_pure(node.op))
              std::ranges::fill(observed, 1);
            break;
          }
        }
        if (const let status = output.bind(labels[region_index]))
          return failure(status, 0);
        if (fuse) {
          const let allowed = output.newLabel();
          const let stopped = output.newLabel();
          const let total = output.newUInt64();
          const let remaining = output.newUInt64();
          const let pc = output.newUInt64();
          if (errors.code)
            return failure(errors.code, 0);
          if (const let e = output.mov(pc, asmjit::Imm{block.info().start_pc}))
            return failure(e, 0);
          if (const let e = output.mov(
                  total.r32(),
                  asmjit::x86::Mem{context,
                                   offsetof(compiled_block::execution_context,
                                            executed_slots),
                                   4}))
            return failure(e, 0);
          const bool cpu_effect =
              std::ranges::any_of(block.nodes(), [](const ir::node &node) {
                return node.op == ir::opcode::read_cop0 ||
                       node.op == ir::opcode::read_cop0_64 ||
                       node.op == ir::opcode::write_cop0 ||
                       node.op == ir::opcode::write_cop0_64 ||
                       node.op == ir::opcode::tlb_operation ||
                       node.op == ir::opcode::exception_return;
              });
          if (cpu_effect) {
            if (const let e = output.test(total, total))
              return failure(e, 0);
            if (const let e = output.jnz(stopped))
              return failure(e, 0);
          }
          if (const let e = output.mov(
                  remaining,
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, budget), 8}))
            return failure(e, 0);
          if (const let e = output.sub(remaining, total))
            return failure(e, 0);
          if (const let e =
                  output.cmp(remaining, block.info().instruction_count))
            return failure(e, 0);
          if (const let e = output.jae(allowed))
            return failure(e, 0);
          if (const let e = output.bind(stopped))
            return failure(e, 0);
          if (const let e = finish(pc))
            return failure(e, 0);
          if (const let e = output.bind(allowed))
            return failure(e, 0);
          if (const let e = output.mov(
                  asmjit::x86::Mem{context,
                                   offsetof(compiled_block::execution_context,
                                            instruction_base),
                                   4},
                  total.r32()))
            return failure(e, 0);
          if (const let e = output.add(total, block.info().instruction_count))
            return failure(e, 0);
          if (const let e = output.mov(
                  asmjit::x86::Mem{context,
                                   offsetof(compiled_block::execution_context,
                                            executed_slots),
                                   4},
                  total.r32()))
            return failure(e, 0);
          if (const let e = output.mov(
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, region), 4},
                  region_index))
            return failure(e, 0);
          if (const let e = output.inc(asmjit::x86::Mem{
                  context,
                  offsetof(compiled_block::execution_context, executed_blocks),
                  4}))
            return failure(e, 0);
          if (likely)
            if (const let e = output.mov(
                    asmjit::x86::Mem{context,
                                     offsetof(compiled_block::execution_context,
                                              annulled_slot),
                                     1},
                    0))
              return failure(e, 0);
        }

        const let operand = [&](ir::value_id id)
            -> std::expected<asmjit::x86::Gp, asmjit::Error> {
          let &value = values[id.index];
          const let &node = block.nodes()[id.index];
          if (!value.isValid() && node.op == ir::opcode::constant) {
            value = node.type == ir::width::bits64 ? output.newUInt64()
                                                   : output.newUInt32();
            if (errors.code)
              return std::unexpected(errors.code);
            if (const let status =
                    output.mov(value, asmjit::Imm{node.data.immediate}))
              return std::unexpected(status);
          }
          return value;
        };
        const let immediate = [&](ir::value_id id) {
          const let &node = block.nodes()[id.index];
          if (node.op != ir::opcode::constant)
            return false;
          constexpr let positive_limit = static_cast<std::uint64_t>(
              std::numeric_limits<std::int32_t>::max());
          return node.type == ir::width::bits32 ||
                 node.data.immediate <= positive_limit ||
                 node.data.immediate >=
                     std::numeric_limits<std::uint64_t>::max() - positive_limit;
        };

        const let checkpoint = [&](std::uint32_t source,
                                   bool force = false) -> asmjit::Error {
          if (!guard_writes && !force)
            return asmjit::kErrorOk;
          const let prefix = output.newUInt32();
          if (errors.code)
            return errors.code;
          if (fuse) {
            if (const let e = output.mov(
                    prefix,
                    asmjit::x86::Mem{context,
                                     offsetof(compiled_block::execution_context,
                                              instruction_base),
                                     4}))
              return e;
            if (const let e = output.add(prefix, source))
              return e;
          } else if (const let e = output.mov(prefix, source))
            return e;
          return output.mov(
              asmjit::x86::Mem{
                  context,
                  offsetof(compiled_block::execution_context, effect_prefix),
                  4},
              prefix);
        };
        const let cached_read = [&](asmjit::x86::Gp address,
                                    asmjit::x86::Gp result, std::uint32_t bytes,
                                    bool sign, decltype(output.newLabel()) slow,
                                    bool aligned = true) -> asmjit::Error {
          const let index = output.newUInt64();
          const let tag = output.newUInt64();
          const let host = output.newUIntPtr();
          const let offset = output.newUInt64();
          if (errors.code)
            return errors.code;
          if (bytes > 1 && aligned) {
            if (const let error = output.test(address, asmjit::Imm{bytes - 1}))
              return error;
            if (const let error = output.jnz(slow))
              return error;
          }
          if (const let error = output.mov(index, address))
            return error;
          if (const let error = output.shr(index, 12))
            return error;
          if (const let error = output.and_(
                  index, memory::address_space::first_level_entries - 1))
            return error;
          if (const let error = output.shl(index, 4))
            return error;
          if (const let error = output.mov(
                  host,
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, read_cache),
                      8}))
            return error;
          if (const let error = output.add(index, host))
            return error;
          if (const let error = output.mov(tag, address))
            return error;
          if (const let error = output.and_(tag, asmjit::Imm{-4096}))
            return error;
          if (const let error = output.or_(tag, 1))
            return error;
          if (const let error = output.cmp(
                  tag, asmjit::x86::Mem{index,
                                        offsetof(memory::cache_entry, tag), 8}))
            return error;
          if (const let error = output.jne(slow))
            return error;
          if (const let error = output.mov(
                  host,
                  asmjit::x86::Mem{
                      index, offsetof(memory::cache_entry, host_page), 8}))
            return error;
          if (const let error = output.mov(offset, address))
            return error;
          if (const let error = output.and_(offset, memory::page_size - 1))
            return error;
          if (const let error = output.add(host, offset))
            return error;
          const asmjit::x86::Mem data{host, 0, bytes};
          if (bytes == 8) {
            if (const let error = output.mov(result, data))
              return error;
            if (const let error = output.bswap(result))
              return error;
          } else if (bytes == 4) {
            if (const let error = output.mov(result.r32(), data))
              return error;
            if (const let error = output.bswap(result.r32()))
              return error;
            if (sign)
              if (const let error = output.movsxd(result, result.r32()))
                return error;
          } else if (bytes == 2) {
            if (const let error = output.movzx(result.r32(), data))
              return error;
            if (const let error = output.ror(result.r16(), 8))
              return error;
            if (sign)
              if (const let error = output.movsx(result, result.r16()))
                return error;
          } else if (sign) {
            if (const let error = output.movsx(result, data))
              return error;
          } else if (const let error = output.movzx(result.r32(), data))
            return error;
          return asmjit::kErrorOk;
        };
        const let check_memory = [&](const ir::node &node) -> asmjit::Error {
          const let success = output.newLabel();
          if (errors.code)
            return errors.code;
          if (const let emitted = output.cmp(
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, failed), 1},
                  0))
            return emitted;
          if (const let emitted = output.je(success))
            return emitted;
          const let fault_pc = output.newUInt64();
          if (errors.code)
            return errors.code;
          if (const let emitted =
                  output.mov(fault_pc, asmjit::Imm{block.source_pc(node)}))
            return emitted;
          if (const let emitted = finish(fault_pc))
            return emitted;
          if (const let emitted = output.bind(success))
            return emitted;
          return asmjit::kErrorOk;
        };

        const let check_writes = [&](std::uint32_t source) -> asmjit::Error {
          const let clean = output.newLabel();
          if (errors.code)
            return errors.code;
          if (const let emitted = output.cmp(
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, writes) +
                          offsetof(memory::write_monitor, modified),
                      1},
                  0))
            return emitted;
          if (const let emitted = output.je(clean))
            return emitted;
          const let slots = output.newUInt32();
          if (errors.code)
            return errors.code;
          if (const let e = output.mov(
                  slots,
                  asmjit::x86::Mem{context,
                                   offsetof(compiled_block::execution_context,
                                            instruction_base),
                                   4}))
            return e;
          if (const let e = output.add(slots, source + 1))
            return e;
          if (fuse) {
            if (const let e = output.mov(
                    asmjit::x86::Mem{context,
                                     offsetof(compiled_block::execution_context,
                                              executed_slots),
                                     4},
                    slots))
              return e;
          } else if (const let e = output.mov(
                         asmjit::x86::Mem{
                             context,
                             offsetof(compiled_block::execution_context,
                                      executed_slots),
                             4},
                         slots))
            return e;
          const let next = output.newUInt64();
          if (errors.code)
            return errors.code;
          if (const let emitted = output.mov(
                  next, asmjit::Imm{block.info().start_pc +
                                    (std::uint64_t{source} + 1) * 4}))
            return emitted;
          if (const let emitted = finish(next))
            return emitted;
          return output.bind(clean);
        };

        const let annulled = output.newLabel();
        if (errors.code)
          return failure(errors.code, block.nodes().size());
        bool guarded_slot = false;
        bool memory_instruction = false;
        std::uint32_t previous_source = 0;
        for (std::size_t i = 0; i < block.nodes().size(); i++) {
          const let &node = block.nodes()[i];
          if (node.source_index != previous_source) {
            if (guard_writes && memory_instruction)
              if (const let status = check_writes(previous_source))
                return failure(status, i);
            previous_source = node.source_index;
            memory_instruction = false;
          }
          if (likely && !guarded_slot &&
              node.source_index > conditional->source_index) {
            const let left = operand(conditional->left);
            const let right = operand(conditional->right);
            if (!left || !right)
              return failure(!left ? left.error() : right.error(), i);
            if (const let status = output.cmp(*left, *right))
              return failure(status, i);
            if (fuse) {
              const let execute_slot = output.newLabel();
              if (errors.code)
                return failure(errors.code, i);
              if (const let status = conditional->kind == ir::branch_kind::equal
                                         ? output.je(execute_slot)
                                         : output.jne(execute_slot))
                return failure(status, i);
              if (const let status = output.mov(
                      asmjit::x86::Mem{
                          context,
                          offsetof(compiled_block::execution_context,
                                   annulled_slot),
                          1},
                      1))
                return failure(status, i);
              if (const let status = output.jmp(annulled))
                return failure(status, i);
              if (const let status = output.bind(execute_slot))
                return failure(status, i);
            } else {
              if (const let status = conditional->kind == ir::branch_kind::equal
                                         ? output.jne(annulled)
                                         : output.je(annulled))
                return failure(status, i);
            }

            guarded_slot = true;
          }
          if (node.op == ir::opcode::tlb_operation) {
            uses_cop0 = true;
            uses_memory = true;
            asmjit::InvokeNode *call = nullptr;
            if (const let e = checkpoint(node.source_index))
              return failure(e, i);
            if (const let status = output.invoke(
                    &call, asmjit::Imm{&compiled_block::tlb_operation},
                    asmjit::FuncSignature::build<
                        void, compiled_block::execution_context *,
                        std::uint32_t>()))
              return failure(status, i);
            call->setArg(0, context);
            call->setArg(1, asmjit::Imm{node.data.immediate});
            memory_instruction = true;
            if (const let emitted = check_memory(node))
              return failure(emitted, i);
            continue;
          }
          if (node.op == ir::opcode::memory_barrier) {
            if (const let status = output.mfence())
              return failure(status, i);
            continue;
          }
          if (node.op == ir::opcode::alias) {
            const let input = operand(node.data.inputs.left);
            if (!input)
              return failure(input.error(), i);
            values[i] = *input;
            continue;
          }
          if (node.op == ir::opcode::cache_index) {
            uses_memory = true;
            const let address = operand(node.data.inputs.left);
            if (!address)
              return failure(address.error(), i);
            asmjit::InvokeNode *call = nullptr;
            if (const let e = checkpoint(node.source_index))
              return failure(e, i);
            if (const let status = output.invoke(
                    &call, asmjit::Imm{&compiled_block::cache_index},
                    asmjit::FuncSignature::build<
                        void, compiled_block::execution_context *,
                        std::uint64_t>()))
              return failure(status, i);
            call->setArg(0, context);
            call->setArg(1, *address);
            memory_instruction = true;
            if (const let emitted = check_memory(node))
              return failure(emitted, i);
            continue;
          }
          if (node.op == ir::opcode::constant)
            continue;
          if (node.op == ir::opcode::store_conditional32 ||
              node.op == ir::opcode::load32_left ||
              node.op == ir::opcode::load32_right ||
              node.op == ir::opcode::load64_left ||
              node.op == ir::opcode::load64_right) {
            uses_memory = true;
            const let address = operand(node.data.inputs.left);
            const let previous = operand(node.data.inputs.right);
            if (!address || !previous)
              return failure(!address ? address.error() : previous.error(), i);
            const let result = output.newUInt64();
            if (errors.code)
              return failure(errors.code, i);
            const bool fast_merge = node.op != ir::opcode::store_conditional32;
            const let loaded = output.newLabel();
            if (errors.code)
              return failure(errors.code, i);
            if (fuse && pair_first[i] != UINT32_MAX) {
              const let fallback = output.newLabel();
              if (errors.code)
                return failure(errors.code, i);
              if (const let e = output.test(pair_flags[pair_first[i]],
                                            pair_flags[pair_first[i]]))
                return failure(e, i);
              if (const let e = output.jz(fallback))
                return failure(e, i);
              if (const let e = output.mov(result, values[pair_first[i]]))
                return failure(e, i);
              if (const let e = output.jmp(loaded))
                return failure(e, i);
              if (const let e = output.bind(fallback))
                return failure(e, i);
            }
            if (fuse && pair_second[i] != UINT32_MAX) {
              const std::uint32_t bytes =
                  node.op == ir::opcode::load32_left ? 4u : 8u;
              const let fallback = output.newLabel();
              const let offset = output.newUInt32();
              if (errors.code)
                return failure(errors.code, i);
              if (const let e = output.xor_(pair_flags[i], pair_flags[i]))
                return failure(e, i);
              if (const let e = output.mov(offset, address->r32()))
                return failure(e, i);
              if (const let e = output.and_(offset, memory::page_size - 1))
                return failure(e, i);
              if (const let e = output.cmp(offset, memory::page_size - bytes))
                return failure(e, i);
              if (const let e = output.ja(fallback))
                return failure(e, i);
              if (const let e = cached_read(*address, result, bytes, bytes == 4,
                                            fallback, false))
                return failure(e, i);
              if (const let e = output.mov(pair_flags[i], 1))
                return failure(e, i);
              if (const let e = output.jmp(loaded))
                return failure(e, i);
              if (const let e = output.bind(fallback))
                return failure(e, i);
            }
            if (fast_merge) {
              const std::uint32_t bytes =
                  node.op == ir::opcode::load64_left ||
                          node.op == ir::opcode::load64_right
                      ? 8u
                      : 4u;
              const bool right = node.op == ir::opcode::load32_right ||
                                 node.op == ir::opcode::load64_right;
              const let slow = output.newLabel();
              const let aligned = output.newUInt64();
              const let shifted = output.newUInt64();
              const let mask = output.newUInt64();
              const let shift = output.newUInt32();
              if (errors.code)
                return failure(errors.code, i);
              if (const let e = output.mov(aligned, *address))
                return failure(e, i);
              if (const let e = output.and_(
                      aligned, asmjit::Imm{-static_cast<std::int64_t>(bytes)}))
                return failure(e, i);
              if (const let e =
                      cached_read(aligned, shifted, bytes, false, slow))
                return failure(e, i);
              if (const let e = output.mov(shift, address->r32()))
                return failure(e, i);
              if (const let e = output.and_(shift, bytes - 1))
                return failure(e, i);
              if (right) {
                if (const let e = output.xor_(shift, bytes - 1))
                  return failure(e, i);
              }
              if (const let e = output.shl(shift, 3))
                return failure(e, i);
              if (const let e = output.mov(
                      mask, asmjit::Imm{bytes == 8 ? UINT64_MAX : UINT32_MAX}))
                return failure(e, i);
              let loaded_part = bytes == 8 ? shifted : shifted.r32();
              let preserve = bytes == 8 ? mask : mask.r32();
              if (const let e = right ? output.shr(loaded_part, shift.r8())
                                      : output.shl(loaded_part, shift.r8()))
                return failure(e, i);
              if (const let e = right ? output.shr(preserve, shift.r8())
                                      : output.shl(preserve, shift.r8()))
                return failure(e, i);
              if (const let e = output.not_(preserve))
                return failure(e, i);
              if (const let e = output.mov(result, *previous))
                return failure(e, i);
              let merged = bytes == 8 ? result : result.r32();
              if (const let e = output.and_(merged, preserve))
                return failure(e, i);
              if (const let e = output.or_(merged, loaded_part))
                return failure(e, i);
              if (bytes == 4)
                if (const let e = output.movsxd(result, result.r32()))
                  return failure(e, i);
              if (const let e = output.jmp(loaded))
                return failure(e, i);
              if (const let e = output.bind(slow))
                return failure(e, i);
            }
            const let target =
                node.op == ir::opcode::store_conditional32
                    ? &compiled_block::store_conditional32
                : node.op == ir::opcode::load64_left
                    ? &compiled_block::load_merge<std::uint64_t, false>
                : node.op == ir::opcode::load64_right
                    ? &compiled_block::load_merge<std::uint64_t, true>
                : node.op == ir::opcode::load32_left
                    ? &compiled_block::load_merge<std::uint32_t, false>
                    : &compiled_block::load_merge<std::uint32_t, true>;
            asmjit::InvokeNode *call = nullptr;
            if (const let e = checkpoint(node.source_index))
              return failure(e, i);
            if (const let status = output.invoke(
                    &call, asmjit::Imm{target},
                    asmjit::FuncSignature::build<
                        std::uint64_t, compiled_block::execution_context *,
                        std::uint64_t, std::uint64_t>()))
              return failure(status, i);
            call->setArg(0, context);
            call->setArg(1, *address);
            call->setArg(2, *previous);
            call->setRet(0, result);
            values[i] = result;
            memory_instruction = true;
            if (const let emitted = check_memory(node))
              return failure(emitted, i);
            if (fast_merge)
              if (const let e = output.bind(loaded))
                return failure(e, i);
            continue;
          }
          if (node.op == ir::opcode::load16_signed ||
              node.op == ir::opcode::load16_unsigned ||
              node.op == ir::opcode::store16 ||
              node.op == ir::opcode::load8_unsigned ||
              node.op == ir::opcode::load8_signed ||
              node.op == ir::opcode::load_linked32 ||
              node.op == ir::opcode::load32_unsigned ||
              node.op == ir::opcode::load32_signed ||
              node.op == ir::opcode::load64 || node.op == ir::opcode::store8 ||
              node.op == ir::opcode::store64_left ||
              node.op == ir::opcode::store64_right ||
              node.op == ir::opcode::store32_left ||
              node.op == ir::opcode::store32_right ||
              node.op == ir::opcode::store32 ||
              node.op == ir::opcode::store64) {
            uses_memory = true;
            const let address = operand(node.data.inputs.left);
            if (!address)
              return failure(address.error(), i);
            asmjit::InvokeNode *call = nullptr;
            asmjit::Error status;
            const bool fast_load = ir::produces_value(node.op) &&
                                   node.op != ir::opcode::load_linked32;
            const let loaded = output.newLabel();
            if (errors.code)
              return failure(errors.code, i);
            if (ir::produces_value(node.op)) {
              const let result = output.newUInt64();
              if (errors.code)
                return failure(errors.code, i);
              values[i] = result;
              if (fast_load) {
                const std::uint32_t bytes =
                    node.op == ir::opcode::load8_unsigned ||
                            node.op == ir::opcode::load8_signed
                        ? 1u
                    : node.op == ir::opcode::load16_unsigned ||
                            node.op == ir::opcode::load16_signed
                        ? 2u
                    : node.op == ir::opcode::load64 ? 8u
                                                    : 4u;
                const let slow = output.newLabel();
                const bool sign = node.op == ir::opcode::load8_signed ||
                                  node.op == ir::opcode::load16_signed ||
                                  node.op == ir::opcode::load32_signed;
                if (const let error =
                        cached_read(*address, result, bytes, sign, slow))
                  return failure(error, i);
                if (const let error = output.jmp(loaded))
                  return failure(error, i);
                if (const let error = output.bind(slow))
                  return failure(error, i);
              }
              const let target =
                  node.op == ir::opcode::load16_signed
                      ? &compiled_block::load<std::uint16_t>
                  : node.op == ir::opcode::load16_unsigned
                      ? &compiled_block::load<std::uint16_t, false>
                  : node.op == ir::opcode::load8_unsigned
                      ? &compiled_block::load<std::uint8_t, false>
                  : node.op == ir::opcode::load8_signed
                      ? &compiled_block::load<std::uint8_t>
                  : node.op == ir::opcode::load_linked32
                      ? &compiled_block::load<std::uint32_t, true, true>
                  : node.op == ir::opcode::load32_unsigned
                      ? &compiled_block::load<std::uint32_t, false>
                  : node.op == ir::opcode::load32_signed
                      ? &compiled_block::load<std::uint32_t>
                      : &compiled_block::load<std::uint64_t>;
              if (const let e = checkpoint(node.source_index))
                return failure(e, i);
              status = output.invoke(
                  &call, asmjit::Imm{target},
                  asmjit::FuncSignature::build<
                      std::uint64_t, compiled_block::execution_context *,
                      std::uint64_t>());
              if (status)
                return failure(status, i);
              call->setRet(0, result);
            } else {
              const let value = operand(node.data.inputs.right);
              if (!value)
                return failure(value.error(), i);
              const let target =
                  node.op == ir::opcode::store64_left
                      ? &compiled_block::store_merge<std::uint64_t, false>
                  : node.op == ir::opcode::store64_right
                      ? &compiled_block::store_merge<std::uint64_t, true>
                  : node.op == ir::opcode::store32_left
                      ? &compiled_block::store_merge<std::uint32_t, false>
                  : node.op == ir::opcode::store32_right
                      ? &compiled_block::store_merge<std::uint32_t, true>
                  : node.op == ir::opcode::store16
                      ? &compiled_block::store<std::uint16_t>
                  : node.op == ir::opcode::store8
                      ? &compiled_block::store<std::uint8_t>
                  : node.op == ir::opcode::store32
                      ? &compiled_block::store<std::uint32_t>
                      : &compiled_block::store<std::uint64_t>;
              if (const let e = checkpoint(node.source_index))
                return failure(e, i);
              status =
                  output.invoke(&call, asmjit::Imm{target},
                                asmjit::FuncSignature::build<
                                    void, compiled_block::execution_context *,
                                    std::uint64_t, std::uint64_t>());
              if (status)
                return failure(status, i);
              call->setArg(2, *value);
            }
            call->setArg(0, context);
            call->setArg(1, *address);
            memory_instruction = true;
            if (const let emitted = check_memory(node))
              return failure(emitted, i);
            if (fast_load)
              if (const let error = output.bind(loaded))
                return failure(error, i);
            continue;
          }
          if (node.op == ir::opcode::multiply_unsigned32 ||
              node.op == ir::opcode::multiply32 ||
              node.op == ir::opcode::divide_signed32 ||
              node.op == ir::opcode::divide_unsigned32) {
            uses_hilo = true;
            const let left = operand(node.data.inputs.left);
            const let right = operand(node.data.inputs.right);
            if (!left || !right)
              return failure(!left ? left.error() : right.error(), i);
            const let target = node.op == ir::opcode::multiply32
                                   ? &compiled_block::multiply32
                               : node.op == ir::opcode::multiply_unsigned32
                                   ? &compiled_block::multiply_unsigned32
                               : node.op == ir::opcode::divide_signed32
                                   ? &compiled_block::divide_signed32
                                   : &compiled_block::divide_unsigned32;
            asmjit::InvokeNode *call = nullptr;
            if (const let e = checkpoint(node.source_index))
              return failure(e, i);
            if (const let status =
                    output.invoke(&call, asmjit::Imm{target},
                                  asmjit::FuncSignature::build<
                                      void, compiled_block::execution_context *,
                                      std::uint64_t, std::uint64_t>()))
              return failure(status, i);
            call->setArg(0, context);
            call->setArg(1, *left);
            call->setArg(2, *right);
            continue;
          }
          if (node.op == ir::opcode::read_hilo ||
              node.op == ir::opcode::write_hilo) {
            uses_hilo = true;
            const let base = output.newUIntPtr();
            if (errors.code)
              return failure(errors.code, i);
            if (const let status = output.mov(
                    base,
                    asmjit::x86::Mem{
                        context,
                        offsetof(compiled_block::execution_context, hilo), 8}))
              return failure(status, i);
            const asmjit::x86::Mem location{base, node.slot.index * 8, 8};
            if (node.op == ir::opcode::read_hilo) {
              const let value = output.newUInt64();
              if (errors.code)
                return failure(errors.code, i);
              if (const let status = output.mov(value, location))
                return failure(status, i);
              values[i] = value;
            } else {
              const let value = operand(node.data.inputs.left);
              if (!value)
                return failure(value.error(), i);
              if (const let status = output.mov(location, *value))
                return failure(status, i);
            }
            continue;
          }
          if (node.op == ir::opcode::add_checked32) {
            uses_cop0 = true;
            const let left = operand(node.data.inputs.left);
            const let right = operand(node.data.inputs.right);
            if (!left || !right)
              return failure(!left ? left.error() : right.error(), i);
            const let result = output.newUInt64();
            const let vector = output.newUInt64();
            const let normal = output.newLabel();
            if (errors.code)
              return failure(errors.code, i);
            if (const let status = output.mov(result.r32(), left->r32()))
              return failure(status, i);
            if (const let status = output.add(result.r32(), right->r32()))
              return failure(status, i);
            if (const let status = output.jno(normal))
              return failure(status, i);
            const let branch = block.branch_pc();
            const bool delay_slot =
                branch && block.source_pc(node) == *branch + 4;
            const let pc = delay_slot ? *branch : block.source_pc(node);
            asmjit::InvokeNode *call = nullptr;
            if (const let e = checkpoint(node.source_index))
              return failure(e, i);
            if (const let status = output.invoke(
                    &call, asmjit::Imm{&compiled_block::overflow},
                    asmjit::FuncSignature::build<
                        std::uint64_t, compiled_block::execution_context *,
                        std::uint64_t, std::uint32_t>()))
              return failure(status, i);
            call->setArg(0, context);
            call->setArg(1, asmjit::Imm{pc});
            call->setArg(2, asmjit::Imm{delay_slot ? 1 : 0});
            call->setRet(0, vector);
            if (const let status = finish(vector))
              return failure(status, i);
            if (const let status = output.bind(normal))
              return failure(status, i);
            if (const let status = output.movsxd(result, result.r32()))
              return failure(status, i);
            values[i] = result;
            continue;
          }
          if (node.op == ir::opcode::exception_return) {
            uses_cop0 = true;
            const let target = output.newUInt64();
            if (errors.code)
              return failure(errors.code, i);
            asmjit::InvokeNode *call = nullptr;
            if (const let e = checkpoint(node.source_index))
              return failure(e, i);
            if (const let status = output.invoke(
                    &call, asmjit::Imm{&compiled_block::exception_return},
                    asmjit::FuncSignature::build<
                        std::uint64_t, compiled_block::execution_context *>()))
              return failure(status, i);
            call->setArg(0, context);
            call->setRet(0, target);
            if (const let status = finish(target))
              return failure(status, i);
            continue;
          }
          if (node.op == ir::opcode::syscall ||
              node.op == ir::opcode::breakpoint) {
            uses_cop0 = true;
            const let vector = output.newUInt64();
            if (errors.code)
              return failure(errors.code, i);
            const let branch = block.branch_pc();
            const bool delay_slot =
                branch && block.source_pc(node) == *branch + 4;
            const let pc = delay_slot ? *branch : block.source_pc(node);
            asmjit::InvokeNode *call = nullptr;
            if (const let e = checkpoint(node.source_index))
              return failure(e, i);
            if (const let status = output.invoke(
                    &call, asmjit::Imm{&compiled_block::software_exception},
                    asmjit::FuncSignature::build<
                        std::uint64_t, compiled_block::execution_context *,
                        std::uint64_t, std::uint32_t, std::uint32_t>()))
              return failure(status, i);
            call->setArg(0, context);
            call->setArg(1, asmjit::Imm{pc});
            call->setArg(2, asmjit::Imm{delay_slot ? 1 : 0});
            call->setArg(
                3, asmjit::Imm{node.op == ir::opcode::breakpoint ? 9u : 8u});
            call->setRet(0, vector);
            if (const let status = finish(vector))
              return failure(status, i);
            continue;
          }
          if (node.op == ir::opcode::read_cop1_64 ||
              node.op == ir::opcode::write_cop1_64) {
            uses_cop1 = true;
            const let base = output.newUIntPtr();
            if (errors.code)
              return failure(errors.code, i);
            if (const let status = output.mov(
                    base,
                    asmjit::x86::Mem{
                        context,
                        offsetof(compiled_block::execution_context, cop1), 8}))
              return failure(status, i);
            const asmjit::x86::Mem location{base, node.slot.index * 8, 8};
            if (node.op == ir::opcode::read_cop1_64) {
              const let result = output.newUInt64();
              if (errors.code)
                return failure(errors.code, i);
              if (const let status = output.mov(result, location))
                return failure(status, i);
              values[i] = result;
            } else {
              const let value = operand(node.data.inputs.left);
              if (!value)
                return failure(value.error(), i);
              if (const let status = output.mov(location, *value))
                return failure(status, i);
            }
            continue;
          }
          if (node.op == ir::opcode::read_cop0 ||
              node.op == ir::opcode::read_cop0_64) {
            uses_cop0 = true;
            if (node.slot.index == 9 && (node.source_index || fuse)) {
              if (const let e = checkpoint(node.source_index, true))
                return failure(e, i);
              asmjit::InvokeNode *clock_call = nullptr;
              if (const let e = output.invoke(
                      &clock_call,
                      asmjit::Imm{&compiled_block::account_before_external},
                      asmjit::FuncSignature::build<void, void *>()))
                return failure(e, i);
              clock_call->setArg(0, context);
            }
            const let base = output.newUIntPtr();
            const let result = output.newUInt64();
            if (errors.code)
              return failure(errors.code, i);
            if (const let status = output.mov(
                    base,
                    asmjit::x86::Mem{
                        context,
                        offsetof(compiled_block::execution_context, cop0), 8}))
              return failure(status, i);
            const bool wide = node.op == ir::opcode::read_cop0_64;
            const asmjit::x86::Mem location{
                base,
                static_cast<std::int32_t>(
                    offsetof(mips64::cop0_state, registers) +
                    node.slot.index * 8),
                wide ? 8u : 4u};
            const let status = wide ? output.mov(result, location)
                                    : output.movsxd(result, location);
            if (status)
              return failure(status, i);
            values[i] = result;
            continue;
          }
          if (node.op == ir::opcode::write_cop0 ||
              node.op == ir::opcode::write_cop0_64) {
            uses_cop0 = true;
            const let value = operand(node.data.inputs.left);
            if (!value)
              return failure(value.error(), i);
            asmjit::InvokeNode *call = nullptr;
            if (const let e = checkpoint(node.source_index, true))
              return failure(e, i);
            if (const let status = output.invoke(
                    &call, asmjit::Imm{&compiled_block::write_cop0},
                    asmjit::FuncSignature::build<
                        void, compiled_block::execution_context *,
                        std::uint32_t, std::uint64_t, std::uint32_t>()))
              return failure(status, i);
            call->setArg(0, context);
            call->setArg(1, asmjit::Imm{node.slot.index});
            call->setArg(2, *value);
            call->setArg(
                3, asmjit::Imm{node.op == ir::opcode::write_cop0_64 ? 1 : 0});
            if (node.slot.index == 22)
              if (const let emitted = check_memory(node))
                return failure(emitted, i);
            continue;
          }
          if (node.op == ir::opcode::write_state) {
            if (!keep_store[i])
              continue;
            const let value = operand(node.data.inputs.left);
            if (!value)
              return failure(value.error(), i);
            const let status = output.mov(
                asmjit::x86::Mem{state, node.slot.index * 8, 8}, *value);
            if (status)
              return failure(status, i);
            continue;
          }
          const let result = node.type == ir::width::bits64
                                 ? output.newUInt64()
                                 : output.newUInt32();
          if (errors.code)
            return failure(errors.code, i);
          values[i] = result;
          asmjit::Error status = asmjit::kErrorOk;
          switch (node.op) {
          case ir::opcode::alias:
          case ir::opcode::constant:
            break;
          case ir::opcode::shift_left32_variable:
          case ir::opcode::shift_right32_variable:
          case ir::opcode::shift_right_arithmetic32_variable: {
            const let input = operand(node.data.inputs.left);
            const let count = operand(node.data.inputs.right);
            if (!input || !count)
              return failure(!input ? input.error() : count.error(), i);
            status = output.mov(result.r32(), input->r32());
            if (!status)
              status = node.op == ir::opcode::shift_left32_variable
                           ? output.shl(result.r32(), count->r8())
                       : node.op == ir::opcode::shift_right32_variable
                           ? output.shr(result.r32(), count->r8())
                           : output.sar(result.r32(), count->r8());
            if (!status)
              status = output.movsxd(result, result.r32());
            break;
          }
          case ir::opcode::shift_left64:
          case ir::opcode::shift_right64:
          case ir::opcode::shift_right_arithmetic64: {
            const let input = operand(node.data.shift.value);
            if (!input)
              return failure(input.error(), i);
            status = output.mov(result, *input);
            if (!status)
              status =
                  node.op == ir::opcode::shift_left64
                      ? output.shl(result, asmjit::Imm{node.data.shift.amount})
                  : node.op == ir::opcode::shift_right64
                      ? output.shr(result, asmjit::Imm{node.data.shift.amount})
                      : output.sar(result, asmjit::Imm{node.data.shift.amount});
            break;
          }
          case ir::opcode::shift_right_arithmetic32:
          case ir::opcode::shift_left32:
          case ir::opcode::shift_right32: {
            const let input = operand(node.data.shift.value);
            if (!input)
              return failure(input.error(), i);
            status = output.mov(result.r32(), input->r32());
            if (!status)
              status = node.op == ir::opcode::shift_left32
                           ? output.shl(result.r32(),
                                        asmjit::Imm{node.data.shift.amount})
                       : node.op == ir::opcode::shift_right32
                           ? output.shr(result.r32(),
                                        asmjit::Imm{node.data.shift.amount})
                           : output.sar(result.r32(),
                                        asmjit::Imm{node.data.shift.amount});
            if (!status)
              status = output.movsxd(result, result.r32());
            break;
          }
          case ir::opcode::sign_extend32: {
            const let input = operand(node.data.inputs.left);
            if (!input)
              return failure(input.error(), i);
            status = output.movsxd(result, input->r32());
            break;
          }
          case ir::opcode::read_state:
            status = output.mov(
                result, asmjit::x86::Mem{state, node.slot.index * 8, 8});
            break;
          case ir::opcode::unsigned_less:
          case ir::opcode::signed_less: {
            const let left = operand(node.data.inputs.left);
            if (!left)
              return failure(left.error(), i);
            if (const let emitted = output.mov(result, 0))
              return failure(emitted, i);
            const let right = node.data.inputs.right;
            if (immediate(right)) {
              status = output.cmp(
                  *left,
                  asmjit::Imm{block.nodes()[right.index].data.immediate});
            } else {
              const let value = operand(right);
              if (!value)
                return failure(value.error(), i);
              status = output.cmp(*left, *value);
            }
            if (!status)
              status = node.op == ir::opcode::unsigned_less
                           ? output.setb(result.r8())
                           : output.setl(result.r8());
            break;
          }
          case ir::opcode::subtract_wrap:
          case ir::opcode::add_wrap:
          case ir::opcode::bit_and:
          case ir::opcode::bit_or:
          case ir::opcode::bit_xor: {
            let left = node.data.inputs.left;
            let right = node.data.inputs.right;
            if (node.op != ir::opcode::subtract_wrap && !immediate(right) &&
                immediate(left))
              std::swap(left, right);
            const let left_value = operand(left);
            if (!left_value)
              return failure(left_value.error(), i);
            status = output.mov(result, *left_value);
            if (status)
              break;
            const let emit = [&](const auto &right_value) {
              if (node.op == ir::opcode::subtract_wrap)
                return output.sub(result, right_value);
              if (node.op == ir::opcode::add_wrap)
                return output.add(result, right_value);
              if (node.op == ir::opcode::bit_and)
                return output.and_(result, right_value);
              if (node.op == ir::opcode::bit_or)
                return output.or_(result, right_value);
              return output.xor_(result, right_value);
            };
            if (immediate(right)) {
              status =
                  emit(asmjit::Imm{block.nodes()[right.index].data.immediate});
            } else {
              const let right_value = operand(right);
              if (!right_value)
                return failure(right_value.error(), i);
              status = emit(*right_value);
            }
            break;
          }
          case ir::opcode::multiply_unsigned32:
          case ir::opcode::multiply32:
          case ir::opcode::divide_signed32:
          case ir::opcode::divide_unsigned32:
          case ir::opcode::read_hilo:
          case ir::opcode::write_hilo:
          case ir::opcode::add_checked32:
          case ir::opcode::tlb_operation:
          case ir::opcode::memory_barrier:
          case ir::opcode::cache_index:
          case ir::opcode::exception_return:
          case ir::opcode::syscall:
          case ir::opcode::breakpoint:
          case ir::opcode::read_cop1_64:
          case ir::opcode::write_cop1_64:
          case ir::opcode::read_cop0_64:
          case ir::opcode::write_cop0_64:
          case ir::opcode::read_cop0:
          case ir::opcode::write_cop0:
          case ir::opcode::write_state:
          case ir::opcode::load16_signed:
          case ir::opcode::load16_unsigned:
          case ir::opcode::load8_unsigned:
          case ir::opcode::load8_signed:
          case ir::opcode::store_conditional32:
          case ir::opcode::load_linked32:
          case ir::opcode::load32_unsigned:
          case ir::opcode::load32_signed:
          case ir::opcode::load32_left:
          case ir::opcode::load32_right:
          case ir::opcode::load64_left:
          case ir::opcode::load64_right:
          case ir::opcode::load64:
          case ir::opcode::store16:
          case ir::opcode::store8:
          case ir::opcode::store64_left:
          case ir::opcode::store64_right:
          case ir::opcode::store32_left:
          case ir::opcode::store32_right:
          case ir::opcode::store32:
          case ir::opcode::store64:
            break;
          }
          if (status || errors.code)
            return failure(status ? status : errors.code, i);
        }
        if (guard_writes && memory_instruction &&
            previous_source + 1 < block.info().instruction_count)
          if (const let status = check_writes(previous_source))
            return failure(status, block.nodes().size());
        if (const let status = output.bind(annulled))
          return failure(status, block.nodes().size());
        // Constants used by the exit path must dominate every limit check.
        if (fuse && block.branch()) {
          const let &branch = *block.branch();
          const let left = operand(branch.left);
          const let right = operand(branch.right);
          if (!left || !right)
            return failure(!left ? left.error() : right.error(), 0);
          if (likely) {
            if (const let e = output.cmp(*left, *right))
              return failure(e, 0);
            const asmjit::x86::Mem annulled_flag{
                context,
                offsetof(compiled_block::execution_context, annulled_slot), 1};
            if (const let e = branch.kind == ir::branch_kind::equal
                                  ? output.setne(annulled_flag)
                                  : output.sete(annulled_flag))
              return failure(e, 0);
          }
        }
        if (fuse) {
          if (likely) {
            const let annul = output.newUInt32();
            if (const let e = output.movzx(
                    annul,
                    asmjit::x86::Mem{context,
                                     offsetof(compiled_block::execution_context,
                                              annulled_slot),
                                     1}))
              return failure(e, 0);
            if (const let e = output.sub(
                    asmjit::x86::Mem{context,
                                     offsetof(compiled_block::execution_context,
                                              executed_slots),
                                     4},
                    annul))
              return failure(e, 0);
          }
          const let stop = output.newLabel();
          const let total = output.newUInt64();
          const let count = output.newUInt32();
          if (errors.code)
            return failure(errors.code, 0);
          if (const let e = output.mov(
                  total.r32(),
                  asmjit::x86::Mem{context,
                                   offsetof(compiled_block::execution_context,
                                            executed_slots),
                                   4}))
            return failure(e, 0);
          if (const let e = output.cmp(
                  total,
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, cycle_limit),
                      8}))
            return failure(e, 0);
          if (const let e = output.jae(stop))
            return failure(e, 0);
          if (const let e = output.mov(
                  count,
                  asmjit::x86::Mem{context,
                                   offsetof(compiled_block::execution_context,
                                            executed_blocks),
                                   4}))
            return failure(e, 0);
          if (const let e = output.cmp(
                  count,
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, block_limit),
                      4}))
            return failure(e, 0);
          if (const let e = output.jae(stop))
            return failure(e, 0);
          if (const let e = output.cmp(
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, writes) +
                          offsetof(memory::write_monitor, modified),
                      1},
                  0))
            return failure(e, 0);
          if (const let e = output.jne(stop))
            return failure(e, 0);
          const bool cpu_exit =
              std::ranges::any_of(block.nodes(), [](const ir::node &node) {
                return node.op == ir::opcode::write_cop0 ||
                       node.op == ir::opcode::write_cop0_64 ||
                       node.op == ir::opcode::read_cop0 ||
                       node.op == ir::opcode::read_cop0_64 ||
                       node.op == ir::opcode::tlb_operation ||
                       node.op == ir::opcode::cache_index;
              });
          if (!cpu_exit && !block.indirect()) {
            const let target_index =
                [&](std::uint64_t pc) -> std::optional<std::size_t> {
              for (std::size_t target = 0; target < blocks.size(); target++)
                if (blocks[target].info().start_pc == pc)
                  return target;
              return std::nullopt;
            };
            if (const let *branch = block.branch()) {
              const let taken = target_index(branch->target);
              const let fallthrough = target_index(block.next_pc());
              if (taken || fallthrough) {
                const let left = operand(branch->left);
                const let right = operand(branch->right);
                if (!left || !right)
                  return failure(!left ? left.error() : right.error(), 0);
                if (const let e = output.cmp(*left, *right))
                  return failure(e, 0);
                if (taken) {
                  if (const let e = branch->kind == ir::branch_kind::equal
                                        ? output.je(labels[*taken])
                                        : output.jne(labels[*taken]))
                    return failure(e, 0);
                  if (fallthrough)
                    if (const let e = output.jmp(labels[*fallthrough]))
                      return failure(e, 0);
                } else if (fallthrough) {
                  if (const let e = branch->kind == ir::branch_kind::equal
                                        ? output.jne(labels[*fallthrough])
                                        : output.je(labels[*fallthrough]))
                    return failure(e, 0);
                }
              }
            } else if (const let target =
                           target_index(block.jump() ? block.jump()->target
                                                     : block.next_pc())) {
              if (const let e = output.jmp(labels[*target]))
                return failure(e, 0);
            }
          }
          if (const let e = output.bind(stop))
            return failure(e, 0);
        }
        const let next_pc = output.newUInt64();
        if (errors.code)
          return failure(errors.code, block.nodes().size());
        if (const let status =
                output.mov(next_pc, asmjit::Imm{block.next_pc()}))
          return failure(status, block.nodes().size());
        if (block.branch()) {
          const let &branch = *block.branch();
          const let left = operand(branch.left);
          const let right = operand(branch.right);
          if (!left || !right)
            return failure(!left ? left.error() : right.error(),
                           block.nodes().size());
          const let skip = output.newLabel();
          const let done = output.newLabel();
          if (errors.code)
            return failure(errors.code, block.nodes().size());
          if (const let status = output.cmp(*left, *right))
            return failure(status, block.nodes().size());
          if (const let status = branch.kind == ir::branch_kind::equal
                                     ? output.jne(skip)
                                     : output.je(skip))
            return failure(status, block.nodes().size());
          if (const let status =
                  output.mov(next_pc, asmjit::Imm{branch.target}))
            return failure(status, block.nodes().size());
          if (likely)
            if (const let status = output.jmp(done))
              return failure(status, block.nodes().size());
          if (const let status = output.bind(skip))
            return failure(status, block.nodes().size());
          if (likely) {
            // PC alone cannot distinguish an annulled slot when target equals
            // fallthrough. The register-only compatibility entry has no
            // context.
            if (const let status = output.test(context, context))
              return failure(status, block.nodes().size());
            if (const let status = output.je(done))
              return failure(status, block.nodes().size());
            if (const let status = output.mov(
                    asmjit::x86::Mem{context,
                                     offsetof(compiled_block::execution_context,
                                              annulled_slot),
                                     1},
                    1))
              return failure(status, block.nodes().size());
            if (const let status = output.bind(done))
              return failure(status, block.nodes().size());
          }
        }
        if (const let *jump = block.jump())
          if (const let status = output.mov(next_pc, asmjit::Imm{jump->target}))
            return failure(status, block.nodes().size());
        if (const let *jump = block.indirect()) {
          const let target = operand(jump->target);
          if (!target)
            return failure(target.error(), block.nodes().size());
          if (const let status = output.mov(next_pc, *target))
            return failure(status, block.nodes().size());
        }
        if (const let status = finish(next_pc))
          return failure(status, block.nodes().size());
      } // IR blocks share one bounded native function.
      if (const let status = output.endFunc())
        return failure(status, 0);
      if (const let status = output.finalize())
        return failure(status, 0);
      compiled_block::entry_point entry = nullptr;
      if (const let status = runtime_->add(&entry, &code))
        return failure(status, 0);
      return compiled_block{runtime_,
                            entry,
                            slots,
                            uses_memory,
                            uses_cop0,
                            uses_hilo,
                            uses_cop1,
                            blocks.front().branch_pc(),
                            blocks.front().info().start_pc,
                            source_instructions,
                            guard_writes,
                            std::move(regions),
                            !fuse && !uses_memory && !uses_cop0 && !uses_hilo &&
                                !uses_cop1 &&
                                !(blocks.front().branch() &&
                                  blocks.front().branch()->annul_not_taken)};
    } catch (const std::bad_alloc &) {
      return std::unexpected(error{error_code::out_of_memory});
    }
  }

private:
  struct error_handler : asmjit::ErrorHandler {
    asmjit::Error code = asmjit::kErrorOk;
    fn handleError(asmjit::Error value, const char *,
                     asmjit::BaseEmitter *) noexcept -> void override {
      if (!code)
        code = value;
    }
  };

  explicit compiler(std::shared_ptr<asmjit::JitRuntime> runtime) noexcept
      : runtime_(std::move(runtime)) {}

  std::shared_ptr<asmjit::JitRuntime> runtime_;
};

} // namespace jit::amd64
