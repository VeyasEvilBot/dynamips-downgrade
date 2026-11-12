module;
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <dynamips/macros.h>
export module jit.tlb;
import jit.memory;

export namespace jit::tlb {
struct entry {
  std::uint64_t mask = 0;
  std::uint64_t high = 0;
  std::array<std::uint64_t, 2> low{};
};
enum class error { invalid_index, invalid_page_mask, miss, invalid, modified };

class table {
public:
  static constexpr std::size_t capacity = 64;

  fn empty() const noexcept -> bool { return occupied_ == 0; }

  [[nodiscard]] fn write(std::size_t index, entry value) noexcept
      -> std::expected<void, error> {
    if (index >= capacity)
      return std::unexpected(error::invalid_index);
    const let bytes = ((value.mask | 0x1fffULL) + 1) / 2;
    if ((value.mask & ~std::uint64_t{0x01ffe000}) != 0 ||
        !std::has_single_bit(bytes) || (std::countr_zero(bytes) & 1) != 0)
      return std::unexpected(error::invalid_page_mask);
    entries_[index] = value;
    occupied_ |= std::uint64_t{1} << index;
    return {};
  }

  [[nodiscard]] fn read(std::size_t index) const noexcept
      -> std::optional<entry> {
    if (index >= capacity)
      return std::nullopt;
    return entries_[index];
  }

  [[nodiscard]] fn probe(memory::virtual_address address,
                          std::uint8_t asid) const noexcept
      -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < capacity; i++)
      if ((occupied_ & (std::uint64_t{1} << i)) &&
          matches(entries_[i], address, asid))
        return i;
    return std::nullopt;
  }

  [[nodiscard]] fn translate(memory::virtual_address address,
                              std::uint8_t asid,
                              memory::access_kind access) const noexcept
      -> std::expected<memory::physical_address, error> {
    const let index = probe(address, asid);
    if (!index)
      return std::unexpected(error::miss);
    const let &value = entries_[*index];
    const let bytes = ((value.mask | 0x1fffULL) + 1) / 2;
    const let low = value.low[(address.value & bytes) != 0 ? 1 : 0];
    if (!(low & 2))
      return std::unexpected(error::invalid);
    if (access == memory::access_kind::write && !(low & 4))
      return std::unexpected(error::modified);
    const let base = ((low & 0x3fffffc0ULL) << 6) & ~(bytes - 1);
    return memory::physical_address{base | (address.value & (bytes - 1))};
  }

private:
  std::array<entry, capacity> entries_{};
  std::uint64_t occupied_ = 0;

  static fn matches(const entry &value, memory::virtual_address address,
                     std::uint8_t asid) noexcept -> bool {
    const bool global = (value.low[0] & value.low[1] & 1) != 0;
    return (global || (value.high & 255) == asid) &&
           (((value.high ^ address.value) & 0xc00000ffffffe000ULL &
             ~(value.mask | 0x1fffULL)) == 0);
  }
};
} // namespace jit::tlb
