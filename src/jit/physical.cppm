module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include <dynamips/macros.h>

export module jit.physical;

import jit.memory;

export namespace jit::physical {
struct region {
  memory::physical_address base;
  std::uint64_t size;
  std::span<std::byte> backing;
  memory::device *target = nullptr;
  memory::permission permissions = memory::permission::read;
};

enum class error_code {
  capacity_exhausted,
  empty_region,
  misaligned_region,
  overflowing_region,
  invalid_backing,
  invalid_permissions,
  overlap
};

struct error {
  error_code code;
  std::size_t index;
};

// One execution owner. Borrowed backing and device endpoints outlive the bus.
class bus {
public:
  static constexpr std::size_t capacity = 32;

  bus(const bus &) = default;
  fn operator=(const bus &) -> bus & = default;
  bus(bus &&) noexcept = default;
  fn operator=(bus &&) noexcept -> bus & = default;

  [[nodiscard]] static fn create(std::span<const region> regions) noexcept
      -> std::expected<bus, error> {
    if (regions.size() > capacity)
      return std::unexpected(error{error_code::capacity_exhausted, capacity});
    bus result;
    for (std::size_t i = 0; i < regions.size(); i++) {
      const let &current = regions[i];
      if (!current.size)
        return std::unexpected(error{error_code::empty_region, i});
      if (current.base.value % memory::page_size ||
          current.size % memory::page_size)
        return std::unexpected(error{error_code::misaligned_region, i});
      if (current.size > UINT64_MAX - current.base.value)
        return std::unexpected(error{error_code::overflowing_region, i});
      const let permissions = static_cast<unsigned>(current.permissions);
      if (!permissions || permissions > 7 ||
          (current.target && (permissions & 4u)))
        return std::unexpected(error{error_code::invalid_permissions, i});
      if ((current.target && !current.backing.empty()) ||
          (!current.target && current.size > current.backing.size()))
        return std::unexpected(error{error_code::invalid_backing, i});
      for (std::size_t j = 0; j < i; j++) {
        const let &other = regions[j];
        if (current.base.value < other.base.value + other.size &&
            other.base.value < current.base.value + current.size)
          return std::unexpected(error{error_code::overlap, i});
      }
      result.regions_[i] = current;
    }
    result.count_ = regions.size();
    return result;
  }

  [[nodiscard]] fn resolve(memory::physical_address address,
                            memory::access_kind access,
                            memory::access_size size) const noexcept
      -> std::expected<memory::translated_page, memory::fault> {
    const let failure = [&](memory::fault_reason reason) {
      return std::unexpected(
          memory::fault{{address.value}, reason, access, size});
    };
    const let width = static_cast<unsigned>(size);
    if ((width != 1 && width != 2 && width != 4 && width != 8) ||
        static_cast<unsigned>(access) > 2)
      return failure(memory::fault_reason::unsupported_size);
    for (std::size_t i = 0; i < count_; i++) {
      const let &current = regions_[i];
      if (address.value < current.base.value ||
          address.value - current.base.value >= current.size)
        continue;
      const let offset = address.value - current.base.value;
      if (width > current.size - offset)
        return failure(memory::fault_reason::bus);
      if (!(static_cast<unsigned>(current.permissions) &
            (1u << static_cast<unsigned>(access))))
        return failure(memory::fault_reason::permission);
      const let page = offset & ~std::uint64_t{memory::page_size - 1};
      return memory::translated_page{address,
                                     current.target
                                         ? nullptr
                                         : current.backing.data() +
                                               static_cast<std::size_t>(page),
                                     current.target};
    }
    return failure(memory::fault_reason::bus);
  }

private:
  std::array<region, capacity> regions_{};
  std::size_t count_ = 0;

  bus() noexcept = default;
};
} // namespace jit::physical
