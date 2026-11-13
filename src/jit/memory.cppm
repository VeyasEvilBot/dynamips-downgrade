module;

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include <dynamips/macros.h>

export module jit.memory;

export namespace jit::memory {

struct virtual_address {
  std::uint64_t value;
};

struct physical_address {
  std::uint64_t value;
};
enum class access_kind : std::uint8_t { read, write, execute };
enum class access_size : std::uint8_t {
  byte = 1,
  half = 2,
  word = 4,
  doubleword = 8
};
enum class permission : std::uint8_t { read = 1, write = 2, execute = 4 };

constexpr fn operator|(permission left, permission right) noexcept
    -> permission {
  return static_cast<permission>(static_cast<unsigned>(left) |
                                 static_cast<unsigned>(right));
}
enum class fault_reason : std::uint8_t {
  misaligned,
  unmapped,
  permission,
  unsupported_size,
  bus,
  tlb_miss,
  tlb_invalid,
  tlb_modified,
  address_error
};

struct fault {
  virtual_address address;
  fault_reason reason;
  access_kind access;
  access_size size;
};
enum class mapping_error {
  misaligned_page,
  invalid_physical_range,
  invalid_permissions,
  physical_conflict,
  capacity_exhausted,
  not_mapped,
  out_of_memory,
  invalid_operation
};

enum class update_kind : std::uint8_t { ram, device, protect, unmap };
struct device;

struct mapping_update {
  update_kind kind;
  virtual_address address;
  physical_address physical{};
  permission permissions = permission::read;
  device *target = nullptr;
};

struct update_error {
  std::size_t index;
  mapping_error reason;
};

struct update_result {
  std::size_t applied;
  std::size_t invalidated_pages;
  bool full_flush;
};
enum class device_error : std::uint8_t { unsupported_size, bus };

struct device_access {
  physical_address address;
  std::endian order;
  access_size size;
};

// Callbacks exchange numeric values in the requested byte order.
struct device {
  void *context;
  fn (*read)(void *, device_access) noexcept -> std::expected<std::uint64_t, device_error>;
  fn (*write)(void *, device_access, std::uint64_t) noexcept -> std::expected<void, device_error>;
};

// Borrowed backing covers the complete 4 KiB page containing physical.
struct translated_page {
  physical_address physical;
  std::byte *host_page;
  device *target;
};

using translation_callback = auto (*)(void *, virtual_address, access_kind,
                                      access_size) noexcept
    -> std::expected<translated_page, fault>;

inline constexpr std::size_t page_size = 4096;

struct cache_entry {
  std::uint64_t tag = 0;
  std::byte *host_page = nullptr;
};

static_assert(sizeof(cache_entry) == 16);
static_assert(sizeof(fault) == 16);
static_assert(sizeof(device_access) == 16);

// RAM occupies physical [0, size). One execution owner; borrowed RAM and
// devices outlive mappings. Change mappings or device callbacks only while
// stopped.
struct backing_stamp {
  std::span<const std::uint64_t> versions{};

  [[nodiscard]] inline fn value() const noexcept -> std::uint64_t {
    std::uint64_t result = 0;
    for (const let version : versions)
      result += version;
    return result;
  }
};

struct write_monitor {
  std::span<const std::span<const std::byte>> ranges{};
  bool modified = false;
  void *context = nullptr;
  void (*before_external)(void *) noexcept = nullptr;
};

class address_space {
public:
  // Nested scopes observe only the innermost execution. Hooks may account
  // CPU time; they must not access memory or edit mappings recursively.
  class monitoring_scope {
    address_space &owner_;
    write_monitor *previous_;

  public:
    inline monitoring_scope(address_space &owner,
                            write_monitor *monitor) noexcept
        : owner_(owner), previous_(std::exchange(owner.monitor_, monitor)) {}

    monitoring_scope(const monitoring_scope &) = delete;
    fn operator=(const monitoring_scope &) -> monitoring_scope & = delete;

    inline ~monitoring_scope() { owner_.monitor_ = previous_; }
  };

  [[nodiscard]] inline fn monitor_writes(write_monitor *monitor) noexcept
      -> monitoring_scope {
    return monitoring_scope{*this, monitor};
  }

  static constexpr std::size_t first_level_entries = 256;
  static constexpr std::size_t second_level_entries = 4096;
  address_space(const address_space &) = delete;
  fn operator=(const address_space &) -> address_space & = delete;

  inline fn mapping_generation() const noexcept -> std::uint64_t {
    return generation_;
  }

  [[nodiscard]] inline fn external_generation() const noexcept
      -> std::uint64_t {
    return external_generation_;
  }

  [[nodiscard]] inline fn read_cache() const noexcept -> const cache_entry * {
    return first_[static_cast<std::size_t>(access_kind::read)].data();
  }

  // Borrowed counters remain stable for the address space lifetime.
  [[nodiscard]] inline fn
  stamp(std::span<const std::byte> range) const noexcept -> backing_stamp {
    if (range.empty() || ram_.empty())
      return {};
    const let begin = std::max(reinterpret_cast<std::uintptr_t>(range.data()),
                               reinterpret_cast<std::uintptr_t>(ram_.data()));
    const let end =
        std::min(reinterpret_cast<std::uintptr_t>(range.data() + range.size()),
                 reinterpret_cast<std::uintptr_t>(ram_.data() + ram_.size()));
    if (begin >= end)
      return {};
    const let base = reinterpret_cast<std::uintptr_t>(ram_.data());
    const let first = (begin - base) / page_size;
    const let last = (end - 1 - base) / page_size;
    return {std::span{write_versions_}.subspan(first, last - first + 1)};
  }

  // The translation owner and returned backing outlive this address space.
  fn set_translation(void *context, translation_callback callback) noexcept
      -> void {
    translation_context_ = context;
    translate_ = callback;
    invalidate();
  }

  // Sealed execution uses static mappings without writable physical aliases.
  [[nodiscard]] fn seal_executable_pages() noexcept
      -> std::expected<void, mapping_error> {
    for (const let &entry : mappings_)
      if (entry.host && !entry.read_only &&
          (static_cast<unsigned>(entry.permissions) & 4u) != 0 &&
          !compatible_code(entry.address, entry.physical, entry.permissions,
                           true))
        return std::unexpected(mapping_error::physical_conflict);
    if (!sealed_code_) {
      sealed_code_ = true;
      invalidate();
    }
    return {};
  }

  // Successful edits remain applied if a later edit fails. All invalidations
  // finish before returning; no guest access can observe an unflushed batch.
  [[nodiscard]] fn
  apply_updates(std::span<const mapping_update> updates) noexcept
      -> std::expected<update_result, update_error> {
    invalidation_batch batch{*this};
    for (std::size_t i = 0; i < updates.size(); i++) {
      const let &update = updates[i];
      std::expected<void, mapping_error> result;
      switch (update.kind) {
      case update_kind::ram:
        result = map_ram(update.address, update.physical, update.permissions);
        break;
      case update_kind::device:
        result = update.target
                     ? map_device(update.address, update.physical,
                                  *update.target, update.permissions)
                     : std::unexpected(mapping_error::invalid_operation);
        break;
      case update_kind::protect:
        result = protect(update.address, update.permissions);
        break;
      case update_kind::unmap:
        result = unmap(update.address);
        break;
      default:
        result = std::unexpected(mapping_error::invalid_operation);
        break;
      }
      if (!result)
        return std::unexpected(update_error{i, result.error()});
    }
    return update_result{updates.size(), batch.count, batch.full};
  }

  [[nodiscard]] static fn create(std::span<std::byte> ram,
                                  std::size_t mapping_capacity) noexcept
      -> std::expected<std::unique_ptr<address_space>, mapping_error> {
    try {
      let result = std::unique_ptr<address_space>{new address_space{ram}};
      if (mapping_capacity > result->mappings_.max_size())
        return std::unexpected(mapping_error::capacity_exhausted);
      result->mappings_.reserve(mapping_capacity);
      result->write_versions_.resize(ram.size() / page_size +
                                     (ram.size() % page_size != 0));
      result->mapping_limit_ = mapping_capacity;
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(mapping_error::out_of_memory);
    }
  }

  [[nodiscard]] fn map_ram(virtual_address address, physical_address physical,
                            permission permissions) noexcept
      -> std::expected<void, mapping_error> {
    if (physical.value > ram_.size() ||
        page_size > ram_.size() - physical.value)
      return std::unexpected(mapping_error::invalid_physical_range);
    return map({address, physical, ram_.data() + physical.value, nullptr,
                permissions});
  }

  // Borrowed backing stays stable and unmodified while executable code is
  // active.
  [[nodiscard]] fn map_rom(virtual_address address, physical_address physical,
                            std::span<std::byte> backing) noexcept
      -> std::expected<void, mapping_error> {
    if (backing.size() != page_size)
      return std::unexpected(mapping_error::invalid_physical_range);
    if (physical.value < ram_.size())
      return std::unexpected(mapping_error::physical_conflict);
    const let begin = reinterpret_cast<std::uintptr_t>(backing.data());
    const let ram = reinterpret_cast<std::uintptr_t>(ram_.data());
    if (!ram_.empty() && ((begin >= ram && begin - ram < ram_.size()) ||
                          (ram >= begin && ram - begin < page_size)))
      return std::unexpected(mapping_error::physical_conflict);
    return map({address, physical, backing.data(), nullptr,
                permission::read | permission::execute, true});
  }

  [[nodiscard]] fn map_device(virtual_address address,
                               physical_address physical, device &target,
                               permission permissions) noexcept
      -> std::expected<void, mapping_error> {
    if (physical.value < ram_.size())
      return std::unexpected(mapping_error::physical_conflict);
    if ((static_cast<unsigned>(permissions) &
         static_cast<unsigned>(permission::execute)) != 0)
      return std::unexpected(mapping_error::invalid_permissions);
    return map({address, physical, nullptr, &target, permissions});
  }

  [[nodiscard]] fn unmap(virtual_address address) noexcept
      -> std::expected<void, mapping_error> {
    if (address.value % page_size != 0)
      return std::unexpected(mapping_error::misaligned_page);
    const let found = find(address);
    if (found == mappings_.end() || found->address.value != address.value)
      return std::unexpected(mapping_error::not_mapped);
    mappings_.erase(found);
    invalidate(address);
    return {};
  }

  [[nodiscard]] fn protect(virtual_address address,
                            permission permissions) noexcept
      -> std::expected<void, mapping_error> {
    if (address.value % page_size != 0)
      return std::unexpected(mapping_error::misaligned_page);
    if (!valid(permissions))
      return std::unexpected(mapping_error::invalid_permissions);
    const let found = find(address);
    if (found == mappings_.end() || found->address.value != address.value)
      return std::unexpected(mapping_error::not_mapped);
    if (found->read_only && (static_cast<unsigned>(permissions) & 2u) != 0)
      return std::unexpected(mapping_error::invalid_permissions);
    if (found->target && (static_cast<unsigned>(permissions) &
                          static_cast<unsigned>(permission::execute)) != 0)
      return std::unexpected(mapping_error::invalid_permissions);
    if (found->host &&
        !compatible_code(address, found->physical, permissions, sealed_code_))
      return std::unexpected(mapping_error::physical_conflict);
    found->permissions = permissions;
    invalidate(address);
    return {};
  }

  fn invalidate() noexcept -> void {
    external_effect();
    clear_reservation();
    for (let &table : first_)
      std::ranges::fill(table, cache_entry{});
    for (let &table : second_)
      std::ranges::fill(table, cache_entry{});
    generation_++;
  }

  template <std::unsigned_integral T, std::endian Order = std::endian::big>
  [[nodiscard]] fn load(virtual_address address) noexcept
      -> std::expected<T, fault> {
    return read<T, Order, access_kind::read>(address);
  }

  template <std::unsigned_integral T, std::endian Order = std::endian::big>
  [[nodiscard]] fn fetch(virtual_address address) noexcept
      -> std::expected<T, fault> {
    return read<T, Order, access_kind::execute>(address);
  }

  [[nodiscard]] fn cache_index(virtual_address address) noexcept
      -> std::expected<void, fault> {
    // Direct kernel indices select cache lines even beyond installed RAM.
    if (address.value >= 0xffffffff80000000ULL &&
        address.value < 0xffffffffc0000000ULL)
      return {};
    if (lookup<access_kind::read>(address))
      return {};
    const let mapped = resolve(address, access_kind::read, access_size::byte);
    if (!mapped)
      return std::unexpected(mapped.error());
    return {};
  }

  // Borrowed executable RAM/ROM, ending at the current page boundary.
  // Mappings and permissions must be rechecked after mapping_generation
  // changes.
  [[nodiscard]] fn executable_tail(virtual_address address) noexcept
      -> std::expected<std::span<const std::byte>, fault> {
    if (address.value % 4 != 0)
      return std::unexpected(fault{address, fault_reason::misaligned,
                                   access_kind::execute, access_size::word});
    const let entry = lookup<access_kind::execute>(address);
    const std::byte *host;
    if (entry)
      host = entry->host_page + (address.value & (page_size - 1));
    else {
      const let resolved =
          resolve(address, access_kind::execute, access_size::word);
      if (!resolved)
        return std::unexpected(resolved.error());
      host = resolved->host;
    }
    return std::span<const std::byte>{
        host, page_size - (address.value & (page_size - 1))};
  }

  template <std::unsigned_integral T, std::endian Order = std::endian::big>
  [[nodiscard]] fn store(virtual_address address, T value) noexcept
      -> std::expected<void, fault> {
    static_assert(sizeof(T) <= 8 && !std::same_as<T, bool>);
    static_assert(Order == std::endian::big || Order == std::endian::little);
    clear_reservation();
    constexpr let size = static_cast<access_size>(sizeof(T));
    if (address.value % sizeof(T) != 0)
      return std::unexpected(
          fault{address, fault_reason::misaligned, access_kind::write, size});
    const let entry = lookup<access_kind::write>(address);
    if (!entry)
      return store_slow<T, Order>(address, value);
    let *host = entry->host_page + (address.value & (page_size - 1));
    encode<T, Order>(host, value);
    written(host, sizeof(T));
    return {};
  }

  inline fn clear_reservation() noexcept -> void {
    if (reservation_.host)
      reservation_.host = nullptr;
  }

  // One execution owner. External RAM writers must clear the reservation.
  template <std::unsigned_integral T, std::endian Order = std::endian::big>
  [[nodiscard]] fn load_linked(virtual_address address) noexcept
      -> std::expected<T, fault> {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    clear_reservation();
    const let host = ram_pointer<T, access_kind::read>(address);
    if (!host)
      return std::unexpected(host.error());
    const let value = decode<T, Order>(*host);
    reservation_ = {*host, static_cast<access_size>(sizeof(T))};
    return value;
  }

  template <std::unsigned_integral T, std::endian Order = std::endian::big>
  [[nodiscard]] fn store_conditional(virtual_address address, T value) noexcept
      -> std::expected<bool, fault> {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    const let linked = reservation_;
    clear_reservation();
    const let host = ram_pointer<T, access_kind::write>(address);
    if (!host)
      return std::unexpected(host.error());
    if (linked.host != *host ||
        linked.size != static_cast<access_size>(sizeof(T)))
      return false;
    encode<T, Order>(*host, value);
    written(*host, sizeof(T));
    return true;
  }

  // DMA/debugger/restore writes share CPU coherency and reservation tracking.
  [[nodiscard]] fn write_physical(physical_address address,
                                   std::span<const std::byte> bytes) noexcept
      -> std::expected<void, fault> {
    if (address.value > ram_.size() ||
        bytes.size() > ram_.size() - address.value)
      return std::unexpected(fault{{address.value},
                                   fault_reason::bus,
                                   access_kind::write,
                                   access_size::byte});
    if (bytes.empty())
      return {};
    clear_reservation();
    let *host = ram_.data() + address.value;
    std::memmove(host, bytes.data(), bytes.size());
    written(host, bytes.size());
    return {};
  }

  // A single-page byte write. Device partial writes need an explicit bus
  // contract.
  [[nodiscard]] fn store_bytes(virtual_address address,
                                std::span<const std::byte> bytes) noexcept
      -> std::expected<void, fault> {
    if (bytes.empty())
      return {};
    clear_reservation();
    if (bytes.size() > page_size - (address.value & (page_size - 1)))
      return std::unexpected(fault{address, fault_reason::unsupported_size,
                                   access_kind::write, access_size::byte});
    if (const let cached = lookup<access_kind::write>(address)) {
      let *host = cached->host_page + (address.value & (page_size - 1));
      std::memmove(host, bytes.data(), bytes.size());
      written(host, bytes.size());
      return {};
    }
    const let target = resolve(address, access_kind::write, access_size::byte);
    if (!target)
      return std::unexpected(target.error());
    if (target->target)
      return std::unexpected(fault{address, fault_reason::unsupported_size,
                                   access_kind::write, access_size::byte});
    std::memmove(target->host, bytes.data(), bytes.size());
    written(target->host, bytes.size());
    return {};
  }

private:
  struct invalidation_batch {
    address_space &owner;
    std::array<virtual_address, 64> pages;
    std::size_t count = 0;
    bool full = false;

    explicit invalidation_batch(address_space &space) noexcept : owner(space) {
      owner.batch_ = this;
    }

    ~invalidation_batch() {
      owner.batch_ = nullptr;
      if (full)
        owner.invalidate();
      else if (count != 0) {
        for (std::size_t i = 0; i < count; i++)
          owner.invalidate_page(pages[i]);
        owner.generation_++;
      }
    }

    fn enqueue(virtual_address address) noexcept -> void {
      if (full)
        return;
      for (std::size_t i = 0; i < count; i++)
        if (pages[i].value == address.value)
          return;
      if (count == pages.size()) {
        full = true;
        return;
      }
      pages[count] = address;
      count++;
    }
  };

  struct mapping {
    virtual_address address;
    physical_address physical;
    std::byte *host;
    device *target;
    permission permissions;
    bool read_only = false;
  };

  struct resolved_access {
    std::byte *host;
    device *target;
    physical_address physical;
  };

  alignas(
      64) std::array<std::array<cache_entry, first_level_entries>, 3> first_{};
  alignas(64)
      std::array<std::array<cache_entry, second_level_entries>, 3> second_{};

  struct reservation {
    std::byte *host = nullptr;
    access_size size = access_size::word;
  };

  write_monitor *monitor_ = nullptr;
  reservation reservation_;
  std::span<std::byte> ram_;
  std::vector<std::uint64_t> write_versions_;
  std::uint64_t external_generation_ = 0;

  struct backing_hint {
    std::uint64_t tag = 0;
    std::uintptr_t host = 0;
    std::uintptr_t target = 0;
    bool read_only = false;
  };

  std::array<backing_hint, 1024> backing_hints_{};
  std::array<std::uint64_t, 64> seen_backing_{};
  std::vector<mapping> mappings_;
  std::size_t mapping_limit_ = 0;
  std::uint64_t generation_ = 0;
  invalidation_batch *batch_ = nullptr;
  bool sealed_code_ = false;
  void *translation_context_ = nullptr;
  translation_callback translate_ = nullptr;

  explicit address_space(std::span<std::byte> ram) noexcept : ram_(ram) {}

  inline fn external_effect() noexcept -> void {
    external_generation_++;
    if (monitor_) {
      monitor_->modified = true;
      if (monitor_->before_external)
        monitor_->before_external(monitor_->context);
    }
  }

  inline fn written(const std::byte *host, std::size_t size) noexcept -> void {
    const let begin = reinterpret_cast<std::uintptr_t>(host);
    const let end = reinterpret_cast<std::uintptr_t>(host + size);
    const let base = reinterpret_cast<std::uintptr_t>(ram_.data());
    const let limit =
        reinterpret_cast<std::uintptr_t>(ram_.data() + ram_.size());
    if (begin < base || end > limit)
      external_generation_++;
    if (std::max(begin, base) < std::min(end, limit))
      for (let page = (std::max(begin, base) - base) / page_size;
           page <= (std::min(end, limit) - 1 - base) / page_size; page++)
        write_versions_[page]++;
    if (!monitor_ || monitor_->modified)
      return;
    for (const let range : monitor_->ranges) {
      const let code = reinterpret_cast<std::uintptr_t>(range.data());
      if (begin < code + range.size() && code < begin + size) {
        monitor_->modified = true;
        return;
      }
    }
  }

  static constexpr fn valid(permission permissions) noexcept -> bool {
    const let bits = static_cast<unsigned>(permissions);
    return bits != 0 && bits <= 7;
  }

  fn compatible_code(virtual_address address, physical_address physical,
                       permission permissions, bool enforce) const noexcept
      -> bool {
    if (!enforce)
      return true;
    const let bits = static_cast<unsigned>(permissions);
    if ((bits & 6u) == 6u)
      return false;
    for (const let &entry : mappings_) {
      if (!entry.host || entry.address.value == address.value ||
          entry.physical.value != physical.value)
        continue;
      const let other = static_cast<unsigned>(entry.permissions);
      if (((bits & 4u) && (other & 2u)) || ((bits & 2u) && (other & 4u)))
        return false;
    }
    return true;
  }

  fn find(virtual_address address) noexcept
      -> std::vector<mapping>::iterator {
    return std::lower_bound(mappings_.begin(), mappings_.end(), address.value,
                            [](const mapping &entry, std::uint64_t value) {
                              return entry.address.value < value;
                            });
  }

  static constexpr fn second_index(std::uint64_t page) noexcept
      -> std::size_t {
    return (page ^ (page / second_level_entries)) & (second_level_entries - 1);
  }

  fn invalidate(virtual_address address) noexcept -> void {
    clear_reservation();
    if (batch_) {
      batch_->enqueue(address);
      if (sealed_code_ && translate_)
        batch_->full = true;
      return;
    }
    if (sealed_code_ && translate_) {
      invalidate();
      return;
    }
    invalidate_page(address);
    generation_++;
  }

  fn invalidate_page(virtual_address address) noexcept -> void {
    external_effect();
    const let page = address.value / page_size;
    const let tag = address.value | 1;
    for (let &table : first_)
      if (table[page & (first_level_entries - 1)].tag == tag)
        table[page & (first_level_entries - 1)] = {};
    for (let &table : second_)
      if (table[second_index(page)].tag == tag)
        table[second_index(page)] = {};
  }

  fn map(mapping value) noexcept -> std::expected<void, mapping_error> {
    if (value.address.value % page_size != 0 ||
        value.physical.value % page_size != 0)
      return std::unexpected(mapping_error::misaligned_page);
    if (!valid(value.permissions))
      return std::unexpected(mapping_error::invalid_permissions);
    if (value.host && !compatible_code(value.address, value.physical,
                                       value.permissions, sealed_code_))
      return std::unexpected(mapping_error::physical_conflict);
    const let page = value.physical.value / page_size;
    const let seen_index = page & 4095;
    const let bit = std::uint64_t{1} << (seen_index % 64);
    let &hint = backing_hints_[page & (backing_hints_.size() - 1)];
    const let host = reinterpret_cast<std::uintptr_t>(value.host);
    const let target = reinterpret_cast<std::uintptr_t>(value.target);
    if (value.physical.value >= ram_.size() &&
        (seen_backing_[seen_index / 64] & bit) &&
        !(hint.tag == (value.physical.value | 1) && hint.host == host &&
          hint.target == target && hint.read_only == value.read_only))
      for (const let &entry : mappings_)
        if (entry.address.value != value.address.value &&
            entry.physical.value == value.physical.value &&
            (entry.host != value.host || entry.target != value.target ||
             entry.read_only != value.read_only))
          return std::unexpected(mapping_error::physical_conflict);
    const let found = find(value.address);
    if (found != mappings_.end() &&
        found->address.value == value.address.value) {
      *found = value;
    } else {
      if (mappings_.size() == mapping_limit_)
        return std::unexpected(mapping_error::capacity_exhausted);
      mappings_.insert(found, value);
    }
    if (value.physical.value >= ram_.size()) {
      // Bits never clear: unset proves absence; hint misses use the exact scan.
      seen_backing_[seen_index / 64] |= bit;
      hint = {value.physical.value | 1, host, target, value.read_only};
    }
    invalidate(value.address);
    return {};
  }

  template <access_kind Access>
  fn lookup(virtual_address address) noexcept -> const cache_entry * {
    constexpr let access = static_cast<std::size_t>(Access);
    const let page = address.value / page_size;
    const let tag = (address.value & ~std::uint64_t{page_size - 1}) | 1;
    let &first = first_[access][page & (first_level_entries - 1)];
    if (first.tag == tag)
      return &first;
    const let &second = second_[access][second_index(page)];
    if (second.tag == tag) {
      first = second;
      return &first;
    }
    return nullptr;
  }

  [[gnu::cold]] fn resolve(virtual_address address, access_kind access,
                            access_size size) noexcept
      -> std::expected<resolved_access, fault> {
    const virtual_address base{address.value & ~std::uint64_t{page_size - 1}};
    const let found = find(base);
    if (found == mappings_.end() || found->address.value != base.value) {
      if (!translate_)
        return std::unexpected(
            fault{address, fault_reason::unmapped, access, size});
      const let translated =
          translate_(translation_context_, address, access, size);
      if (!translated)
        return std::unexpected(translated.error());
      if (translated->target && access != access_kind::execute)
        return resolved_access{nullptr, translated->target,
                               translated->physical};
      if (!translated->host_page || translated->target)
        return std::unexpected(
            fault{address, fault_reason::permission, access, size});
      if (sealed_code_ && (access == access_kind::execute ||
                           (access == access_kind::write &&
                            !compatible_code(base,
                                             {translated->physical.value &
                                              ~std::uint64_t{page_size - 1}},
                                             permission::write, true))))
        return std::unexpected(
            fault{address, fault_reason::permission, access, size});
      const cache_entry cached{base.value | 1, translated->host_page};
      const let index = static_cast<std::size_t>(access);
      first_[index][(base.value / page_size) & (first_level_entries - 1)] =
          cached;
      second_[index][second_index(base.value / page_size)] = cached;
      return resolved_access{translated->host_page +
                                 (address.value & (page_size - 1)),
                             nullptr, translated->physical};
    }
    if ((static_cast<unsigned>(found->permissions) &
         (1u << static_cast<unsigned>(access))) == 0)
      return std::unexpected(
          fault{address, fault_reason::permission, access, size});
    const let offset = address.value & (page_size - 1);
    if (found->target) {
      if (access == access_kind::execute)
        return std::unexpected(
            fault{address, fault_reason::permission, access, size});
      return resolved_access{
          nullptr, found->target, {found->physical.value + offset}};
    }
    const cache_entry cached{base.value | 1, found->host};
    const let index = static_cast<std::size_t>(access);
    first_[index][(base.value / page_size) & (first_level_entries - 1)] =
        cached;
    second_[index][second_index(base.value / page_size)] = cached;
    return resolved_access{
        found->host + offset, nullptr, {found->physical.value + offset}};
  }

  static fn device_fault(virtual_address address, access_kind access,
                          access_size size, device_error error) noexcept
      -> fault {
    return {address,
            error == device_error::unsupported_size
                ? fault_reason::unsupported_size
                : fault_reason::bus,
            access, size};
  }

  template <std::unsigned_integral T, std::endian Order>
  static fn decode(const std::byte *host) noexcept -> T {
    T value;
    std::memcpy(&value, host, sizeof(T));
    if constexpr (Order != std::endian::native && sizeof(T) > 1)
      value = std::byteswap(value);
    return value;
  }

  template <std::unsigned_integral T, std::endian Order>
  static fn encode(std::byte *host, T value) noexcept -> void {
    if constexpr (Order != std::endian::native && sizeof(T) > 1)
      value = std::byteswap(value);
    std::memcpy(host, &value, sizeof(T));
  }

  template <std::unsigned_integral T, std::endian Order>
  [[gnu::noinline, gnu::cold]] fn store_slow(virtual_address address,
                                              T value) noexcept
      -> std::expected<void, fault> {
    constexpr let size = static_cast<access_size>(sizeof(T));
    let resolved = resolve(address, access_kind::write, size);
    if (!resolved)
      return std::unexpected(resolved.error());
    if (resolved->target) {
      if (!resolved->target->write)
        return std::unexpected(fault{address, fault_reason::unsupported_size,
                                     access_kind::write, size});
      external_effect();
      let result = resolved->target->write(
          resolved->target->context, {resolved->physical, Order, size}, value);
      if (!result)
        return std::unexpected(
            device_fault(address, access_kind::write, size, result.error()));
      return {};
    }
    encode<T, Order>(resolved->host, value);
    written(resolved->host, sizeof(T));
    return {};
  }

  template <std::unsigned_integral T, access_kind Access>
  fn ram_pointer(virtual_address address) noexcept
      -> std::expected<std::byte *, fault> {
    constexpr let size = static_cast<access_size>(sizeof(T));
    if (address.value % sizeof(T))
      return std::unexpected(
          fault{address, fault_reason::misaligned, Access, size});
    if (const let cached = lookup<Access>(address))
      return cached->host_page + (address.value & (page_size - 1));
    const let resolved = resolve(address, Access, size);
    if (!resolved)
      return std::unexpected(resolved.error());
    if (resolved->target)
      return std::unexpected(
          fault{address, fault_reason::unsupported_size, Access, size});
    return resolved->host;
  }

  template <std::unsigned_integral T, std::endian Order, access_kind Access>
  [[gnu::noinline, gnu::cold]] fn read_slow(virtual_address address) noexcept
      -> std::expected<T, fault> {
    constexpr let size = static_cast<access_size>(sizeof(T));
    let resolved = resolve(address, Access, size);
    if (!resolved)
      return std::unexpected(resolved.error());
    if (resolved->target) {
      if (!resolved->target->read)
        return std::unexpected(
            fault{address, fault_reason::unsupported_size, Access, size});
      external_effect();
      let result = resolved->target->read(resolved->target->context,
                                           {resolved->physical, Order, size});
      if (!result)
        return std::unexpected(
            device_fault(address, Access, size, result.error()));
      return static_cast<T>(*result);
    }
    return decode<T, Order>(resolved->host);
  }

  template <std::unsigned_integral T, std::endian Order, access_kind Access>
  fn read(virtual_address address) noexcept -> std::expected<T, fault> {
    static_assert(sizeof(T) <= 8 && !std::same_as<T, bool>);
    static_assert(Order == std::endian::big || Order == std::endian::little);
    constexpr let size = static_cast<access_size>(sizeof(T));
    if (address.value % sizeof(T) != 0)
      return std::unexpected(
          fault{address, fault_reason::misaligned, Access, size});
    const let entry = lookup<Access>(address);
    if (!entry)
      return read_slow<T, Order, Access>(address);
    return decode<T, Order>(entry->host_page +
                            (address.value & (page_size - 1)));
  }
};
} // namespace jit::memory
