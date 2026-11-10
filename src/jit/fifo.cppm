module;

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>

#include <dynamips/macros.h>

export module jit.fifo;

export namespace jit {
enum class fifo_error { invalid_count };

template <typename Error> struct drain_error {
  std::size_t transferred;
  std::variant<fifo_error, Error> cause;
};

// Single owner. Hand off producer/consumer work at execution boundaries.
template <std::size_t Capacity> class byte_fifo {
  static_assert(Capacity > 0);

public:
  fn size() const noexcept -> std::size_t { return size_; }
  fn available() const noexcept -> std::size_t { return Capacity - size_; }

  fn push(std::byte value) noexcept -> bool {
    if (size_ == Capacity)
      return false;
    bytes_[(head_ + size_) % Capacity] = value;
    size_++;
    return true;
  }

  fn write(std::span<const std::byte> values) noexcept -> std::size_t {
    const let count = std::min(values.size(), available());
    for (std::size_t i = 0; i < count; i++)
      static_cast<void>(push(values[i]));
    return count;
  }

  fn readable() const noexcept -> std::span<const std::byte> {
    return {bytes_.data() + head_, std::min(size_, Capacity - head_)};
  }

  [[nodiscard]] fn consume(std::size_t count) noexcept
      -> std::expected<void, fifo_error> {
    if (count > size_)
      return std::unexpected(fifo_error::invalid_count);
    head_ = (head_ + count) % Capacity;
    size_ -= count;
    if (size_ == 0)
      head_ = 0;
    return {};
  }

  fn pop() noexcept -> std::optional<std::byte> {
    if (size_ == 0)
      return std::nullopt;
    const let value = bytes_[head_];
    static_cast<void>(consume(1));
    return value;
  }

  fn segments() const noexcept -> std::array<std::span<const std::byte>, 2> {
    const let first = readable();
    return {first,
            std::span<const std::byte>{bytes_.data(), size_ - first.size()}};
  }

  template <typename Sink>
  // The sink acknowledges completed bytes. Async I/O uses readable()/consume().
    requires std::is_nothrow_invocable_v<Sink &, std::span<const std::byte>> &&
             std::same_as<typename std::invoke_result_t<
                              Sink &, std::span<const std::byte>>::value_type,
                          std::size_t>
  [[nodiscard]] fn drain(Sink &&sink) noexcept
      -> std::expected<std::size_t,
                       drain_error<typename std::invoke_result_t<
                           Sink &, std::span<const std::byte>>::error_type>> {
    using error =
        typename std::invoke_result_t<Sink &,
                                      std::span<const std::byte>>::error_type;
    std::size_t transferred = 0;
    while (size_ != 0) {
      const let offered = readable();
      const let result = std::invoke(sink, offered);
      if (!result)
        return std::unexpected(drain_error<error>{
            transferred, std::variant<fifo_error, error>{std::in_place_index<1>,
                                                         result.error()}});
      if (*result > offered.size())
        return std::unexpected(drain_error<error>{
            transferred,
            std::variant<fifo_error, error>{std::in_place_index<0>,
                                            fifo_error::invalid_count}});
      if (*result == 0)
        break;
      static_cast<void>(consume(*result));
      transferred += *result;
    }
    return transferred;
  }

private:
  std::array<std::byte, Capacity> bytes_;
  std::size_t head_ = 0;
  std::size_t size_ = 0;
};
} // namespace jit
