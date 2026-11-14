module;

#include <asio/buffer.hpp>
#include <asio/error.hpp>
#include <asio/io_context.hpp>
#include <asio/local/stream_protocol.hpp>
#include <asio/posix/stream_descriptor.hpp>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <asio/signal_set.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <sys/stat.h>
#include <csignal>
#include <cstdio>
#include <sys/un.h>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <new>
#include <print>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

#include <dynamips/macros.h>

export module host.console;

import native_fd;

export namespace dynamips::host {
// One execution owner. Completion handlers run only in poll() or close().
class console_transport {
public:
  static constexpr std::size_t capacity = 8192;
  console_transport(const console_transport &) = delete;
  fn operator=(const console_transport &) -> console_transport & = delete;

  [[nodiscard]] static fn listen(std::string_view path,
                                  bool process_signals = false) noexcept
      -> std::expected<std::unique_ptr<console_transport>, std::error_code> {
    if (path.empty() || path.find('\0') != std::string_view::npos)
      return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    if (path.size() >= sizeof(sockaddr_un{}.sun_path))
      return std::unexpected(
          std::make_error_code(std::errc::filename_too_long));
    try {
      let result = std::unique_ptr<console_transport>{new console_transport};
      const std::filesystem::path requested{path};
      std::error_code error;
      result->path_ = std::filesystem::absolute(requested, error);
      if (error)
        return std::unexpected(error);
      const asio::local::stream_protocol::endpoint endpoint{requested.native()};
      error = result->acceptor_.open(endpoint.protocol(), error);
      if (error)
        return std::unexpected(error);
      error = result->acceptor_.bind(endpoint, error);
      if (error)
        return std::unexpected(error);
      if (::lstat(result->path_.c_str(), &result->path_identity_) != 0)
        return std::unexpected(std::error_code{errno, std::generic_category()});
      result->owns_path_ = true;
      error = result->acceptor_.listen(1, error);
      if (error)
        return std::unexpected(error);
      if (process_signals) {
        error = result->signals_.add(SIGINT, error);
        if (error)
          return std::unexpected(error);
        error = result->signals_.add(SIGTERM, error);
        if (error)
          return std::unexpected(error);
        result->signals_.async_wait(
            [self = result.get()](std::error_code status, int) noexcept {
              if (!status)
                self->interrupted_ = true;
            });
      }
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          std::make_error_code(std::errc::not_enough_memory));
    } catch (const std::system_error &error) {
      return std::unexpected(error.code());
    }
  }

  [[nodiscard]] static fn adopt(file_descriptor descriptor) noexcept
      -> std::expected<std::unique_ptr<console_transport>, std::error_code> {
    if (descriptor.get() < 0)
      return std::unexpected(
          std::make_error_code(std::errc::bad_file_descriptor));
    try {
      let result = std::unique_ptr<console_transport>{new console_transport};
      std::error_code error;
      error = result->socket_.assign(asio::local::stream_protocol{},
                                     descriptor.get(), error);
      if (error)
        return std::unexpected(error);
      static_cast<void>(descriptor.release());
      result->connected_ = true;
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          std::make_error_code(std::errc::not_enough_memory));
    } catch (const std::system_error &error) {
      return std::unexpected(error.code());
    }
  }

  [[nodiscard]] static fn terminal(bool process_signals = false,
                                    int input = STDIN_FILENO,
                                    int output = STDOUT_FILENO) noexcept
      -> std::expected<std::unique_ptr<console_transport>, std::error_code> {
    let read_fd = duplicate_fd(input);
    if (!read_fd)
      return std::unexpected(read_fd.error());
    let write_fd = duplicate_fd(output);
    if (!write_fd)
      return std::unexpected(write_fd.error());
    const int read_flags = ::fcntl(read_fd->get(), F_GETFL, 0);
    const int write_flags = ::fcntl(write_fd->get(), F_GETFL, 0);
    if (read_flags < 0 || write_flags < 0)
      return std::unexpected(std::error_code{errno, std::generic_category()});
    try {
      let result = std::unique_ptr<console_transport>{new console_transport};
      std::error_code error;
      result->terminal_mode_ = true;
      result->input_.assign(read_fd->get(), error);
      if (error)
        return std::unexpected(error);
      static_cast<void>(read_fd->release());
      result->input_flags_ = read_flags;
      result->output_.assign(write_fd->get(), error);
      if (error)
        return std::unexpected(error);
      static_cast<void>(write_fd->release());
      result->output_flags_ = write_flags;
      if (::isatty(result->input_.native_handle())) {
        if (::tcgetattr(result->input_.native_handle(), &result->attributes_) <
            0)
          return std::unexpected(
              std::error_code{errno, std::generic_category()});
        let raw = result->attributes_;
        ::cfmakeraw(&raw);
        if (::tcsetattr(result->input_.native_handle(), TCSANOW, &raw) < 0)
          return std::unexpected(
              std::error_code{errno, std::generic_category()});
        result->interactive_ = true;
      }
      if (process_signals) {
        error = result->signals_.add(SIGINT, error);
        if (error)
          return std::unexpected(error);
        error = result->signals_.add(SIGTERM, error);
        if (error)
          return std::unexpected(error);
        result->signals_.async_wait(
            [self = result.get()](std::error_code status, int) noexcept {
              if (!status)
                self->interrupted_ = true;
            });
      }
      result->connected_ = true;
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          std::make_error_code(std::errc::not_enough_memory));
    } catch (const std::system_error &error) {
      return std::unexpected(error.code());
    }
  }

  ~console_transport() {
    if (const let error = shutdown(); error) {
      try {
        std::println(stderr, "console close: {}", error.message());
      } catch (...) {
      } // Diagnostics must not escape destruction.
    }
  }

  fn connected() const noexcept -> bool { return connected_; }
  fn input_closed() const noexcept -> bool { return input_closed_; }
  fn interrupted() const noexcept -> bool { return interrupted_; }
  fn pending_output() const noexcept -> std::size_t { return tx_size_; }
  fn received() const noexcept -> std::span<const std::byte> {
    return {rx_.data() + rx_head_, rx_size_};
  }

  [[nodiscard]] fn consume_received(std::size_t count) noexcept
      -> std::expected<void, std::error_code> {
    if (count > rx_size_)
      return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    rx_head_ += count;
    rx_size_ -= count;
    if (rx_size_ == 0)
      rx_head_ = 0;
    return {};
  }

  // Accepted bytes are copied into owned storage before the caller releases
  // them.
  fn send(std::span<const std::byte> bytes) noexcept -> std::size_t {
    if (!connected_ || stopping_ || failure_ || tx_size_ != 0)
      return 0;
    const let count = std::min(bytes.size(), tx_.size());
    std::copy_n(bytes.begin(), count, tx_.begin());
    tx_head_ = 0;
    tx_size_ = count;
    return count;
  }

  [[nodiscard]] fn poll(std::chrono::milliseconds wait = {}) noexcept
      -> std::expected<void, std::error_code> {
    if (stopping_)
      return std::unexpected(
          std::make_error_code(std::errc::operation_canceled));
    if (failure_)
      return std::unexpected(failure_);
    try {
      schedule();
      context_.restart();
      if (wait.count() > 0)
        context_.run_one_for(wait);
      for (unsigned i = 0; i < 8 && context_.poll_one() != 0; i++) {
      }
      if (!failure_)
        schedule();
    } catch (const std::bad_alloc &) {
      failure_ = std::make_error_code(std::errc::not_enough_memory);
    } catch (const std::system_error &error) {
      failure_ = error.code();
    }
    if (failure_)
      return std::unexpected(failure_);
    return {};
  }

  [[nodiscard]] fn close() noexcept -> std::expected<void, std::error_code> {
    const let error = shutdown();
    if (error)
      return std::unexpected(error);
    return {};
  }

private:
  // Buffers outlive the reactor and every operation that borrows their storage.
  std::array<std::byte, capacity> rx_{};
  std::array<std::byte, capacity> tx_{};
  asio::io_context context_;
  asio::local::stream_protocol::acceptor acceptor_{context_};
  asio::local::stream_protocol::socket socket_{context_};
  asio::signal_set signals_{context_};
  asio::posix::stream_descriptor input_{context_}, output_{context_};
  struct termios attributes_{};
  int input_flags_ = -1, output_flags_ = -1;
  bool terminal_mode_ = false, interactive_ = false;
  std::filesystem::path path_;
  struct stat path_identity_{};
  std::size_t rx_head_ = 0, rx_size_ = 0;
  std::size_t tx_head_ = 0, tx_size_ = 0;
  std::error_code failure_;
  bool owns_path_ = false, connected_ = false, input_closed_ = false;
  bool accepting_ = false, reading_ = false, writing_ = false;
  bool stopping_ = false, interrupted_ = false;

  console_transport() = default;

  fn schedule() -> void {
    if (!connected_) {
      if (!accepting_ && acceptor_.is_open()) {
        accepting_ = true;
        acceptor_.async_accept(socket_, [this](std::error_code error) noexcept {
          accepting_ = false;
          if (stopping_)
            return;
          if (error) {
            failure_ = error;
            return;
          }
          connected_ = true;
          error = acceptor_.close(error);
          if (error)
            failure_ = error;
        });
      }
      return;
    }
    if (!reading_ && !input_closed_ && rx_size_ == 0) {
      reading_ = true;
      let completed = [this](std::error_code error,
                              std::size_t count) noexcept {
        reading_ = false;
        if (stopping_)
          return;
        rx_head_ = 0;
        rx_size_ = count;
        if (interactive_) {
          const let escape = std::find(
              rx_.begin(), rx_.begin() + static_cast<std::ptrdiff_t>(count),
              std::byte{0x1d});
          if (escape != rx_.begin() + static_cast<std::ptrdiff_t>(count)) {
            rx_size_ = static_cast<std::size_t>(escape - rx_.begin());
            interrupted_ = true;
          }
        }
        if (error == asio::error::eof)
          input_closed_ = true;
        else if (error)
          failure_ = error;
      };
      if (terminal_mode_)
        input_.async_read_some(asio::buffer(rx_), std::move(completed));
      else
        socket_.async_read_some(asio::buffer(rx_), std::move(completed));
    }
    if (!writing_ && tx_size_ != 0) {
      writing_ = true;
      let completed = [this](std::error_code error,
                              std::size_t count) noexcept {
        writing_ = false;
        if (stopping_)
          return;
        tx_head_ += count;
        tx_size_ -= count;
        if (tx_size_ == 0)
          tx_head_ = 0;
        if (error)
          failure_ = error;
      };
      const let bytes = asio::buffer(tx_.data() + tx_head_, tx_size_);
      if (terminal_mode_)
        output_.async_write_some(bytes, std::move(completed));
      else
        socket_.async_write_some(bytes, std::move(completed));
    }
  }

  fn shutdown() noexcept -> std::error_code {
    if (stopping_)
      return {};
    stopping_ = true;
    std::error_code first, error;
    if (terminal_mode_) {
      if (interactive_ &&
          ::tcsetattr(input_.native_handle(), TCSANOW, &attributes_) < 0)
        first = {errno, std::generic_category()};
      if (input_flags_ >= 0 &&
          ::fcntl(input_.native_handle(), F_SETFL, input_flags_) < 0 && !first)
        first = {errno, std::generic_category()};
      if (output_flags_ >= 0 &&
          ::fcntl(output_.native_handle(), F_SETFL, output_flags_) < 0 &&
          !first)
        first = {errno, std::generic_category()};
      error = input_.close(error);
      if (error && !first)
        first = error;
      error = output_.close(error);
      if (error && !first)
        first = error;
    }
    error = acceptor_.close(error);
    if (error)
      first = error;
    error = socket_.close(error);
    if (error && !first)
      first = error;
    error = signals_.cancel(error);
    if (error && !first)
      first = error;
    error = signals_.clear(error);
    if (error && !first)
      first = error;
    try {
      context_.restart();
      context_.run();
    } catch (const std::bad_alloc &) {
      if (!first)
        first = std::make_error_code(std::errc::not_enough_memory);
    } catch (const std::system_error &status) {
      if (!first)
        first = status.code();
    }
    if (owns_path_) {
      struct stat current{};
      if (::lstat(path_.c_str(), &current) == 0) {
        if (current.st_dev == path_identity_.st_dev &&
            current.st_ino == path_identity_.st_ino) {
          std::filesystem::remove(path_, error);
          if (error && !first)
            first = error;
        }
      } else if (errno != ENOENT && !first) {
        first = {errno, std::generic_category()};
      }
      owns_path_ = false;
    }
    return first;
  }
};
} // namespace dynamips::host
