module;

#include "vtty_io.h"
#include <dynamips/macros.h>
#include <expected>
#include <span>
#include <system_error>

#include <asio/io_context.hpp>
#include <asio/executor_work_guard.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/posix/stream_descriptor.hpp>
#include <asio/write.hpp>
#include <asio/post.hpp>
#include <array>
#include <chrono>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <vector>

export module vtty_io;

import native_fd;

export namespace dynamips::console {
enum class transport { none, terminal, tcp, serial };
fn attach(virtual_tty &owner, transport type,
            std::span<const int> descriptors) noexcept
    -> std::expected<void, std::error_code>;
fn detach(virtual_tty &owner) noexcept
    -> std::expected<void, std::error_code>;
fn write(virtual_tty &owner, std::span<const unsigned char> bytes) noexcept
    -> std::expected<void, std::error_code>;
} // namespace dynamips::console

namespace dynamips::console {
namespace {
using result = std::expected<void, std::error_code>;
using bytes = std::span<const unsigned char>;
constexpr std::size_t queue_limit = 1024 * 1024;

fn failure(std::errc error) -> std::unexpected<std::error_code> {
  return std::unexpected(std::make_error_code(error));
}

template <typename Operation>
fn attempt(Operation operation) noexcept -> result {
  try {
    return operation();
  } catch (const std::bad_alloc &) {
    return failure(std::errc::not_enough_memory);
  } catch (const std::system_error &error) {
    return std::unexpected(error.code());
  } catch (...) {
    return failure(std::errc::io_error);
  }
}
struct endpoint;

struct channel {
  std::variant<asio::ip::tcp::socket, asio::posix::stream_descriptor> stream;
  std::weak_ptr<endpoint> owner;
  std::array<unsigned char, 256> input{};
  std::deque<std::shared_ptr<std::vector<unsigned char>>> output;
  std::size_t pending = 0;
  int fd = -1;
  int original_flags = -1;
  bool active = true;
  bool ready = false;
  bool read_closed = false;

  channel(asio::io_context &context, bool tcp)
      : stream(tcp ? decltype(stream){std::in_place_type<asio::ip::tcp::socket>,
                                      context}
                   : decltype(stream){
                         std::in_place_type<asio::posix::stream_descriptor>,
                         context}) {}

  ~channel() {
    if (original_flags >= 0) {
      const int descriptor =
          std::get<asio::posix::stream_descriptor>(stream).native_handle();
      if (fcntl(descriptor, F_SETFL, original_flags) < 0)
        std::fprintf(stderr, "console restore FD flags: error %d\n", errno);
    }
  }
};

struct endpoint {
  virtual_tty *native;
  bool active = true;
  bool closing = false;
  std::vector<std::shared_ptr<asio::ip::tcp::acceptor>> listeners;
  std::map<int, std::shared_ptr<channel>> channels;

  explicit endpoint(virtual_tty &owner) : native(&owner) {}
};

struct input {
  std::shared_ptr<endpoint> owner;
  std::shared_ptr<channel> source;
  bool connected;
  std::size_t size;
};

struct terminals {
  asio::io_context context;
  std::optional<asio::executor_work_guard<asio::io_context::executor_type>>
      work;
  std::mutex mutex;
  std::map<virtual_tty *, std::shared_ptr<endpoint>> endpoints;
  std::map<int, std::weak_ptr<channel>> descriptors;
  std::deque<std::unique_ptr<input>> events;
  bool started = false;

  fn start() noexcept -> result;
  fn stop() noexcept -> result;
  fn add(virtual_tty &, transport, std::span<const int>) noexcept -> result;
  fn remove(virtual_tty &) noexcept -> result;
  fn accept(const std::shared_ptr<endpoint> &,
              const std::shared_ptr<asio::ip::tcp::acceptor> &) noexcept
      -> result;
  fn read(const std::shared_ptr<channel> &) noexcept -> result;
  fn send(const std::shared_ptr<channel> &) noexcept -> result;
  fn enqueue(const std::shared_ptr<channel> &, bytes) noexcept -> result;

  fn cancel(std::shared_ptr<channel> client) -> void {
    client->active = false;
    std::error_code error;
    std::visit([&](auto &stream) { stream.cancel(error); }, client->stream);
    if (error)
      std::fprintf(stderr, "console cancel FD %d: error %d\n", client->fd,
                   error.value());
    if (auto *socket = std::get_if<asio::ip::tcp::socket>(&client->stream))
      socket->shutdown(asio::ip::tcp::socket::shutdown_both, error);
    descriptors.erase(client->fd);
    if (let owner = client->owner.lock())
      owner->channels.erase(client->fd);
    client->output.clear();
    client->pending = 0;
  }

  fn cancel(const std::shared_ptr<endpoint> &owner) -> void {
    owner->active = false;
    owner->closing = true;
    for (let &listener : owner->listeners) {
      std::error_code error;
      listener->close(error);
      if (error)
        std::fprintf(stderr, "console listener close: error %d\n",
                     error.value());
    }
    while (!owner->channels.empty())
      cancel(owner->channels.begin()->second);
  }

  fn failed(const std::shared_ptr<channel> &client,
              std::error_code error) noexcept -> void {
    const let cleaned = attempt([&]() -> result {
      let lock = std::lock_guard{mutex};
      cancel(client);
      return {};
    });
    std::fprintf(stderr, "console FD %d: error %d (cleanup %d)\n", client->fd,
                 error.value(), cleaned ? 0 : cleaned.error().value());
  }

  ~terminals() {
    if (!stop())
      std::terminate();
  }
};

terminals state;

fn terminals::start() noexcept -> result {
  return attempt([&]() -> result {
    let lock = std::lock_guard{mutex};
    if (started)
      return {};
    context.restart();
    work.emplace(context.get_executor());
    started = true;
    return {};
  });
}

fn terminals::stop() noexcept -> result {
  return attempt([&]() -> result {
    let lock = std::lock_guard{mutex};
    if (!started)
      return {};
    for (let &[native, owner] : endpoints)
      cancel(owner);
    endpoints.clear();
    events.clear();
    started = false;
    work.reset();
    context.stop();
    return {};
  });
}

fn terminals::accept(
    const std::shared_ptr<endpoint> &owner,
    const std::shared_ptr<asio::ip::tcp::acceptor> &listener) noexcept
    -> result {
  return attempt([&]() -> result {
    let client = std::make_shared<channel>(context, true);
    client->owner = owner;
    listener->async_accept(
        std::get<asio::ip::tcp::socket>(client->stream),
        [this, owner, listener, client](const std::error_code &error) noexcept {
          const let handled = attempt([&]() -> result {
            let lock = std::lock_guard{mutex};
            if (!owner->active || owner->closing)
              return {};
            if (!error) {
              client->fd = std::get<asio::ip::tcp::socket>(client->stream)
                               .native_handle();
              owner->channels.emplace(client->fd, client);
              descriptors[client->fd] = client;
              events.push_back(
                  std::make_unique<input>(input{owner, client, true, 0}));
              asio::post(context, [] {});
            }
            return accept(owner, listener);
          });
          if (!handled)
            failed(client, handled.error());
        });
    return {};
  });
}

fn terminals::read(const std::shared_ptr<channel> &client) noexcept
    -> result {
  return attempt([&]() -> result {
    std::visit(
        [&](auto &stream) {
          stream.async_read_some(asio::buffer(client->input),
                                 [this, client](const std::error_code &error,
                                                std::size_t size) noexcept {
                                   const let handled = attempt([&]() -> result {
                                     let lock = std::lock_guard{mutex};
                                     let owner = client->owner.lock();
                                     if (!client->active || !owner ||
                                         !owner->active || owner->closing)
                                       return {};
                                     if (error == asio::error::eof) {
                                       client->read_closed = true;
                                       client->ready = false;
                                       if (client->output.empty())
                                         cancel(client);
                                       return {};
                                     }
                                     if (error || !size) {
                                       cancel(client);
                                       return {};
                                     }
                                     events.push_back(std::make_unique<input>(
                                         input{owner, client, false, size}));
                                     asio::post(context, [] {});
                                     return {};
                                   });
                                   if (!handled)
                                     failed(client, handled.error());
                                 });
        },
        client->stream);
    return {};
  });
}

fn terminals::send(const std::shared_ptr<channel> &client) noexcept
    -> result {
  return attempt([&]() -> result {
    let message = client->output.front();
    std::visit(
        [&](auto &stream) {
          asio::async_write(
              stream, asio::buffer(*message),
              [this, client, message](const std::error_code &error,
                                      std::size_t) noexcept {
                const let handled = attempt([&]() -> result {
                  let lock = std::lock_guard{mutex};
                  if (!client->active)
                    return {};
                  if (error) {
                    cancel(client);
                    return {};
                  }
                  client->pending -= message->size();
                  client->output.pop_front();
                  if (!client->output.empty())
                    return send(client);
                  if (client->read_closed)
                    cancel(client);
                  return {};
                });
                if (!handled)
                  failed(client, handled.error());
              });
        },
        client->stream);
    return {};
  });
}

fn terminals::enqueue(const std::shared_ptr<channel> &client,
                        bytes data) noexcept -> result {
  const let queued = attempt([&]() -> result {
    if (!client->active)
      return failure(std::errc::broken_pipe);
    if (data.size() > queue_limit - client->pending)
      return failure(std::errc::no_buffer_space);
    const bool idle = client->output.empty();
    while (!data.empty()) {
      if (client->output.size() < 2 || client->output.back()->size() == 4096) {
        auto buffer = std::make_shared<std::vector<unsigned char>>();
        buffer->reserve(4096);
        client->output.push_back(std::move(buffer));
      }
      let &buffer = *client->output.back();
      const let size = std::min(data.size(), std::size_t{4096} - buffer.size());
      buffer.insert(buffer.end(), data.begin(),
                    data.begin() + static_cast<std::ptrdiff_t>(size));
      client->pending += size;
      data = data.subspan(size);
    }
    if (idle && !client->output.empty())
      return send(client);
    return {};
  });
  if (!queued)
    cancel(client);
  return queued;
}

fn terminals::add(virtual_tty &native, transport type,
                    std::span<const int> fds) noexcept -> result {
  return attempt([&]() -> result {
    let lock = std::lock_guard{mutex};
    if (!started || endpoints.contains(&native))
      return failure(std::errc::invalid_argument);
    let owner = std::make_shared<endpoint>(native);
    endpoints.emplace(&native, owner);
    const let configured = attempt([&]() -> result {
      for (int fd : fds) {
        if (fd < 0)
          continue;
        let duplicate = host::duplicate_fd(fd);
        if (!duplicate)
          return std::unexpected(duplicate.error());
        std::error_code error;
        if (type == transport::tcp) {
          sockaddr_storage address{};
          socklen_t length = sizeof(address);
          if (getsockname(fd, reinterpret_cast<sockaddr *>(&address),
                          &length) != 0)
            return std::unexpected(
                std::error_code{errno, std::generic_category()});
          let listener = std::make_shared<asio::ip::tcp::acceptor>(context);
          listener->assign(address.ss_family == AF_INET6 ? asio::ip::tcp::v6()
                                                         : asio::ip::tcp::v4(),
                           duplicate->get(), error);
          if (error)
            return std::unexpected(error);
          duplicate->release();
          owner->listeners.push_back(listener);
          if (let armed = accept(owner, listener); !armed)
            return armed;
        } else {
          let client = std::make_shared<channel>(context, false);
          client->owner = owner;
          client->fd = fd;
          client->ready = true;
          const int flags = fcntl(duplicate->get(), F_GETFL, 0);
          if (flags < 0)
            return std::unexpected(
                std::error_code{errno, std::generic_category()});
          std::get<asio::posix::stream_descriptor>(client->stream)
              .assign(duplicate->get(), error);
          if (error)
            return std::unexpected(error);
          duplicate->release();
          client->original_flags = flags;
          owner->channels.emplace(fd, client);
          descriptors[fd] = client;
          if (let armed = read(client); !armed)
            return armed;
        }
      }
      return {};
    });
    if (!configured) {
      cancel(owner);
      endpoints.erase(&native);
    }
    return configured;
  });
}

fn terminals::remove(virtual_tty &native) noexcept -> result {
  return attempt([&]() -> result {
    let lock = std::unique_lock{mutex};
    let found = endpoints.find(&native);
    if (found == endpoints.end())
      return {};
    let owner = found->second;
    owner->closing = true;
    const let deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds{250};
    const let pending = [&] {
      for (const let &[fd, client] : owner->channels)
        if (client->pending)
          return true;
      return false;
    };
    while (pending() && std::chrono::steady_clock::now() < deadline) {
      lock.unlock();
      context.run_one_for(std::chrono::milliseconds{1});
      lock.lock();
    }
    const bool drained = !pending();
    cancel(owner);
    endpoints.erase(&native);
    if (!drained)
      return failure(std::errc::timed_out);
    return {};
  });
}
} // namespace

fn attach(virtual_tty &owner, transport type,
            std::span<const int> descriptors) noexcept -> result {
  return state.add(owner, type, descriptors);
}

fn detach(virtual_tty &owner) noexcept -> result {
  return state.remove(owner);
}

fn write(virtual_tty &native, bytes data) noexcept -> result {
  return attempt([&]() -> result {
    let lock = std::unique_lock{state.mutex};
    let found = state.endpoints.find(&native);
    if (found == state.endpoints.end() || found->second->closing)
      return failure(std::errc::broken_pipe);
    let owner = found->second;
    bool pressure = data.size() > queue_limit;
    for (const let &[fd, client] : owner->channels)
      if (client->ready && data.size() > queue_limit - client->pending)
        pressure = true;
    if (!pressure) {
      for (let it = owner->channels.begin(); it != owner->channels.end();) {
        let client = (it++)->second;
        if (client->ready)
          if (let queued = state.enqueue(client, data); !queued)
            std::fprintf(stderr, "console output FD %d: error %d\n", client->fd,
                         queued.error().value());
      }
      return {};
    }
    std::vector<std::shared_ptr<channel>> clients;
    for (const let &[fd, client] : owner->channels)
      if (client->ready)
        clients.push_back(client);
    for (let &client : clients) {
      let remaining = data;
      while (!remaining.empty() && client->active && !owner->closing) {
        const let size = std::min(remaining.size(), std::size_t{4096});
        const let deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds{250};
        while (client->active && !owner->closing &&
               size > queue_limit - client->pending &&
               std::chrono::steady_clock::now() < deadline) {
          lock.unlock();
          state.context.run_one_for(std::chrono::milliseconds{1});
          lock.lock();
        }
        const bool writable = !client->active || owner->closing ||
                              size <= queue_limit - client->pending;
        if (!writable) {
          state.cancel(client);
          std::fprintf(stderr, "console FD %d: output backpressure timeout\n",
                       client->fd);
          break;
        }
        if (!client->active || owner->closing)
          break;
        if (let queued = state.enqueue(client, remaining.first(size));
            !queued) {
          std::fprintf(stderr, "console output FD %d: error %d\n", client->fd,
                       queued.error().value());
          break;
        }
        remaining = remaining.subspan(size);
      }
    }
    return {};
  });
}
} // namespace dynamips::console

cfn vtty_io_start() -> int { return dynamips::console::state.start() ? 0 : -1; }

cfn vtty_io_stop() -> void {
  if (let stopped = dynamips::console::state.stop(); !stopped)
    std::fprintf(stderr, "console shutdown: error %d\n",
                 stopped.error().value());
}

cfn vtty_io_attach(virtual_tty *owner, int type, const int *fds, size_t count)
    -> int {
  if (!owner || type < 0 || type > 3)
    return -1;
  return dynamips::console::attach(
             *owner, static_cast<dynamips::console::transport>(type),
             {fds, count})
             ? 0
             : -1;
}

cfn vtty_io_detach(virtual_tty *owner) -> int {
  return owner && dynamips::console::detach(*owner) ? 0 : -1;
}

cfn vtty_io_write(virtual_tty *owner, const void *data, size_t size)
    -> ssize_t {
  return owner && dynamips::console::write(
                      *owner, {static_cast<const unsigned char *>(data), size})
             ? static_cast<ssize_t>(size)
             : -1;
}

cfn vtty_io_write_fd(int fd, const void *data, size_t size) -> ssize_t {
  using namespace dynamips::console;
  const let queued = attempt([&]() -> result {
    let lock = std::lock_guard{state.mutex};
    let found = state.descriptors.find(fd);
    if (found == state.descriptors.end())
      return failure(std::errc::bad_file_descriptor);
    let client = found->second.lock();
    if (!client)
      return failure(std::errc::bad_file_descriptor);
    return state.enqueue(client,
                         {static_cast<const unsigned char *>(data), size});
  });
  return queued ? static_cast<ssize_t>(size) : -1;
}

cfn vtty_io_take(vtty_io_event *out) -> int {
  using namespace dynamips::console;
  if (!out)
    return -1;
  int status = 0;
  const let taken = attempt([&]() -> result {
    const let deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds{50};
    for (;;) {
      {
        let lock = std::lock_guard{state.mutex};
        if (!state.started) {
          status = -1;
          return {};
        }
        if (!state.events.empty()) {
          let event = std::move(state.events.front());
          state.events.pop_front();
          *out = {event->owner->native, event->source->fd,
                  event->connected,     event->source->input.data(),
                  event->size,          event.get()};
          event.release();
          status = 1;
          return {};
        }
      }
      if (!state.context.run_one_until(deadline))
        return {};
    }
  });
  return taken ? status : -1;
}

cfn vtty_io_drain() -> void {
  using namespace dynamips::console;
  const let drained = attempt([&]() -> result {
    state.context.restart();
    state.context.poll();
    return {};
  });
  if (!drained)
    std::fprintf(stderr, "console cancellation drain: error %d\n",
                 drained.error().value());
}

cfn vtty_io_active(const vtty_io_event *event) -> int {
  using namespace dynamips::console;
  bool active = false;
  const let checked = attempt([&]() -> result {
    let lock = std::lock_guard{state.mutex};
    let *input = static_cast<struct input *>(event->handle);
    active = input && input->owner->active && !input->owner->closing &&
             input->source->active;
    return {};
  });
  return checked && active;
}

cfn vtty_io_complete(vtty_io_event *event) -> void {
  using namespace dynamips::console;
  std::unique_ptr<input> input{static_cast<struct input *>(event->handle)};
  event->handle = nullptr;
  if (!input)
    return;
  const let completed = attempt([&]() -> result {
    let lock = std::lock_guard{state.mutex};
    let client = input->source;
    if (!input->owner->active || input->owner->closing || !client->active)
      return {};
    if (input->connected)
      client->ready = true;
    if (client->read_closed)
      return {};
    if (let armed = state.read(client); !armed) {
      state.cancel(client);
      return armed;
    }
    return {};
  });
  if (!completed)
    std::fprintf(stderr, "console input completion: error %d\n",
                 completed.error().value());
}
