module;

#include "net_io_udp.h"
#include <dynamips/macros.h>
#include <expected>
#include <system_error>

#include <asio/executor_work_guard.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/udp.hpp>
#include <array>
#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "net_io.h"
#include "net.h"

export module net_io_udp;

import native_fd;

export namespace dynamips::nio {
fn listen_udp(netio_desc &endpoint, netio_udp_handler handler, void *arg1,
                void *arg2) noexcept -> std::expected<void, std::error_code>;
fn unlisten_udp(netio_desc &endpoint, bool all = false) noexcept
    -> std::expected<void, std::error_code>;
} // namespace dynamips::nio

namespace dynamips::nio {
namespace {
using result = std::expected<void, std::error_code>;

fn failure(std::errc value) -> std::unexpected<std::error_code> {
  return std::unexpected(std::make_error_code(value));
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

struct listener {
  asio::ip::udp::socket socket;
  asio::ip::udp::endpoint sender;
  std::array<unsigned char, NETIO_MAX_PKT_SIZE> buffer{};
  netio_desc *endpoint;
  netio_udp_handler handler;
  void *arg1, *arg2;
  unsigned int references = 1;
  bool active = true;

  listener(asio::io_context &context, netio_desc &nio,
           netio_udp_handler callback, void *first, void *second)
      : socket(context), endpoint(&nio), handler(callback), arg1(first),
        arg2(second) {}
};

struct receivers {
  asio::io_context context;
  std::optional<asio::executor_work_guard<asio::io_context::executor_type>>
      work;
  std::recursive_mutex mutex;
  std::map<netio_desc *, std::shared_ptr<listener>> listeners;
  std::thread worker;
  bool started = false;

  fn start() noexcept -> result;
  fn stop() noexcept -> result;
  fn arm(const std::shared_ptr<listener> &entry) noexcept -> result;
  fn add(netio_desc &nio, netio_udp_handler handler, void *a,
           void *b) noexcept -> result;
  fn remove(netio_desc &nio, bool all) noexcept -> result;

  fn cancel(listener &entry) -> void {
    entry.active = false;
    std::error_code error;
    entry.socket.close(error);
    if (error)
      std::fprintf(stderr, "UDP listener close: error %d\n", error.value());
  }

  ~receivers() {
    if (!stop())
      std::terminate();
  }
};

receivers state;

fn receivers::start() noexcept -> result {
  return attempt([&]() -> result {
    let lock = std::lock_guard{mutex};
    if (started)
      return {};
    context.restart();
    work.emplace(context.get_executor());
    try {
      worker = std::thread([this] {
        const let ran = attempt([&]() -> result {
          context.run();
          return {};
        });
        if (!ran)
          std::fprintf(stderr, "UDP receive worker: error %d\n",
                       ran.error().value());
      });
    } catch (...) {
      work.reset();
      throw;
    }
    started = true;
    return {};
  });
}

fn receivers::stop() noexcept -> result {
  return attempt([&]() -> result {
    {
      let lock = std::lock_guard{mutex};
      if (!started)
        return {};
      if (worker.get_id() == std::this_thread::get_id())
        return failure(std::errc::resource_deadlock_would_occur);
      for (let &[nio, entry] : listeners)
        cancel(*entry);
      listeners.clear();
      work.reset();
      context.stop();
    }
    worker.join();
    let lock = std::lock_guard{mutex};
    context.restart();
    context.poll();
    started = false;
    return {};
  });
}

fn receivers::arm(const std::shared_ptr<listener> &entry) noexcept -> result {
  return attempt([&]() -> result {
    entry->socket.async_receive_from(
        asio::buffer(entry->buffer), entry->sender,
        [this, entry](const std::error_code &error, std::size_t size) noexcept {
          const let dispatched = attempt([&]() -> result {
            let lock = std::lock_guard{mutex};
            if (!entry->active)
              return {};
            if (!error || error == asio::error::message_size) {
              for (unsigned int burst = 0; burst < 16; ++burst) {
                if (size) {
                  std::fill(entry->buffer.begin() +
                                static_cast<std::ptrdiff_t>(size),
                            entry->buffer.end(), 0);
                  if (netio_receive_dispatch(entry->endpoint,
                                             entry->buffer.data(),
                                             static_cast<ssize_t>(size)) > 0)
                    entry->handler(entry->endpoint, entry->buffer.data(),
                                   static_cast<ssize_t>(size), entry->arg1,
                                   entry->arg2);
                }
                if (!entry->active || burst == 15)
                  break;
                std::error_code received;
                size = entry->socket.receive_from(asio::buffer(entry->buffer),
                                                  entry->sender, 0, received);
                if (received)
                  break;
              }
            }
            if (entry->active)
              return arm(entry);
            return {};
          });
          if (!dispatched) {
            const let canceled = attempt([&]() -> result {
              let lock = std::lock_guard{mutex};
              cancel(*entry);
              return {};
            });
            std::fprintf(stderr, "UDP receive dispatch: error %d (cancel %d)\n",
                         dispatched.error().value(),
                         canceled ? 0 : canceled.error().value());
          }
        });
    return {};
  });
}

fn receivers::add(netio_desc &nio, netio_udp_handler handler, void *a,
                    void *b) noexcept -> result {
  return attempt([&]() -> result {
    let lock = std::lock_guard{mutex};
    if (!started || !handler ||
        (nio.type != NETIO_TYPE_UDP && nio.type != NETIO_TYPE_UDP_AUTO))
      return failure(std::errc::invalid_argument);
    if (let found = listeners.find(&nio); found != listeners.end()) {
      if (!found->second->active)
        return failure(std::errc::io_error);
      if (found->second->references == std::numeric_limits<unsigned int>::max())
        return failure(std::errc::value_too_large);
      ++found->second->references;
      return {};
    }
    auto entry = std::make_shared<listener>(context, nio, handler, a, b);
    sockaddr_storage address{};
    socklen_t length = sizeof(address);
    if (getsockname(nio.u.nid.fd, reinterpret_cast<sockaddr *>(&address),
                    &length) < 0)
      return std::unexpected(std::error_code{errno, std::generic_category()});
    let fd = host::duplicate_fd(nio.u.nid.fd);
    if (!fd)
      return std::unexpected(fd.error());
    std::error_code error;
    entry->socket.assign(address.ss_family == AF_INET6 ? asio::ip::udp::v6()
                                                       : asio::ip::udp::v4(),
                         fd->get(), error);
    if (error)
      return std::unexpected(error);
    fd->release();
    entry->socket.non_blocking(true, error);
    if (error)
      return std::unexpected(error);
    listeners.emplace(&nio, entry);
    if (let armed = arm(entry); !armed) {
      listeners.erase(&nio);
      return armed;
    }
    return {};
  });
}

fn receivers::remove(netio_desc &nio, bool all) noexcept -> result {
  return attempt([&]() -> result {
    let lock = std::lock_guard{mutex};
    let found = listeners.find(&nio);
    if (found == listeners.end())
      return {};
    let entry = found->second;
    if (!all && --entry->references)
      return {};
    cancel(*entry);
    listeners.erase(found);
    return {};
  });
}
} // namespace

fn listen_udp(netio_desc &endpoint, netio_udp_handler handler, void *a,
                void *b) noexcept -> result {
  return state.add(endpoint, handler, a, b);
}

fn unlisten_udp(netio_desc &endpoint, bool all) noexcept -> result {
  return state.remove(endpoint, all);
}
} // namespace dynamips::nio

cfn netio_udp_rxl_init() -> int {
  return dynamips::nio::state.start() ? 0 : -1;
}

cfn netio_udp_rxl_shutdown() -> void {
  if (let stopped = dynamips::nio::state.stop(); !stopped)
    std::fprintf(stderr, "UDP listener shutdown: error %d\n",
                 stopped.error().value());
}

cfn netio_udp_rxl_add(netio_desc *nio, netio_udp_handler handler, void *a,
                      void *b) -> int {
  return nio && dynamips::nio::listen_udp(*nio, handler, a, b) ? 0 : -1;
}

cfn netio_udp_rxl_remove(netio_desc *nio, int all) -> int {
  return nio && dynamips::nio::unlisten_udp(*nio, all != 0) ? 0 : -1;
}

cfn netio_udp_rxl_connect(netio_desc *nio, char *host, int port) -> int {
  if (!nio || !host)
    return -1;
  const let connected = dynamips::nio::attempt([&]() -> dynamips::nio::result {
    let lock = std::lock_guard{dynamips::nio::state.mutex};
    if (ip_connect_fd(nio->u.nid.fd, host, port) < 0)
      return std::unexpected(std::error_code{errno, std::generic_category()});
    return {};
  });
  return connected ? 0 : -1;
}
