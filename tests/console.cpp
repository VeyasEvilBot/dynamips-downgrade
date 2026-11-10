#include <asio/buffer.hpp>
#include <asio/error.hpp>
#include <asio/io_context.hpp>
#include <asio/local/connect_pair.hpp>
#include <asio/local/stream_protocol.hpp>
#include <asio/socket_base.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <dynamips/macros.h>

import host.console;
import native_fd;

fn console_guest_echo(std::unique_ptr<dynamips::host::console_transport>,
                        dynamips::host::file_descriptor) -> bool;

namespace {
using transport = dynamips::host::console_transport;
fn check(bool condition, const char *message) -> bool {
  if (!condition)
    std::println(stderr, "console test: {}", message);
  return condition;
}
fn blocked(std::error_code error) -> bool {
  return error == asio::error::would_block || error == asio::error::try_again;
}

struct connection {
  asio::io_context context;
  asio::local::stream_protocol::socket peer{context};
  std::unique_ptr<transport> server;
  fn open() -> bool {
    asio::local::stream_protocol::socket local{context};
    std::error_code error;
    error = asio::local::connect_pair(local, peer, error);
    if (!check(!error, "create real Unix stream pair"))
      return false;
    error = local.set_option(asio::socket_base::send_buffer_size{1024}, error);
    if (!check(!error, "limit kernel TX buffering"))
      return false;
    error = peer.non_blocking(true, error);
    if (!check(!error, "set nonblocking client"))
      return false;
    const let descriptor = local.release(error);
    if (!check(!error, "transfer owned stream descriptor"))
      return false;
    let adopted =
        transport::adopt(dynamips::host::file_descriptor{descriptor});
    if (!check(adopted.has_value(), "adopt stream using shared FD ownership"))
      return false;
    server = std::move(*adopted);
    return true;
  }
};

fn receive_backpressure() -> bool {
  connection pair;
  if (!pair.open())
    return false;
  std::vector<std::byte> input(32768), output(input.size());
  for (std::size_t i = 0; i < input.size(); i++)
    input[i] = static_cast<std::byte>(i & 255);
  std::size_t sent = 0, received = 0;
  bool paused = false, half_closed = false;
  const let deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    std::error_code error;
    if (sent < input.size()) {
      sent += pair.peer.write_some(
          asio::buffer(input.data() + sent, input.size() - sent), error);
      if (!check(!error || blocked(error), "client sends binary input"))
        return false;
    } else if (!half_closed) {
      error = pair.peer.shutdown(asio::socket_base::shutdown_send, error);
      if (!check(!error, "half-close input"))
        return false;
      half_closed = true;
    }
    if (!check(pair.server->poll().has_value(), "poll incoming stream"))
      return false;
    const let bytes = pair.server->received();
    if (!paused && !bytes.empty()) {
      const std::vector<std::byte> snapshot(bytes.begin(), bytes.end());
      for (unsigned i = 0; i < 20; i++)
        if (!check(pair.server->poll().has_value(), "poll with staged RX held"))
          return false;
      if (!check(std::ranges::equal(snapshot, pair.server->received()),
                 "held RX is never overwritten"))
        return false;
      if (!check(!pair.server->consume_received(bytes.size() + 1),
                 "reject excessive RX acknowledgement"))
        return false;
      paused = true;
    }
    const let count = std::min<std::size_t>(13, bytes.size());
    if (!check(count <= output.size() - received, "bounded input delivery"))
      return false;
    std::copy_n(bytes.begin(), count,
                output.begin() + static_cast<std::ptrdiff_t>(received));
    received += count;
    if (!check(pair.server->consume_received(count).has_value(),
               "acknowledge partial RX"))
      return false;
    if (pair.server->input_closed() && pair.server->received().empty())
      break;
  }
  return check(paused && sent == input.size() && received == input.size() &&
                   input == output && pair.server->input_closed(),
               "RX staging, partial consumption and EOF preserve every byte") &&
         check(pair.server->close().has_value(), "close drained RX");
}

fn transmit_backpressure() -> bool {
  connection pair;
  if (!pair.open())
    return false;
  std::vector<std::byte> input(1024 * 1024), output(input.size());
  for (std::size_t i = 0; i < input.size(); i++)
    input[i] = static_cast<std::byte>((i * 7) & 255);
  std::size_t sent = 0, received = 0, stage_size = 0;
  bool stalled = false, partial = false;
  for (unsigned i = 0; i < 1000; i++) {
    const let count = pair.server->send(std::span{input}.subspan(sent));
    sent += count;
    if (count)
      stage_size = count;
    if (!check(pair.server->poll().has_value(), "poll blocked TX"))
      return false;
    const let pending = pair.server->pending_output();
    partial |= pending != 0 && pending < stage_size;
    if (count == 0 && pending != 0) {
      stalled = true;
      break;
    }
  }
  if (!check(stalled, "a non-reading peer applies bounded backpressure"))
    return false;
  const let deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (std::chrono::steady_clock::now() < deadline &&
         received < input.size()) {
    const let count = pair.server->send(std::span{input}.subspan(sent));
    sent += count;
    if (count)
      stage_size = count;
    if (!check(pair.server->poll().has_value(), "poll partial TX completions"))
      return false;
    const let pending = pair.server->pending_output();
    partial |= pending != 0 && pending < stage_size;
    std::error_code error;
    const let space = std::min<std::size_t>(97, output.size() - received);
    received += pair.peer.read_some(
        asio::buffer(output.data() + received, space), error);
    if (!check(!error || blocked(error), "slow client receives output"))
      return false;
  }
  for (unsigned i = 0; i < 10 && pair.server->pending_output() != 0; i++)
    if (!check(pair.server->poll().has_value(), "settle final TX completion"))
      return false;
  return check(partial, "exercise actual partial kernel writes") &&
         check(sent == input.size() && received == input.size() &&
                   input == output && pair.server->pending_output() == 0,
               "slow peer receives every TX byte in order") &&
         check(pair.server->close().has_value(), "close drained TX");
}

fn guest_echo() -> bool {
  connection pair;
  if (!pair.open())
    return false;
  std::error_code error;
  const let descriptor = pair.peer.release(error);
  if (!check(!error, "transfer client endpoint for guest integration"))
    return false;
  return console_guest_echo(std::move(pair.server),
                            dynamips::host::file_descriptor{descriptor});
}

fn disconnect_and_cancel() -> bool {
  for (unsigned i = 0; i < 32; i++) {
    connection pair;
    if (!pair.open())
      return false;
    std::array<std::byte, transport::capacity> bytes{};
    if (!check(pair.server->send(bytes) == bytes.size() &&
                   pair.server->poll().has_value(),
               "start pending read and write"))
      return false;
    if (!check(pair.server->close().has_value() &&
                   pair.server->close().has_value() && !pair.server->poll(),
               "shutdown settles callbacks and is idempotent"))
      return false;
  }
  connection pair;
  if (!pair.open())
    return false;
  std::error_code error;
  error = pair.peer.close(error);
  if (!check(!error, "disconnect peer"))
    return false;
  for (unsigned i = 0; i < 100 && !pair.server->input_closed(); i++)
    if (!check(pair.server->poll().has_value(), "observe peer EOF"))
      return false;
  std::array<std::byte, 1024> bytes{};
  if (!check(pair.server->input_closed() &&
                 pair.server->send(bytes) == bytes.size(),
             "retain output until write completion"))
    return false;
  bool failed = false;
  for (unsigned i = 0; i < 100 && !failed; i++)
    failed = !pair.server->poll();
  return check(failed && pair.server->pending_output() != 0,
               "write failure reports disconnected peer and retains unsent "
               "suffix") &&
         check(pair.server->close().has_value(), "close failed connection");
}

fn listener(const std::filesystem::path &directory) -> bool {
  const let path = (directory / "console.sock").string();
  {
    let server = transport::listen(path);
    if (!check(server && (*server)->poll().has_value(),
               "listen and start accept"))
      return false;
    if (!check(!transport::listen(path) && std::filesystem::exists(path),
               "do not replace another listener"))
      return false;
    asio::io_context context;
    asio::local::stream_protocol::socket peer{context};
    std::error_code error;
    error = peer.connect(asio::local::stream_protocol::endpoint{path}, error);
    if (!check(!error, "connect real filesystem Unix socket"))
      return false;
    if (!check((*server)->poll().has_value() && (*server)->connected(),
               "accept one client"))
      return false;
  }
  if (!check(!std::filesystem::exists(path),
             "destroy listener and remove owned path"))
    return false;
  {
    std::error_code error;
    const let original = std::filesystem::current_path(error);
    if (!check(!error, "save listener working directory"))
      return false;
    std::filesystem::current_path(directory, error);
    if (!check(!error, "enter relative listener directory"))
      return false;
    let relative = transport::listen("relative.sock");
    std::filesystem::current_path(original, error);
    if (!check(!error && relative,
               "restore working directory with relative listener alive"))
      return false;
    if (!check((*relative)->close().has_value() &&
                   !std::filesystem::exists(directory / "relative.sock"),
               "relative socket cleanup survives a working-directory change"))
      return false;
  }
  {
    let server = transport::listen(path);
    if (!check(server && (*server)->poll().has_value() &&
                   (*server)->close().has_value(),
               "cancel pending accept"))
      return false;
  }
  {
    std::ofstream existing{path};
    existing << "preserve";
  }
  if (!check(!transport::listen(path),
             "do not replace an existing regular file"))
    return false;
  std::ifstream existing{path};
  std::string content;
  existing >> content;
  if (!check(content == "preserve", "failed bind preserves existing contents"))
    return false;
  std::filesystem::remove(path);
  {
    let server = transport::listen(path);
    if (!check(server.has_value(), "listen before path replacement"))
      return false;
    std::filesystem::remove(path);
    std::ofstream replacement{path};
    replacement << "replacement";
    replacement.close();
    if (!check((*server)->close().has_value() && std::filesystem::exists(path),
               "cleanup preserves a replaced path"))
      return false;
  }
  return check(!transport::listen("") &&
                   !transport::listen(std::string(256, 'x')) &&
                   !transport::adopt(dynamips::host::file_descriptor{}),
               "invalid endpoints return typed failures");
}
} // namespace

fn main() -> int {
  try {
    char directory[] = "/tmp/dynamips-console-XXXXXX";
    if (!check(mkdtemp(directory) != nullptr,
               "create socket fixture directory"))
      return 1;
    const let passed = listener(directory) && receive_backpressure() &&
                       transmit_backpressure() && guest_echo() &&
                       disconnect_and_cancel();
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    if (!passed || !check(!error, "remove socket fixture directory"))
      return 1;
    std::println("console: real Unix streams, partial transfers, backpressure, "
                 "EOF and settled cancellation passed");
    return 0;
  } catch (const std::exception &error) {
    std::println(stderr, "console test: {}", error.what());
    return 1;
  }
}
