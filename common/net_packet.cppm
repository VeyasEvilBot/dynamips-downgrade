module;

#include "net.h"
#include <dynamips/macros.h>
#include <array>
#include <expected>
#include <span>

#include "crc.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <cstdint>
#include <algorithm>

export module net_packet;

export namespace dynamips::packet {
enum class error { truncated, invalid_header, invalid_address, out_of_memory };
using bytes = std::span<const unsigned char>;
fn checksum(bytes data, unsigned int seed = 0) noexcept -> unsigned short;
fn analyze(std::span<unsigned char> frame) noexcept
    -> std::expected<n_pkt_ctx_t, error>;
fn ipv4_prefix(const char *text) noexcept
    -> std::expected<n_ip_network_t, error>;
fn ipv6_prefix(const char *text) noexcept
    -> std::expected<n_ipv6_network_t, error>;
fn ipv4(const char *text) noexcept -> std::expected<n_ip_addr_t, error>;
fn ipv6(const char *text) noexcept -> std::expected<n_ipv6_addr_t, error>;
fn mac(const char *text) noexcept -> std::expected<n_eth_addr_t, error>;
fn board_id(const char *text, int encoding) noexcept
    -> std::expected<std::array<unsigned char, 11>, error>;
} // namespace dynamips::packet

extern "C" n_ip_addr_t ip_masks[N_IP_ADDR_BITS + 1];

/* Convert an IPv4 address into a string */
cfn n_ip_ntoa(char *buffer, n_ip_addr_t ip_addr) -> char * {
  u_char *p = (u_char *)&ip_addr;
  sprintf(buffer, "%u.%u.%u.%u", p[0], p[1], p[2], p[3]);
  return (buffer);
}

#if HAS_RFC2553
/* Convert in IPv6 address into a string */
cfn n_ipv6_ntoa(char *buffer, n_ipv6_addr_t *ipv6_addr) -> char * {
  return ((char *)inet_ntop(AF_INET6, ipv6_addr, buffer, INET6_ADDRSTRLEN));
}
#endif

/* Convert a string containing an IP address in binary */
cfn n_ip_aton(n_ip_addr_t *ip_addr, char *ip_str) -> int {
  struct in_addr addr;

  if (inet_aton(ip_str, &addr) == 0)
    return (-1);

  *ip_addr = ntohl(addr.s_addr);
  return (0);
}

#if HAS_RFC2553
/* Convert an IPv6 address from string into binary */
cfn n_ipv6_aton(n_ipv6_addr_t *ipv6_addr, char *ip_str) -> int {
  return (inet_pton(AF_INET6, ip_str, ipv6_addr));
}
#endif

namespace {
struct free_text {
  fn operator()(char *p) const noexcept -> void { std::free(p); }
};

using text_copy = std::unique_ptr<char, free_text>;

fn cidr(const char *text, unsigned long maximum) noexcept
    -> std::expected<std::pair<text_copy, unsigned long>,
                     dynamips::packet::error> {
  using dynamips::packet::error;
  if (!text)
    return std::unexpected(error::invalid_address);
  const let *slash = std::strchr(text, '/');
  if (!slash)
    return std::unexpected(error::invalid_address);
  char *end = nullptr;
  const let bits = std::strtoul(slash + 1, &end, 0);
  if (*end || bits > maximum)
    return std::unexpected(error::invalid_address);
  text_copy copy{strdup(text)};
  if (!copy)
    return std::unexpected(error::out_of_memory);
  copy.get()[slash - text] = '\0';
  return std::pair{std::move(copy), bits};
}
} // namespace

cfn ip_parse_cidr(char *token, n_ip_addr_t *address, n_ip_addr_t *mask) -> int {
  let parsed = dynamips::packet::ipv4_prefix(token);
  if (!parsed)
    return -1;
  *address = parsed->net_addr;
  *mask = parsed->net_mask;
  return 0;
}

cfn ipv6_parse_cidr(char *token, n_ipv6_addr_t *address, u_int *mask) -> int {
  let parsed = dynamips::packet::ipv6_prefix(token);
  if (!parsed)
    return -1;
  *address = parsed->net_addr;
  *mask = parsed->net_mask;
  return 0;
}

/* Parse a processor board id and return the eeprom settings in a buffer */
cfn parse_board_id(m_uint8_t *buf, const char *id, int encode) -> int {
  memset(buf, 0, 11);
  if (encode == 4) {
    char *end = nullptr;
    const let parsed = std::strtoll(id, &end, 10);
    if (end == id)
      return -1;
    const let value = static_cast<std::uint32_t>(parsed);
    for (unsigned int i = 0; i < 4; ++i)
      buf[i] = static_cast<unsigned char>(value >> ((3 - i) * 8));
    return (0);
  } else if (encode == 9) {
    int res;
    unsigned short a = 0, b = 0;
    res = sscanf(id, "%c%c%c%2hx%2hx%c%c%c%c", &buf[0], &buf[1], &buf[2], &a,
                 &b, &buf[5], &buf[6], &buf[7], &buf[8]);
    buf[3] = static_cast<m_uint8_t>(a);
    buf[4] = static_cast<m_uint8_t>(b);
    if (res != 9)
      return (-1);
    return (0);
  } else if (encode == 11) {
    int res;

    res =
        sscanf(id, "%c%c%c%c%c%c%c%c%c%c%c", &buf[0], &buf[1], &buf[2], &buf[3],
               &buf[4], &buf[5], &buf[6], &buf[7], &buf[8], &buf[9], &buf[10]);
    if (res != 11)
      return (-1);
    return (0);
  }
  return (-1);
}

/* Parse a MAC address */
cfn parse_mac_addr(n_eth_addr_t *addr, char *str) -> int {
  u_int v[N_ETH_ALEN];
  int i, res;

  /* First try, standard format (00:01:02:03:04:05) */
  res = sscanf(str, "%x:%x:%x:%x:%x:%x", &v[0], &v[1], &v[2], &v[3], &v[4],
               &v[5]);

  if (res == 6) {
    for (i = 0; i < N_ETH_ALEN; i++)
      addr->eth_addr_byte[i] = static_cast<m_uint8_t>(v[i]);
    return (0);
  }

  /* Second try, Cisco format (0001.0002.0003) */
  res = sscanf(str, "%x.%x.%x", &v[0], &v[1], &v[2]);

  if (res == 3) {
    addr->eth_addr_byte[0] = (v[0] >> 8) & 0xFF;
    addr->eth_addr_byte[1] = v[0] & 0xFF;
    addr->eth_addr_byte[2] = (v[1] >> 8) & 0xFF;
    addr->eth_addr_byte[3] = v[1] & 0xFF;
    addr->eth_addr_byte[4] = (v[2] >> 8) & 0xFF;
    addr->eth_addr_byte[5] = v[2] & 0xFF;
    return (0);
  }

  return (-1);
}

/* Convert an Ethernet address into a string */
cfn n_eth_ntoa(char *buffer, n_eth_addr_t *addr, int format) -> char * {
  const char *str_format;

  if (format == 0) {
    str_format = "%2.2x:%2.2x:%2.2x:%2.2x:%2.2x:%2.2x";
  } else {
    str_format = "%2.2x%2.2x.%2.2x%2.2x.%2.2x%2.2x";
  }

  sprintf(buffer, str_format, addr->eth_addr_byte[0], addr->eth_addr_byte[1],
          addr->eth_addr_byte[2], addr->eth_addr_byte[3],
          addr->eth_addr_byte[4], addr->eth_addr_byte[5]);
  return (buffer);
}

namespace dynamips::packet {
fn ipv4_prefix(const char *text) noexcept
    -> std::expected<n_ip_network_t, error> {
  let split = cidr(text, 32);
  if (!split)
    return std::unexpected(split.error());
  let address = ipv4(split->first.get());
  if (!address)
    return std::unexpected(address.error());
  return n_ip_network_t{*address, ip_masks[split->second]};
}

fn ipv6_prefix(const char *text) noexcept
    -> std::expected<n_ipv6_network_t, error> {
  let split = cidr(text, 128);
  if (!split)
    return std::unexpected(split.error());
  let address = ipv6(split->first.get());
  if (!address)
    return std::unexpected(address.error());
  return n_ipv6_network_t{*address, static_cast<unsigned int>(split->second)};
}

namespace {
fn word(const unsigned char *p) noexcept -> unsigned short {
  return static_cast<unsigned short>((p[0] << 8) | p[1]);
}

fn sum_bytes(bytes data) noexcept -> unsigned int {
  unsigned int sum = 0;
  const let pairs = data.size() / 2;
  for (std::size_t i = 0; i < pairs; ++i)
    sum += (static_cast<unsigned int>(data[i * 2]) << 8) | data[i * 2 + 1];
  if (data.size() & 1)
    sum += static_cast<unsigned int>(data.back()) << 8;
  return sum;
}

fn folded(unsigned int sum) noexcept -> unsigned short {
  sum = (sum & 0xffff) + (sum >> 16);
  sum = (sum & 0xffff) + (sum >> 16);
  return static_cast<unsigned short>(~sum);
}
} // namespace

fn checksum(bytes data, unsigned int seed) noexcept -> unsigned short {
  seed = (seed & 0xffff) + (seed >> 16);
  while (!data.empty()) {
    const let size = std::min(data.size(), std::size_t{65536});
    seed += sum_bytes(data.first(size));
    seed = (seed & 0xffff) + (seed >> 16);
    data = data.subspan(size);
  }
  return folded(seed);
}

static fn analyze_into(n_pkt_ctx_t &ctx,
                        std::span<unsigned char> frame) noexcept
    -> std::expected<void, error> {
  ctx = {};
  ctx.pkt = frame.data();
  ctx.pkt_len = frame.size();
  if (frame.size() < N_ETH_HLEN)
    return std::unexpected(error::truncated);
  let type = word(frame.data() + 12);
  std::size_t offset = N_ETH_HLEN;
  if (type == N_ETH_PROTO_DOT1Q) {
    if (frame.size() < offset + 4)
      return std::unexpected(error::truncated);
    ctx.flags |= N_PKT_CTX_FLAG_VLAN;
    ctx.vlan_id = word(frame.data() + offset);
    type = word(frame.data() + offset + 2);
    offset += 4;
  }
  if (type < N_ETH_MTU)
    return {};
  ctx.flags |= N_PKT_CTX_FLAG_ETHV2;
  let payload = frame.subspan(offset);
  if (type == N_ETH_PROTO_ARP) {
    if (payload.size() < sizeof(n_arp_hdr_t))
      return std::unexpected(error::truncated);
    ctx.flags |= N_PKT_CTX_FLAG_L3_ARP;
    ctx.l3 = payload.data();
  } else if (type == N_ETH_PROTO_IP) {
    if (payload.size() < sizeof(n_ip_hdr_t))
      return std::unexpected(error::truncated);
    const std::size_t hlen = (payload[0] & 15) * 4;
    const std::size_t total = word(payload.data() + 2);
    ctx.flags |= N_PKT_CTX_FLAG_L3_IP;
    ctx.l3 = payload.data();
    if ((payload[0] & 0xf0) != 0x40 || hlen < 20 || hlen > total)
      return {};
    if (total > payload.size())
      return std::unexpected(error::truncated);
    if (folded(sum_bytes(payload.first(hlen))) != 0)
      return {};
    const let frag = word(payload.data() + 6);
    if ((frag & N_IP_OFFMASK) || (frag & N_IP_FLAG_MF))
      ctx.flags |= N_PKT_CTX_FLAG_IP_FRAG;
    ctx.ip_l4_proto = payload[9];
    if (!(ctx.flags & N_PKT_CTX_FLAG_IP_FRAG)) {
      std::size_t minimum = 0;
      if (ctx.ip_l4_proto == N_IP_PROTO_TCP)
        minimum = sizeof(n_tcp_hdr_t);
      else if (ctx.ip_l4_proto == N_IP_PROTO_UDP)
        minimum = sizeof(n_udp_hdr_t);
      if (total - hlen < minimum)
        return std::unexpected(error::truncated);
    }
    ctx.flags |= N_PKT_CTX_FLAG_IPH_OK;
    ctx.l4 = payload.data() + hlen;
  }
  return {};
}

fn analyze(std::span<unsigned char> frame) noexcept
    -> std::expected<n_pkt_ctx_t, error> {
  n_pkt_ctx_t ctx;
  if (let parsed = analyze_into(ctx, frame); !parsed)
    return std::unexpected(parsed.error());
  return ctx;
}

fn ipv4(const char *text) noexcept -> std::expected<n_ip_addr_t, error> {
  n_ip_addr_t value;
  if (!text || n_ip_aton(&value, const_cast<char *>(text)) != 0)
    return std::unexpected(error::invalid_address);
  return value;
}

fn ipv6(const char *text) noexcept -> std::expected<n_ipv6_addr_t, error> {
  n_ipv6_addr_t value;
  if (!text || n_ipv6_aton(&value, const_cast<char *>(text)) != 1)
    return std::unexpected(error::invalid_address);
  return value;
}

fn mac(const char *text) noexcept -> std::expected<n_eth_addr_t, error> {
  n_eth_addr_t value;
  if (!text || parse_mac_addr(&value, const_cast<char *>(text)) != 0)
    return std::unexpected(error::invalid_address);
  return value;
}

fn board_id(const char *text, int encoding) noexcept
    -> std::expected<std::array<unsigned char, 11>, error> {
  std::array<unsigned char, 11> value{};
  if (!text || parse_board_id(value.data(), text, encoding) != 0)
    return std::unexpected(error::invalid_address);
  return value;
}
} // namespace dynamips::packet

cfn ip_verify_cksum(n_ip_hdr_t *hdr) -> int {
  let *p = reinterpret_cast<unsigned char *>(hdr);
  return dynamips::packet::folded(dynamips::packet::sum_bytes(
             {p, static_cast<std::size_t>(p[0] & 15) * 4})) == 0;
}

cfn ip_compute_cksum(n_ip_hdr_t *hdr) -> void {
  let *p = reinterpret_cast<unsigned char *>(hdr);
  p[10] = p[11] = 0;
  const let value = dynamips::packet::folded(dynamips::packet::sum_bytes(
      {p, static_cast<std::size_t>(p[0] & 15) * 4}));
  p[10] = static_cast<unsigned char>(value >> 8);
  p[11] = static_cast<unsigned char>(value);
}

cfn pkt_ctx_analyze(n_pkt_ctx_t *ctx, m_uint8_t *pkt, size_t len) -> int {
  if (!ctx || (!pkt && len))
    return FALSE;
  let parsed = dynamips::packet::analyze_into(*ctx, {pkt, len});
  if (!parsed) {
    *ctx = {};
    ctx->pkt = pkt;
    ctx->pkt_len = len;
    return FALSE;
  }
  return TRUE;
}

cfn pkt_ctx_ip_analyze_l4(n_pkt_ctx_t *ctx) -> int {
  switch (ctx->ip_l4_proto) {
  case N_IP_PROTO_TCP:
    ctx->flags |= N_PKT_CTX_FLAG_L4_TCP;
    break;
  case N_IP_PROTO_UDP:
    ctx->flags |= N_PKT_CTX_FLAG_L4_UDP;
    break;
  case N_IP_PROTO_ICMP:
    ctx->flags |= N_PKT_CTX_FLAG_L4_ICMP;
    break;
  }
  return TRUE;
}

cfn pkt_ctx_tcp_cksum(n_pkt_ctx_t *ctx, int ph) -> m_uint16_t {
  using namespace dynamips::packet;
  if (!ctx || !ctx->ip || !ctx->l4)
    return 0;
  const let *ip = reinterpret_cast<const unsigned char *>(ctx->ip);
  const std::size_t hlen = (ip[0] & 15) * 4;
  const std::size_t total = (static_cast<std::size_t>(ip[2]) << 8) | ip[3];
  if (total < hlen)
    return 0;
  let *data = static_cast<unsigned char *>(ctx->l4);
  const std::size_t len = total - hlen;
  const std::size_t skip = ctx->ip_l4_proto == N_IP_PROTO_TCP   ? 16
                           : ctx->ip_l4_proto == N_IP_PROTO_UDP ? 6
                                                                : len;
  unsigned int sum = sum_bytes({data, len});
  if (!(ctx->flags & N_PKT_CTX_FLAG_IP_FRAG) && skip + 1 < len)
    sum -= (static_cast<unsigned int>(data[skip]) << 8) | data[skip + 1];
  if (ph) {
    for (std::size_t i = 12; i < 20; i += 2)
      sum += (static_cast<unsigned int>(ip[i]) << 8) | ip[i + 1];
    sum += ctx->ip_l4_proto + len;
  }
  return folded(sum);
}

cfn cisco_isl_rewrite(m_uint8_t *pkt, m_uint32_t len) -> void {
  static constexpr unsigned char address[] = {1, 0, 12, 0, 16, 0};
  if (!pkt || len < N_ETH_HLEN || std::memcmp(pkt, address, 6) != 0)
    return;
  const unsigned int wire_length =
      (static_cast<unsigned int>(pkt[12]) << 8) | pkt[13];
  const unsigned int offset = N_ETH_HLEN + N_ISL_HDR_SIZE;
  if (wire_length < N_ISL_HDR_SIZE + 4 || len < 4)
    return;
  const unsigned int inner = wire_length - N_ISL_HDR_SIZE - 4;
  if (offset > len || inner > len - offset)
    return;
  pkt[4] = 0;
  const let crc =
      crc32_compute(0xffffffff, pkt + offset, static_cast<int>(inner));
  for (unsigned int i = 0; i < 4; ++i)
    pkt[len - 4 + i] = static_cast<unsigned char>(crc >> (8 * i));
}
