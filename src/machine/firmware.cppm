module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <system_error>
#include <string_view>
#include <utility>
#include <elf.h>
#include <gelf.h>
#include <libelf.h>

#include <dynamips/macros.h>

export module machine.firmware;

import host_memory;

export namespace dynamips::firmware {
enum class error_code {
  host,
  invalid_elf,
  unsupported_format,
  invalid_section,
  too_many_sections,
  too_large,
  invalid_entry
};

constexpr fn message(error_code code) noexcept -> std::string_view {
  switch (code) {
  case error_code::host:
    return "host I/O failed";
  case error_code::invalid_elf:
    return "invalid ELF metadata";
  case error_code::unsupported_format:
    return "expected an ELF32 big-endian MIPS executable";
  case error_code::invalid_section:
    return "malformed allocated section";
  case error_code::too_many_sections:
    return "allocated section limit exceeded";
  case error_code::too_large:
    return "firmware file limit exceeded";
  case error_code::invalid_entry:
    return "entry point is not an aligned executable instruction";
  }
  return "invalid firmware";
}

struct error {
  error_code code;
  std::size_t section = 0;
  std::error_code host{};
};

struct section {
  std::uint64_t address;
  std::uint64_t size;
  std::span<const std::byte> contents;
  bool executable;
};

constexpr fn canonical(std::uint32_t address) noexcept -> std::uint64_t {
  return address & 0x80000000u ? 0xffffffff00000000ULL | address : address;
}

class image {
public:
  static constexpr std::size_t max_sections = 64;
  static constexpr std::size_t max_file_size = 256 * 1024 * 1024;
  fn entry() const noexcept -> std::uint64_t { return entry_; }
  fn sections() const noexcept -> std::span<const section> {
    return {sections_.data(), count_};
  }

  [[nodiscard]] static fn read(const char *path) noexcept
      -> std::expected<image, error> {
    let loaded = host::read_file_bounded(path, max_file_size);
    if (!loaded) {
      if (loaded.error() == std::errc::file_too_large)
        return std::unexpected(error{error_code::too_large});
      return std::unexpected(error{error_code::host, 0, loaded.error()});
    }
    if (elf_version(EV_CURRENT) == EV_NONE)
      return std::unexpected(error{error_code::invalid_elf});

    struct end_elf {
      fn operator()(Elf *value) const noexcept -> void { elf_end(value); }
    };

    image result;
    result.file_ = std::move(*loaded);
    std::unique_ptr<Elf, end_elf> elf{elf_memory(
        reinterpret_cast<char *>(result.file_.bytes.get()), result.file_.size)};
    GElf_Ehdr header{};
    if (!elf || !gelf_getehdr(elf.get(), &header))
      return std::unexpected(error{error_code::invalid_elf});
    if (header.e_ident[EI_CLASS] != ELFCLASS32 ||
        header.e_ident[EI_DATA] != ELFDATA2MSB || header.e_type != ET_EXEC ||
        header.e_version != EV_CURRENT ||
        (header.e_machine != EM_MIPS && header.e_machine != 0x19))
      return std::unexpected(error{error_code::unsupported_format});
    result.entry_ = canonical(static_cast<std::uint32_t>(header.e_entry));
    const let bytes =
        std::as_bytes(std::span{result.file_.bytes.get(), result.file_.size});
    std::size_t section_count = 0;
    if (elf_getshdrnum(elf.get(), &section_count) != 0 || section_count > 4096)
      return std::unexpected(error{error_code::invalid_elf});
    bool entry_found = false;
    for (std::size_t index = 1; index < section_count; index++) {
      Elf_Scn *current = elf_getscn(elf.get(), index);
      if (!current)
        return std::unexpected(error{error_code::invalid_section, index});
      GElf_Shdr description{};
      if (!gelf_getshdr(current, &description))
        return std::unexpected(error{error_code::invalid_section, index});
      if (!(description.sh_flags & SHF_ALLOC) || description.sh_size == 0)
        continue;
      if (result.count_ == max_sections)
        return std::unexpected(error{error_code::too_many_sections, index});
      if (description.sh_type == SHT_NULL ||
          (description.sh_flags & SHF_COMPRESSED) ||
          description.sh_addr > UINT32_MAX ||
          description.sh_size > UINT32_MAX ||
          description.sh_size > (1ULL << 32) - description.sh_addr)
        return std::unexpected(error{error_code::invalid_section, index});
      std::span<const std::byte> contents;
      if (description.sh_type != SHT_NOBITS) {
        if (description.sh_offset > bytes.size() ||
            description.sh_size > bytes.size() - description.sh_offset)
          return std::unexpected(error{error_code::invalid_section, index});
        contents =
            bytes.subspan(static_cast<std::size_t>(description.sh_offset),
                          static_cast<std::size_t>(description.sh_size));
      }
      const let address =
          canonical(static_cast<std::uint32_t>(description.sh_addr));
      if (description.sh_size > UINT64_MAX - address)
        return std::unexpected(error{error_code::invalid_section, index});
      const bool executable = (description.sh_flags & SHF_EXECINSTR) != 0;
      for (std::size_t i = 0; i < result.count_; i++) {
        const let &other = result.sections_[i];
        if (address < other.address + other.size &&
            other.address < address + description.sh_size)
          return std::unexpected(error{error_code::invalid_section, index});
      }
      result.sections_[result.count_++] = {address, description.sh_size,
                                           contents, executable};
      entry_found |= executable && description.sh_size >= 4 &&
                     result.entry_ >= address &&
                     result.entry_ - address <= description.sh_size - 4;
    }
    if (result.count_ == 0 || !entry_found || result.entry_ % 4)
      return std::unexpected(error{error_code::invalid_entry});
    return result;
  }

private:
  host::file_buffer file_{};
  std::array<section, max_sections> sections_{};
  std::size_t count_ = 0;
  std::uint64_t entry_ = 0;
};
} // namespace dynamips::firmware
