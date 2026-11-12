module;

#include <algorithm>
#include <cerrno>
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
#include <minizip/unzip.h>

#include <dynamips/macros.h>

export module machine.firmware;

import host_memory;

export namespace dynamips::firmware {
enum class error_code {
  host,
  invalid_archive,
  missing_archive_image,
  multiple_archive_images,
  unsupported_archive,
  invalid_elf,
  unsupported_format,
  invalid_section,
  too_many_sections,
  too_large,
  invalid_entry
};

constexpr fn message(error_code code) noexcept -> std::string_view {
  switch (code) {
  case error_code::invalid_archive:
    return "invalid or corrupt ZIP archive";
  case error_code::missing_archive_image:
    return "ZIP needs one nonempty .bin or .image file";
  case error_code::multiple_archive_images:
    return "ZIP contains multiple IOS images";
  case error_code::unsupported_archive:
    return "ZIP image must be unencrypted and use stored or deflate "
           "compression";
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
  fn machine_id() const noexcept -> std::uint16_t { return machine_id_; }
  fn entry() const noexcept -> std::uint64_t { return entry_; }
  fn sections() const noexcept -> std::span<const section> {
    return {sections_.data(), count_};
  }

  [[nodiscard]] static fn read(const char *path) noexcept
      -> std::expected<image, error> {
    let loaded = read_source(path);
    if (!loaded)
      return std::unexpected(loaded.error());
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
    result.machine_id_ = header.e_machine;
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
  static fn suffix(std::string_view name, std::string_view ending) noexcept
      -> bool {
    if (name.size() < ending.size())
      return false;
    name.remove_prefix(name.size() - ending.size());
    for (std::size_t i = 0; i < name.size(); i++) {
      const char folded = name[i] >= 'A' && name[i] <= 'Z'
                              ? static_cast<char>(name[i] + ('a' - 'A'))
                              : name[i];
      if (folded != ending[i])
        return false;
    }
    return true;
  }

  static fn read_source(const char *path) noexcept
      -> std::expected<host::file_buffer, error> {
    if (!path)
      return std::unexpected(
          error{error_code::host, 0,
                std::make_error_code(std::errc::invalid_argument)});
    if (!suffix(path, ".zip")) {
      let loaded = host::read_file_bounded(path, max_file_size);
      if (!loaded)
        return std::unexpected(error{loaded.error() == std::errc::file_too_large
                                         ? error_code::too_large
                                         : error_code::host,
                                     0, loaded.error()});
      return std::move(*loaded);
    }
    errno = 0;
    unzFile archive = unzOpen64(path);
    if (!archive)
      return std::unexpected(
          errno ? error{error_code::host, 0, {errno, std::generic_category()}}
                : error{error_code::invalid_archive});
    let extracted = [&]() -> std::expected<host::file_buffer, error> {
      unz_global_info64 global{};
      if (unzGetGlobalInfo64(archive, &global) != UNZ_OK ||
          global.number_entry > 4096)
        return std::unexpected(error{error_code::invalid_archive});
      bool found = false;
      unz64_file_pos selected{};
      std::size_t size = 0;
      std::array<char, 65536> name{};
      int status = unzGoToFirstFile(archive);
      for (std::uint64_t i = 0; i < global.number_entry; i++) {
        unz_file_info64 info{};
        if (status != UNZ_OK ||
            unzGetCurrentFileInfo64(archive, &info, name.data(),
                                    static_cast<uLong>(name.size()), nullptr, 0,
                                    nullptr, 0) != UNZ_OK ||
            info.size_filename >= name.size())
          return std::unexpected(error{error_code::invalid_archive});
        const std::string_view filename{
            name.data(), static_cast<std::size_t>(info.size_filename)};
        if (suffix(filename, ".bin") || suffix(filename, ".image")) {
          if (found)
            return std::unexpected(error{error_code::multiple_archive_images});
          if (info.uncompressed_size == 0)
            return std::unexpected(error{error_code::missing_archive_image});
          if (info.uncompressed_size > max_file_size)
            return std::unexpected(error{error_code::too_large});
          if ((info.flag & 1) ||
              (info.compression_method != 0 && info.compression_method != 8))
            return std::unexpected(error{error_code::unsupported_archive});
          if (unzGetFilePos64(archive, &selected) != UNZ_OK)
            return std::unexpected(error{error_code::invalid_archive});
          size = static_cast<std::size_t>(info.uncompressed_size);
          found = true;
        }
        status = unzGoToNextFile(archive);
      }
      if (!found)
        return std::unexpected(error{error_code::missing_archive_image});
      if (status != UNZ_END_OF_LIST_OF_FILE ||
          unzGoToFilePos64(archive, &selected) != UNZ_OK ||
          unzOpenCurrentFile(archive) != UNZ_OK)
        return std::unexpected(error{error_code::invalid_archive});
      let contents = [&]() -> std::expected<host::file_buffer, error> {
        let allocated =
            host::allocate_aligned(alignof(std::max_align_t), size);
        if (!allocated)
          return std::unexpected(error{error_code::host, 0, allocated.error()});
        std::size_t offset = 0;
        while (offset < size) {
          const let chunk = static_cast<unsigned>(
              std::min<std::size_t>(size - offset, 64 * 1024));
          const int count =
              unzReadCurrentFile(archive, allocated->get() + offset, chunk);
          if (count <= 0)
            return std::unexpected(error{error_code::invalid_archive});
          offset += static_cast<std::size_t>(count);
        }
        unsigned char extra;
        if (unzReadCurrentFile(archive, &extra, 1) != 0)
          return std::unexpected(error{error_code::invalid_archive});
        return host::file_buffer{std::move(*allocated), size};
      }();
      const int closed = unzCloseCurrentFile(archive);
      if (closed != UNZ_OK && contents)
        return std::unexpected(error{error_code::invalid_archive});
      return contents;
    }();
    const int closed = unzClose(archive);
    if (closed != UNZ_OK && extracted)
      return std::unexpected(error{error_code::invalid_archive});
    return extracted;
  }

  host::file_buffer file_{};
  std::array<section, max_sections> sections_{};
  std::size_t count_ = 0;
  std::uint64_t entry_ = 0;
  std::uint16_t machine_id_ = 0;
};
} // namespace dynamips::firmware
