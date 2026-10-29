#include "rommon_var.h"
#include <dynamips/macros.h>

#include <expected>
#include <fstream>
#include <memory>
#include <new>
#include <string>

namespace {

enum class error { invalid_argument, out_of_memory, io, internal };
template <typename T> using result = std::expected<T, error>;

template <typename Operation>
fn attempt(Operation operation) noexcept -> result<void> {
  try {
    return operation();
  } catch (const std::bad_alloc &) {
    return std::unexpected(error::out_of_memory);
  } catch (...) {
    return std::unexpected(error::internal);
  }
}

struct free_string {
  fn operator()(char *text) const noexcept -> void { free(text); }
};

struct free_variable {
  fn operator()(rommon_var *var) const noexcept -> void {
    free(var->value);
    free(var->name);
    free(var);
  }
};

using owned_string = std::unique_ptr<char, free_string>;
using owned_variable = std::unique_ptr<rommon_var, free_variable>;
using owned_file = std::unique_ptr<FILE, decltype(&fclose)>;

fn find_variable(rommon_var_list &list, const char *name) -> rommon_var * {
  for (let *var = list.var_list; var; var = var->next)
    if (strcmp(var->name, name) == 0)
      return var;
  return nullptr;
}

fn set_variable(rommon_var_list &list, const char *name, const char *value)
    -> result<void> {
  owned_string copy{strdup(value)};
  if (!copy)
    return std::unexpected(error::out_of_memory);

  if (let *var = find_variable(list, name)) {
    free(var->value);
    var->value = copy.release();
    return {};
  }

  owned_variable var{static_cast<rommon_var *>(calloc(1, sizeof(rommon_var)))};
  if (!var)
    return std::unexpected(error::out_of_memory);
  var->name = strdup(name);
  if (!var->name)
    return std::unexpected(error::out_of_memory);
  var->value = copy.release();
  var->next = list.var_list;
  list.var_list = var.release();
  return {};
}

fn update_file(rommon_var_list &list) -> result<void> {
  if (!list.filename)
    return std::unexpected(error::invalid_argument);

  owned_file file{fopen(list.filename, "w"), fclose};
  if (!file) {
    fprintf(stderr, "rommon_var_update_file: unable to create file %s (%s)\n",
            list.filename, strerror(errno));
    return std::unexpected(error::io);
  }

  for (let *var = list.var_list; var; var = var->next)
    if (fprintf(file.get(), "%s=%s\n", var->name,
                var->value ? var->value : "") < 0)
      return std::unexpected(error::io);

  if (fclose(file.release()) != 0)
    return std::unexpected(error::io);
  return {};
}

fn load_file(rommon_var_list &list) -> result<void> {
  if (!list.filename)
    return std::unexpected(error::invalid_argument);

  std::ifstream file{list.filename};
  if (!file.is_open())
    return std::unexpected(error::io);

  std::string line;
  while (std::getline(file, line)) {
    let *equals = strchr(line.data(), '=');
    if (!equals || equals == line.data())
      continue;
    *equals = '\0';
    if (let changed = set_variable(list, line.c_str(), equals + 1); !changed)
      return changed;
  }
  if (file.bad() || !file.eof())
    return std::unexpected(error::io);
  file.clear();
  file.close();
  if (file.fail())
    return std::unexpected(error::io);
  return {};
}

} // namespace

cfn rommon_load_file(rommon_var_list *list) -> int {
  if (!list)
    return -1;
  return attempt([&] { return load_file(*list); }) ? 0 : -1;
}

cfn rommon_var_update_file(rommon_var_list *list) -> int {
  if (!list)
    return -1;
  return attempt([&] { return update_file(*list); }) ? 0 : -1;
}

cfn rommon_var_find(rommon_var_list *list, char *name) -> rommon_var * {
  return list && name ? find_variable(*list, name) : nullptr;
}

cfn rommon_var_add(rommon_var_list *list, char *name, char *value) -> int {
  if (!list || !name || !value)
    return -1;
  return attempt([&]() -> result<void> {
    if (let changed = set_variable(*list, name, value); !changed)
      return changed;
    return update_file(*list);
  })
             ? 0
             : -1;
}

cfn rommon_var_add_str(rommon_var_list *list, char *text) -> int {
  if (!list || !text)
    return -1;
  let *equals = strchr(text, '=');
  if (!equals || equals == text)
    return -1;
  *equals = '\0';
  return rommon_var_add(list, text, equals + 1);
}

cfn rommon_var_get(rommon_var_list *list, char *name, char *buffer, size_t len)
    -> int {
  if (!list || !name || !buffer || len == 0)
    return -1;
  let *var = find_variable(*list, name);
  if (!var || !var->value)
    return -1;
  strncpy(buffer, var->value, len - 1);
  buffer[len - 1] = '\0';
  return 0;
}

cfn rommon_var_clear(rommon_var_list *list) -> void {
  if (!list)
    return;
  while (list->var_list) {
    owned_variable var{list->var_list};
    list->var_list = var->next;
  }
}
