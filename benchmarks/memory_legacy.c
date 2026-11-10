#include <stdint.h>
#include <stdlib.h>

#include "cpu.h"
#include "device.h"
#include "memory.h"
#include "vm.h"

int mips64_mts64_init(cpu_mips_t *cpu);
void mips64_mts64_init_memop_vectors(cpu_mips_t *cpu);

struct legacy_memory {
  cpu_gen_t gen;
  vm_instance_t vm;
  struct vdevice ram;
};

void *legacy_memory_create(void *ram, size_t size) {
  if (size > UINT32_MAX)
    return NULL;
  struct legacy_memory *result = calloc(1, sizeof(*result));
  if (!result)
    return NULL;
  cpu_mips_t *cpu = CPU_MIPS64(&result->gen);
  result->gen.vm = &result->vm;
  cpu->gen = &result->gen;
  cpu->vm = &result->vm;
  result->ram.host_addr = (m_iptr_t)ram;
  result->ram.phys_len = (m_uint32_t)size;
  result->ram.flags = VDEVICE_FLAG_CACHING;
  result->vm.dev_list = &result->ram;
  if (mips64_mts64_init(cpu) != 0) {
    free(result);
    return NULL;
  }
  mips64_mts64_init_memop_vectors(cpu);
  return result;
}

void legacy_memory_destroy(void *context) {
  if (context) {
    struct legacy_memory *memory = context;
    cpu_mips_t *cpu = CPU_MIPS64(&memory->gen);
    cpu->mts_shutdown(cpu);
    free(memory);
  }
}

uint64_t legacy_memory_load(void *context, uint64_t address) {
  struct legacy_memory *memory = context;
  cpu_mips_t *cpu = CPU_MIPS64(&memory->gen);
  cpu->mem_op_fn[MIPS_MEMOP_LD](cpu, address, 1);
  return cpu->gpr[1];
}
