local version = "0.2.28"

set_project("dynamips")
set_version(version)
set_languages("c23", "cxx23")
set_toolchains("clang")

add_rules("mode.debug", "mode.release")
if not has_config("system_packages") then
    add_rules("plugin.compile_commands.autoupdate", { outputdir = "." })
end

local default_jit_arch = is_arch("x86_64") and "amd64" or "nojit"

option("dynamips_arch")
    set_default(default_jit_arch)
    set_values("amd64", "x86", "ppc32", "nojit")
    set_showmenu(true)
    set_description("Select the JIT backend")
option_end()

option("enable_gen_eth")
    set_default(true)
    set_showmenu(true)
    set_description("Enable generic Ethernet through libpcap when available")
option_end()

option("enable_linux_eth")
    set_default(is_plat("linux"))
    set_showmenu(true)
    set_description("Enable Linux raw-socket Ethernet")
option_end()

option("system_packages")
    set_default(false)
    set_showmenu(true)
    set_description("Use system libelf and libpcap")
option_end()

if has_config("system_packages") then
    add_requires("libelf", { system = true })
else
    add_requires("libelf")
end
add_requires("asio 1.36.0", { system = false })
add_requires("asmjit 2024.05.21", { system = false })
add_requires("zlib 1.3.2", { system = false })
if has_config("enable_gen_eth") then
    if has_config("system_packages") then
        add_requires("libpcap", { system = true })
    else
        add_requires("libpcap", { optional = true })
    end
end

local code = "stable"
local jit_arch = get_config("dynamips_arch") or default_jit_arch
local generated_dir = path.join("$(builddir)", "generated", code)

rule("dynamips.microcode")
    before_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local outputfile = path.join("build", "generated", code, path.basename(sourcefile) .. "_dump.inc")
        local address = path.basename(sourcefile) == "mips64_microcode" and "0xbfc00000" or "0xfff00000"
        batchcmds:show_progress(opt.progress, "${color.build.object}generating %s", outputfile)
        batchcmds:mkdir(path.directory(outputfile))
        batchcmds:vrunv(target:dep("rom2c"):targetfile(), { sourcefile, outputfile, address })
        batchcmds:add_depfiles(sourcefile, target:dep("rom2c"):targetfile())
        batchcmds:set_depmtime(os.mtime(outputfile))
        batchcmds:set_depcache(target:dependfile(outputfile))
    end)
rule_end()

local function configure_dynamips_target()
    set_warnings("all")
    set_optimize("fastest")
    add_cflags("-fasm", "-fomit-frame-pointer", { force = true })
    add_includedirs("common", code, generated_dir)
    add_includedirs("include", { public = true })
    add_defines(
        'DYNAMIPS_VERSION="' .. version .. '"',
        'JIT_ARCH="' .. jit_arch .. '"',
        "JIT_CPU=CPU_" .. jit_arch,
        'MIPS64_ARCH_INC_FILE="mips64_' .. jit_arch .. '_trans.h"',
        'PPC32_ARCH_INC_FILE="ppc32_' .. jit_arch .. '_trans.h"',
        "HAS_POSIX_MEMALIGN=1",
        "HAS_RFC2553=1",
        "_FILE_OFFSET_BITS=64",
        "_LARGEFILE_SOURCE",
        "_LARGEFILE64_SOURCE",
        "_DEFAULT_SOURCE",
        "OSNAME=" .. (is_plat("macosx") and "Darwin" or is_plat("windows") and "Windows" or "Linux")
    )
    add_packages("libelf", { public = true })
    if is_plat("linux") then
        add_syslinks("dl", "rt", "nsl", "pthread", { public = true })
    else
        add_syslinks("pthread", { public = true })
    end
    if has_config("enable_linux_eth") and is_plat("linux") then
        add_defines("LINUX_ETH")
    end
    if has_config("enable_gen_eth") and has_package("libpcap") then
        add_defines("GEN_ETH")
        add_packages("libpcap", { public = true })
    end
    if jit_arch == "amd64" then
        add_cflags("-m64", { force = true })
        add_ldflags("-m64", { force = true })
    elseif jit_arch == "x86" or jit_arch == "ppc32" then
        add_cflags("-m32", { force = true })
        add_ldflags("-m32", { force = true })
    end
end

target("rom2c")
    set_default(false)
    set_kind("binary")
    set_plat(os.host())
    set_arch(os.arch())
    set_policy("build.fence", true)
    add_defines("_DEFAULT_SOURCE")
    add_includedirs("common", code)
    add_packages("libelf")
    add_files("common/rom2c.c")
target_end()

target("dynamips-core")
    configure_dynamips_target()
    set_kind("static")
    add_packages("asio")
    add_packages("zlib", { public = true })
    add_files(
        "common/base64.cppm", "common/base64_compat.cpp",
        "common/crc.cpp", "common/hash.cpp", "common/mempool.cpp", "common/ptask.cpp",
        "common/native_fd.cppm", "common/rbtree.cpp", "common/timer.cpp", "common/rommon_var.cpp", "common/host_memory.cppm", "common/net_packet.cppm", "common/net_io_udp.cppm", "common/vtty_io.cppm"
    )
    add_defines("DYNAMIPS_CPP_HOST_MEMORY", "DYNAMIPS_CPP_PACKET", "DYNAMIPS_ASIO_VTTY")
    add_files("common/registry.cppm", { public = true })
    add_deps("rom2c")
    add_rules("dynamips.microcode")
    add_headerfiles("include/(dynamips/*.h)")
    add_files("common/*.c"
        .. "|dynamips_bridge_*.c|dynamips_main.c|hv_*.c"
        .. "|ppc32_nojit_trans.c|ppc32_ppc32_trans.c|profiler.c|rom2c.c"
        .. "|base64.c|crc.c|hash.c|mempool.c|ptask.c|rbtree.c|timer.c|registry.c|rommon_var.c")
    after_install(function (target)
        for _, file in ipairs(target:pkg("zlib"):get("libfiles")) do
            os.cp(file, path.join(target:installdir(), "lib"))
        end
    end)
    if not (has_config("enable_linux_eth") and is_plat("linux")) then
        remove_files("common/linux_eth.c")
    end
    if not (has_config("enable_gen_eth") and has_package("libpcap")) then
        remove_files("common/gen_eth.c")
    end
    add_files(code .. "/*.c|hv_vm.c|hypervisor.c|mips_mts.c|mips64_*_trans.c|ppc32_*_trans.c")
    add_files(code .. "/mips64_microcode", code .. "/ppc32_microcode", { rule = "dynamips.microcode" })
    if jit_arch == "ppc32" then
        add_files(code .. "/mips64_ppc32_trans.c", "common/ppc32_ppc32_trans.c")
        add_cflags("-Wa,-mregnames", { force = true })
    else
        add_files(code .. "/mips64_" .. jit_arch .. "_trans.c")
        if jit_arch == "nojit" then
            add_files("common/ppc32_nojit_trans.c")
        else
            add_files(code .. "/ppc32_" .. jit_arch .. "_trans.c")
        end
    end
target_end()

target("dynamips-bindings")
    configure_dynamips_target()
    set_kind("static")
    add_deps("dynamips-core", { public = true })
    add_headerfiles("include/(dynamips/*.h)")
    add_installfiles("pkgconfig/dynamips-bindings.pc", { prefixdir = "lib/pkgconfig" })
    add_files("modules/*.cppm|asmjit.cppm", { public = true })
    add_files("bindings/*.cpp")
    add_files("common/dynamips_bridge_*.c")
target_end()

target("dynamips-legacy-hypervisor")
    configure_dynamips_target()
    set_kind("static")
    set_default(false)
    add_deps("dynamips-core", { public = true })
    add_files("common/hv_*.c", "stable/hv_vm.c", "stable/hypervisor.c")
target_end()

target("dynamips-module-test")
    set_default(false)
    set_kind("binary")
    add_deps("dynamips-bindings")
    add_files("tests/module.cpp")
    add_tests("lifecycle")
target_end()

target("jit-core")
    set_default(false)
    set_kind("static")
    set_warnings("all", "error")
    add_cxxflags("-Wextra", "-Wpedantic", "-Wshadow", "-Wconversion", "-Wsign-conversion")
    add_includedirs("include")
    add_packages("asmjit", { public = true })
    add_files("modules/asmjit.cppm", "src/jit/ir.cppm", "src/jit/mips64.cppm", "src/jit/amd64.cppm", "src/jit/memory.cppm", "src/jit/fifo.cppm", "src/jit/uart.cppm", "src/jit/example.cppm", "src/jit/dispatch.cppm", { public = true })
target_end()

target("jit-playground")
    set_default(false)
    set_kind("binary")
    set_warnings("all", "error")
    add_cxxflags("-Wextra", "-Wpedantic", "-Wshadow", "-Wconversion", "-Wsign-conversion")
    add_includedirs("include")
    add_deps("jit-core")
    add_files("src/jit_main.cpp")
target_end()

target("jit-test")
    set_default(false)
    set_kind("binary")
    set_warnings("all", "error")
    add_cxxflags("-Wextra", "-Wpedantic", "-Wshadow", "-Wconversion", "-Wsign-conversion")
    add_includedirs("include")
    add_deps("jit-core")
    add_files("tests/jit.cpp")
    add_tests("lowering")
target_end()

target("memory-test")
    set_default(false)
    set_kind("binary")
    set_warnings("all", "error")
    add_cxxflags("-Wextra", "-Wpedantic", "-Wshadow", "-Wconversion", "-Wsign-conversion")
    add_includedirs("include")
    add_deps("jit-core")
    add_files("tests/memory.cpp")
    add_tests("access")
target_end()

target("memory-bench")
    configure_dynamips_target()
    set_default(false)
    set_kind("binary")
    add_includedirs("include")
    add_defines("MEMORY_BENCH_RELEASE=" .. (is_mode("release") and "1" or "0"))
    add_deps("jit-core", "dynamips-core")
    add_files("benchmarks/memory.cpp", "benchmarks/memory_legacy.c")
target_end()

target("dynamips")
    configure_dynamips_target()
    set_kind("binary")
    add_deps("dynamips-core", "dynamips-legacy-hypervisor")
    add_files("common/dynamips_main.c")
    add_installfiles("LICENSE", "README.md", { prefixdir = "share/doc/dynamips" })
target_end()
