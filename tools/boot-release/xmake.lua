local source_root = path.absolute("../..", os.scriptdir())

set_languages("cxx23")
set_toolchains("clang")
includes(path.join(source_root, "xmake.lua"))

target("c7200-boot")
    set_rundir(source_root)
target_end()
