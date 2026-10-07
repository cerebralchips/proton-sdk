cmake_minimum_required(VERSION 3.26)
project(proton_frontend C CXX ASM)
get_filename_component(SDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Reuse the qualified runtime and BSP without editing the historical targets.
add_subdirectory("${SDK_ROOT}" sdk)
# Only this separate build uses the pinned, patched source copy. The source
# dependency and existing qualified runtime builds remain unmodified.
get_target_property(_deferred_sources iree_hal_utils_deferred_command_buffer.objects SOURCES)
list(FILTER _deferred_sources EXCLUDE REGEX "deferred_command_buffer\\.c$")
list(APPEND _deferred_sources "${PROTON_FRONTEND_RUN}/deferred_command_buffer.c")
set_property(TARGET iree_hal_utils_deferred_command_buffer.objects PROPERTY SOURCES "${_deferred_sources}")
foreach(mode fp32 w8a8)
  set(gen "${PROTON_FRONTEND_RUN}/${mode}")
  if(NOT EXISTS "${gen}/embed.c")
    continue()
  endif()
  add_executable(frontend_${mode} "${SDK_ROOT}/runtime/frontend_runner.c"
    "${SDK_ROOT}/runtime/proton/start.S" "${SDK_ROOT}/runtime/proton/platform.c"
    "${SDK_ROOT}/runtime/proton/console.c" "${gen}/embed.c" "${gen}/dispatch.o")
  target_include_directories(frontend_${mode} PRIVATE "${SDK_ROOT}/runtime" "${gen}")
  target_compile_options(frontend_${mode} PRIVATE -Wno-unused-command-line-argument)
  target_link_libraries(frontend_${mode} PRIVATE iree::runtime
    iree::hal::drivers::local_sync::sync_driver
    iree::hal::local::loaders::static_library_loader printf::printf m)
  target_link_options(frontend_${mode} PRIVATE "-Wl,-Map=${CMAKE_BINARY_DIR}/frontend_${mode}.map")
endforeach()
