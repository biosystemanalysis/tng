cmake_minimum_required(VERSION 2.8.8)

project(TNG_IO)


if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall")
elseif(WIN32)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /W2")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(BUILD_SHARED_LIBS "Enable shared libraries" ON)

option(TNG_BUILD_FORTRAN "Build Fortran compatible library and examples for testing" OFF)

option(TNG_BUILD_EXAMPLES "Build examples showing usage of the TNG API" ON)
option(TNG_BUILD_TEST "Build TNG testing binary." ON)
option(TNG_BUILD_COMPRESSION_TESTS "Build tests of the TNG compression library" OFF)

find_package(ZLIB QUIET)
option(TNG_BUILD_WITH_ZLIB "Build TNG with zlib compression" ${ZLIB_FOUND})

include(CheckIncludeFile)
check_include_file(inttypes.h   HAVE_INTTYPES_H)

include(BuildTNG.cmake)
tng_get_source_list(TNG_SOURCES TNG_COMPILE_DEFS)

tng_set_source_properties(WITH_ZLIB ${ZLIB_FOUND})

SET( hrtc_object_files
  src/compression/hrtc/hrtc_wrapper.o
  src/compression/hrtc/integer_encoding_library/src/compress/VSE-R.o
  src/compression/hrtc/integer_encoding_library/src/compress/VSEncodingBlocksHybrind.o
  src/compression/hrtc/integer_encoding_library/src/compress/VSEncodingSimple.o
  src/compression/hrtc/integer_encoding_library/src/compress/Simple16.o
  src/compression/hrtc/integer_encoding_library/src/compress/VSEncodingRest.o
  src/compression/hrtc/integer_encoding_library/src/compress/PForDelta.o
  src/compression/hrtc/integer_encoding_library/src/compress/VSEncodingBlocks.o
  src/compression/hrtc/integer_encoding_library/src/compress/VariableByte.o
  src/compression/hrtc/integer_encoding_library/src/compress/table/decGamma.o
  src/compression/hrtc/integer_encoding_library/src/compress/table/decUnary.o
  src/compression/hrtc/integer_encoding_library/src/compress/table/decDelta.o
  src/compression/hrtc/integer_encoding_library/src/compress/VSEncodingNaive.o
  src/compression/hrtc/integer_encoding_library/src/compress/VSEncodingDP.o
  src/compression/hrtc/integer_encoding_library/src/compress/Simple9.o
  src/compression/hrtc/integer_encoding_library/src/compress/OPTPForDelta.o
  src/compression/hrtc/integer_encoding_library/src/interger_encoding.o
  src/compression/hrtc/integer_encoding_library/src/io/BitsWriter.o
  src/compression/hrtc/integer_encoding_library/src/io/BitsReader.o
)

SET_SOURCE_FILES_PROPERTIES(
  ${hrtc_object_files}
  PROPERTIES
  EXTERNAL_OBJECT true
  # to say that "this is actually an object file, so it should not be compiled, only linked"
  # GENERATED true
  # to say that "it is OK that the obj-files do not exist before build time"
)

add_library(tng_io ${TNG_SOURCES} ${hrtc_object_files})
 
if (ZLIB_FOUND)
  list(APPEND EXTRA_LIBRARIES ${ZLIB_LIBRARIES})
  include_directories(${ZLIB_INCLUDE_DIRS})
endif()

if (UNIX)
  list(APPEND EXTRA_LIBRARIES m)
endif()
  
list(APPEND EXTRA_LIBRARIES stdc++)

target_link_libraries(tng_io ${EXTRA_LIBRARIES})

set_target_properties(tng_io PROPERTIES VERSION ${TNG_IO_VERSION} SOVERSION ${TNG_MAJOR_VERSION})

# Create the tng_ioConfig.cmake and tng_ioConfigVersion.cmake files for the install tree
set(CONF_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")
configure_file(              src/lib/tng_io-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/tng_io-config.cmake" @ONLY)
configure_file(              src/lib/tng_io-configVersion.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/tng_io-configVersion.cmake" @ONLY)

# Use GNUInstallDirst to set paths on multiarch systems
include(GNUInstallDirs)

# Install the tng_ioConfig.cmake and tng_ioConfigVersion.cmake
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/tng_io-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/tng_io-configVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tng_io")

install(TARGETS tng_io
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES include/tng/tng_io.h include/tng/tng_io_fwd.h ${CMAKE_CURRENT_BINARY_DIR}/include/tng/version.h
        DESTINATION include/tng)

#-- Add an Option to toggle the generation of the API documentation
option(TNG_BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)
if(TNG_BUILD_DOCUMENTATION)
  find_package(Doxygen)
  if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR
      "Doxygen is needed to build the documentation. Please install it correctly")
  endif()
  #-- Configure the Template Doxyfile for our specific project
  configure_file(Doxyfile.in
                 ${PROJECT_BINARY_DIR}/Doxyfile  @ONLY IMMEDIATE)
  #-- Add a custom target to run Doxygen when ever the project is built
  add_custom_target (Docs ALL
                                        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
                                        SOURCES ${PROJECT_BINARY_DIR}/Doxyfile)
  # IF you do NOT want the documentation to be generated EVERY time you build the project
  # then leave out the 'ALL' keyword from the above command.

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Documentation/
      DESTINATION share/tng/doc)
endif()

add_subdirectory(src)

