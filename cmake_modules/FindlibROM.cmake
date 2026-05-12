if(NOT LIBROM_PATH)
    message(FATAL_ERROR "LIBROM_PATH not specified.")
endif(NOT LIBROM_PATH)

find_library(LIBROM_LIB libROM.so HINTS "${LIBROM_PATH}/build/lib")
find_path(LIBROM_INCLUDES librom.h HINTS "${LIBROM_PATH}/lib")

mark_as_advanced(LIBROM_LIB LIBROM_INCLUDES)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(libROM REQUIRED_VARS LIBROM_LIB LIBROM_INCLUDES)