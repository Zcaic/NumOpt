cmake_minimum_required(VERSION 3.26.0)
# if (POLICY CMP0091)
#   cmake_policy(SET CMP0091 NEW)
# endif (POLICY CMP0091)

project(SOLVER VERSION 0.1.0 LANGUAGES C CXX)

# set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS off)
set(CMAKE_VERBOSE_MAKEFILE ON)

# set(CASADI_DIR "D:/Doc/casadi-3.7.2-windows64-py312/casadi/")
# set(CASADI_LIB "${CASADI_DIR}/casadi.lib")


add_library(bisection SHARED "./src/bisection.cpp")
set_target_properties(bisection PROPERTIES OUTPUT_NAME "casadi_rootfinder_bisection")
set_target_properties(bisection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/test/)
target_include_directories(bisection AFTER PRIVATE
    "../casadi/" # 这里提供 rootfinder_impl.hpp 等内部底层头文件
    "../casadi-3.7.2-windows64-py312/casadi/include/" # 这里提供 casadi_export.h 等自动生成的头文件
) 
target_link_directories(bisection PRIVATE "D:/micromamba/envs/py12/Lib/site-packages/casadi/")
target_link_libraries(bisection PRIVATE casadi)
# target_link_libraries(bisection PRIVATE ${CASADI_LIB})

# if(MINGW)
#     target_link_options(bisection PRIVATE 
#             "-static-libgcc" 
#             "-static-libstdc++"
#         )
#     target_link_libraries(bisection PRIVATE 
#         "-Wl,-Bstatic,--whole-archive,-lwinpthread,--no-whole-archive,-Bdynamic"
#     )
# endif()
# set_target_properties(bisection PROPERTIES LINK_FLAGS "-Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive")