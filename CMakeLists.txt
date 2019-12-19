cmake_minimum_required(VERSION 2.8) 
PROJECT (IP_binarization)
find_package(OpenCV REQUIRED )
set( NAME_SRC
    src/main.cpp    
)
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
link_directories( ${CMAKE_BINARY_DIR}/bin)
link_directories(${OpenCV_LIB_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(OCV_test ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries( OCV_test ${OpenCV_LIBS} )
