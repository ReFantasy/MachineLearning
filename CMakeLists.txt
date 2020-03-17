cmake_minimum_required(VERSION 3.15)
project(MachineLearning)

set(CMAKE_CXX_STANDARD 14)

add_executable(MachineLearning main.cpp test.cpp test.h src/decision_tree.h src/decision_tree.cpp src/tuple_helper.cpp src/tuple_helper.h)