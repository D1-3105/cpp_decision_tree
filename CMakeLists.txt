cmake_minimum_required(VERSION 3.28)
project(decision_tree)

set(CMAKE_CXX_STANDARD 23)

# tbb
find_package(TBB REQUIRED)

# csv parser
include_directories(include/csv-parser/single_include)
find_package(Threads REQUIRED)

# ai component
add_library(decision_tree src/tree/DecisionTree.cpp src/tree/DecisionTree.h)

target_include_directories(decision_tree PUBLIC src/tree)

add_executable(decision_tree_train src/parser/main_train.cpp
        src/parser/CSVParser.cpp
        src/parser/CSVParser.h
        src/parser/Vectorizer.cpp
        src/parser/Vectorizer.h
        src/parser/MainTrainer.cpp
        src/parser/MainTrainer.h
)

target_link_libraries(decision_tree_train PRIVATE Threads::Threads decision_tree TBB::tbb)
