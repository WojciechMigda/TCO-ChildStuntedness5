#!/bin/sh

cat num.hpp array2d.hpp extract_subject_ranges.hpp ChildStuntedness5.hpp | grep -v "#include \"" > submission.cpp
g++ -std=c++11 -c submission.cpp
gvim submission.cpp &
