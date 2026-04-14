mkdir build
cd build
cmake .. -DBUILD_PYTHON=ON -DENABLE_GUROBI=ON
cmake --build .
ctest -L unit
echo Done
