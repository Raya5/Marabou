curl https://packages.gurobi.com/12.0/gurobi12.0.3_linux64.tar.gz --output gurobi12.0.3_linux64.tar.gz
tar xvfz gurobi12.0.3_linux64.tar.gz
cd gurobi1203/linux64/src/build
make
cp libgurobi_c++.a ../../lib/
cd ../../../..