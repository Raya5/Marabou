
curl https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz --output gurobi11.0.3_linux64.tar.gz
tar xvfz gurobi11.0.3_linux64.tar.gz
cd gurobi1103/linux64/src/build
make 
cp libgurobi_c++.a ../../lib/
cd ../../../../