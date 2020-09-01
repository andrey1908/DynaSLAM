echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building DynaSLAM ..."

if ["$1" == ""]; then
    echo "Specify python version"
    echo "ERROR occurred"
    exit 1
fi

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=$1
make -j
