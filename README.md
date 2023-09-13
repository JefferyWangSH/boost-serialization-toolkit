# boost-serialization-toolkit

This repo contains preliminary implementations of the toolkit
for serializing customized C++ classes with boost::serialization module.

The supported objects, mainly used in numerical computations, include:
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)'s Eigen::Matrix, Eigen::SparseMatrix, and Eigen::Tensor.
* [xtensor](https://github.com/xtensor-stack/xtensor)'s xarray, xtensor, and xtensor_fixed (TODO).

For using the toolkit, just copy the corresponding header files in [`include`](include/) to your own project.

Some basic test and benchamrk can be found in [`test`](test/).
Follow the commands below to run the test.

```shell
# build the test
cd ${project_root}/test
mkdir build && cd build
cmake ..
make

# execute the test with mpi support
mpirun -np 2 ${where_your_executable_file_is}
```