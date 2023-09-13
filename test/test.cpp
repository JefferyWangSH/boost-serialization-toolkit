/*
 *   test.cpp
 * 
 *     Created on: Sep 13, 2023
 *         Author: Jeffery Wang
 * 
 */

#include <iostream>
#include <complex>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>

#include <boost/format.hpp>
#include <boost/mpi.hpp>
#include "boost_serialization_eigen.hpp"
#include "boost_serialization_xtensor.hpp"

template <class T>
void test_communication(boost::mpi::communicator& world, const T& obj,
    const std::size_t r1, const std::size_t r2)
{
    if (world.rank() == r1) {
        world.send(r2, 0, obj);
        T msg;
        world.recv(r2, 1, msg);
        std::cout << boost::format("Rank %d: receive message from Rank %d\n") % r1 % r2; 
        std::cout << msg << std::endl;
    } else if (world.rank() == r2) {
        T msg;
        world.recv(r1, 0, msg);
        std::cout << boost::format("Rank %d: receive message from Rank %d\n") % r2 % r1; 
        std::cout << msg << std::endl;
        world.send(r1, 1, obj);
    }
}

int main(int argc, char* argv[]) {

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const std::size_t rank = static_cast<std::size_t>(world.rank());

    // --------------------------  test Eigen  ---------------------------

    Eigen::Tensor<double, 2> t1(2,2);
    t1.setConstant(rank);
    if (world.rank() == 0) { std::cout << "Eigen::Tensor:" << std::endl; }
    test_communication(world, t1, 0, 1);
    
    Eigen::TensorFixedSize<std::complex<double>, Eigen::Sizes<2,2,2>> t2;
    t2.setConstant(std::complex<double>(rank, rank));
    if (world.rank() == 0) { std::cout << "\nEigen::TensorFixedSize:" << std::endl; }
    test_communication(world, t2, 0, 1);

    // -------------------------  test xtensor  --------------------------

    xt::xarray<double>::shape_type shape = {2, 3};
    xt::xarray<double> a1(shape);
    a1.fill(rank);
    if (world.rank() == 0) { std::cout << "\nxt::xarray:" << std::endl; }
    test_communication(world, a1, 0, 1);

    xt::xtensor<std::complex<int>, 3> a2({2,2,2});
    a2.fill(std::complex<int>(rank, rank));
    if (world.rank() == 0) { std::cout << "\nxt::xtensor:" << std::endl; }
    test_communication(world, a2, 0, 1);

    return 0;
}