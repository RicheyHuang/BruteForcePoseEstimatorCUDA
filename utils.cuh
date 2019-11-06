#include <iostream>
#include <stdio.h>
#include <Eigen/Core>
#include <host_defines.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

int kernelTest(const std::vector<Eigen::Vector3f>& submap, const std::vector<Eigen::Vector3f>& map);