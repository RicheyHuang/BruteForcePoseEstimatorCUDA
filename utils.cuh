#include <iostream>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <host_defines.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>

class Rigid3f
{
public:
    Rigid3f()
    {
        _translation = Eigen::Vector3f(0, 0, 0);
        _rotation = Eigen::Vector3f(0, 0, 0);
    }
    Rigid3f(const Eigen::Vector3f& translation, const Eigen::Vector3f& rotation)
    {
        _translation = translation;
        _rotation = rotation;
    }

    Eigen::Vector3f _translation;
    Eigen::Vector3f _rotation;
};

int GetOptPoseIndex(const std::vector<Eigen::Vector3f>& submap, const std::vector<Eigen::Vector3f>& map, const std::vector<Rigid3f>& poses);