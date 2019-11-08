#include "utils.cuh"


struct get_transform
{
    __host__ __device__
    Eigen::Matrix4f operator()(const Eigen::Matrix<float, 6, 1>& pose)
    {
        float alpha = pose[0];
        float beta  = pose[1];
        float gamma = pose[2];

        float x = pose[3];
        float y = pose[4];
        float z = pose[5];

        Eigen::Matrix4f transform;

        // ZYX order
        transform << cosf(beta)*cosf(gamma),                                     -cosf(beta)*sinf(gamma),                                        sinf(beta),              x,
                     sinf(alpha)*sinf(beta)*cosf(gamma)+cosf(alpha)*sinf(gamma), -sinf(alpha)*sinf(beta)*sinf(gamma)+cosf(alpha)*cosf(gamma),   -sinf(alpha)*cosf(beta),  y,
                    -cosf(alpha)*sinf(beta)*cosf(gamma)+sinf(alpha)*sinf(gamma),  cosf(alpha)*sinf(beta)*sinf(gamma)+sinf(alpha)*cosf(gamma),    cosf(alpha)*cosf(beta),  z,
                     0.0,                                                         0.0,                                                           0.0,                     1.0;

//        // XYZ order
//        transform << cosf(beta)*cosf(alpha),     sinf(gamma)*sinf(beta)*cosf(alpha)-cosf(gamma)*sinf(alpha),    cosf(gamma)*sinf(beta)*cosf(alpha)+sinf(gamma)*sinf(alpha), x,
//                     cosf(beta)*sinf(alpha),     sinf(gamma)*sinf(beta)*sinf(alpha)+cosf(gamma)*cosf(alpha),    cosf(gamma)*sinf(beta)*sinf(alpha)-sinf(gamma)*cosf(alpha), y,
//                    -sinf(beta),                 sinf(gamma)*cosf(beta),                                        cosf(gamma)*cosf(beta)                                      z,
//                     0.0,                         0.0,                                                           0.0,                                                        1.0;

        return transform;
    }
};


struct point_transform
{
    const Eigen::Vector3f _point;
    const float _resolution;

    explicit point_transform(const Eigen::Vector3f& point, const float& resolution):_point(point), _resolution(resolution){}

    __host__ __device__
    Eigen::Vector3f operator()(const Eigen::Matrix4f& transform)
    {
        Eigen::Vector4f homo_point;
        homo_point << _point[0], _point[1], _point[2], 1.0;

        Eigen::Vector4f homo_transformed_point =  homo_point[0] * transform.col(0) + homo_point[1] * transform.col(1) + homo_point[2] * transform.col(2) + homo_point[3] * transform.col(3);

        Eigen::Vector3f transformed_point;
        transformed_point << homo_transformed_point[0]/_resolution, homo_transformed_point[1]/_resolution, homo_transformed_point[2]/_resolution;
        return transformed_point;
    }
};

struct eigen_less_than
{
    __host__ __device__
    bool operator()(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
    {
        return (lhs[0]<rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]<rhs[1])||((lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]<rhs[2]);
    }
};


__host__ __device__ bool operator==(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
{
    return fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6;
}

__host__ __device__ bool operator>(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
{
    return (lhs[0]>rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]>rhs[1])||((lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]>rhs[2]);
}

__host__ __device__ int BinarySearchRecursive(const Eigen::Vector3f* points, int low, int high, Eigen::Vector3f point)
{
    if (low > high)
        return -1;
    int mid = low + (high - low) / 2;
    if (points[mid] == point)
        return mid;
    else if (points[mid] > point)
        return BinarySearchRecursive(points, low, mid - 1, point);
    else
        return BinarySearchRecursive(points, mid + 1, high, point);
}

struct match
{
    const Eigen::Vector3f* _map;
    const int _size;

    explicit match(Eigen::Vector3f* map, int size):_map(map), _size(size){}

    __host__ __device__
    float operator()(const Eigen::Vector3f& point)
    {
        int idx = BinarySearchRecursive(_map, 0, _size-1, point);
        if(idx < 0)
        {
            return 0.0;
        }
        else
        {
            return 1.0;
        }
    }
};

struct compute_score
{
    const int _size;
    explicit compute_score(int size):_size(size){}

    __host__ __device__
    float operator()(const float& sum)
    {
        return float(sum/float(_size));
    }
};


struct get_unit_pose
{
    const int _offset;
    const float _resolution;
    explicit get_unit_pose(const int& offset, const float& resolution):_offset(offset), _resolution(resolution){}

    __host__ __device__
    float operator()(int index)
    {
        return float((index-_offset)*_resolution);
//        return float(index);
    }
};

struct get_6dof
{
    const int _loop_size_rpyxyz;
    const int _loop_size_pyxyz;
    const int _loop_size_yxyz;
    const int _loop_size_xyz;
    const int _loop_size_yz;
    const int _loop_size_z;


    const float* _angles;
    const float* _displacements;


    explicit get_6dof(const int& loop_size_rpyxyz,
                      const int& loop_size_pyxyz,
                      const int& loop_size_yxyz,
                      const int& loop_size_xyz,
                      const int& loop_size_yz,
                      const int& loop_size_z,
                      const float* angles,
                      const float* displacements):
                      _loop_size_rpyxyz(loop_size_rpyxyz),
                      _loop_size_pyxyz(loop_size_pyxyz),
                      _loop_size_yxyz(loop_size_yxyz),
                      _loop_size_xyz(loop_size_xyz),
                      _loop_size_yz(loop_size_yz),
                      _loop_size_z(loop_size_z),
                      _angles(angles),
                      _displacements(displacements){}

    __host__ __device__
    Eigen::Matrix<float, 6, 1> operator()(int pose_index)
    {
        Eigen::Matrix<float, 6, 1> pose;

        pose(0, 0) = _angles[int(pose_index/_loop_size_pyxyz)];
        pose(1, 0) = _angles[int(pose_index%_loop_size_pyxyz/_loop_size_yxyz)];
        pose(2, 0) = _angles[int(pose_index%_loop_size_pyxyz%_loop_size_yxyz/_loop_size_xyz)];
        pose(3, 0) = _displacements[int(pose_index%_loop_size_pyxyz%_loop_size_yxyz%_loop_size_xyz/_loop_size_yz)];
        pose(4, 0) = _displacements[int(pose_index%_loop_size_pyxyz%_loop_size_yxyz%_loop_size_xyz%_loop_size_yz/_loop_size_z)];
        pose(5, 0) = _displacements[int(pose_index%_loop_size_pyxyz%_loop_size_yxyz%_loop_size_xyz%_loop_size_yz%_loop_size_z)];

        return pose;
    }
};


thrust::device_vector<Eigen::Matrix<float, 6, 1> > GeneratePoses(const int& linear_winsize, const float& linear_step, const int& angular_winsize, const float& angular_step)
{
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int linear_space_size = 2*linear_winsize+1;
    thrust::device_vector<int> linear_indices(linear_space_size);
    thrust::sequence(linear_indices.begin(), linear_indices.end());
    cudaDeviceSynchronize();
    thrust::device_vector<float> displacements(linear_space_size);
    thrust::transform(linear_indices.begin(), linear_indices.end(), displacements.begin(), get_unit_pose(linear_winsize, linear_step));
    cudaDeviceSynchronize();

    int angular_space_size = 2*angular_winsize+1;
    thrust::device_vector<int> angular_indices(angular_space_size);
    thrust::sequence(angular_indices.begin(), angular_indices.end());
    cudaDeviceSynchronize();
    thrust::device_vector<float> angles(angular_space_size);
    thrust::transform(angular_indices.begin(), angular_indices.end(), angles.begin(), get_unit_pose(angular_winsize, angular_step));
    cudaDeviceSynchronize();

    int pose_num = int(pow(angular_space_size,3)*pow(linear_space_size, 3));
    thrust::device_vector<Eigen::Matrix<float, 6, 1> > poses(pose_num);

    thrust::device_vector<int> pose_indices(pose_num);
    thrust::sequence(pose_indices.begin(), pose_indices.end());
    cudaDeviceSynchronize();

    int loop_size_rpyxyz = int(pow(angular_space_size,3)*pow(linear_space_size, 3));
    int loop_size_pyxyz = int(pow(angular_space_size,2)*pow(linear_space_size, 3));
    int loop_size_yxyz = int(angular_space_size*pow(linear_space_size, 3));
    int loop_size_xyz = int(pow(linear_space_size, 3));
    int loop_size_yz = int(pow(linear_space_size, 2));
    int loop_size_z = int(linear_space_size);

    thrust::transform(thrust::device, pose_indices.begin(), pose_indices.end(), poses.begin(), get_6dof(loop_size_rpyxyz, loop_size_pyxyz, loop_size_yxyz, loop_size_xyz,
                      loop_size_yz, loop_size_z, thrust::raw_pointer_cast(&angles[0]), thrust::raw_pointer_cast(&displacements[0])));
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate pose:  %3.1f ms \n", time);

    return poses;
}

int GetOptPoseIndex(const std::vector<Eigen::Vector3f>& scan, const std::vector<Eigen::Vector3f>& map,
                    const int& linear_window_size, const float& linear_step_size, const int& angular_window_size, const float& angular_step_size,
                    const float& map_resolution)
{
    thrust::device_vector<Eigen::Matrix<float, 6, 1> > poses = GeneratePoses(linear_window_size, linear_step_size, angular_window_size, angular_step_size);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<Eigen::Matrix4f> transforms(poses.size());
    thrust::transform(thrust::device, poses.begin(), poses.end(), transforms.begin(), get_transform());
    cudaDeviceSynchronize();

    std::cout<<"pose num:"<<transforms.size()<<std::endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate transforms:  %3.1f ms \n", time);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<Eigen::Vector3f> trans_point(poses.size());

    thrust::device_vector<float> score_tile(poses.size());
    thrust::device_vector<float> score_bins(poses.size());
    thrust::fill(thrust::device, score_bins.begin(), score_bins.end(), 0.0);
    cudaDeviceSynchronize();

    int map_size = map.size();
    int scan_size = scan.size();

    thrust::device_vector<Eigen::Vector3f> dev_map = map;
    thrust::sort(thrust::device, dev_map.begin(), dev_map.end(), eigen_less_than());
    cudaDeviceSynchronize();

    for(int i = 0 ; i < scan.size(); i++)
    {
        thrust::transform(thrust::device, transforms.begin(), transforms.end(), trans_point.begin(), point_transform(scan[i], map_resolution));
        cudaDeviceSynchronize();
        thrust::transform(thrust::device, trans_point.begin(), trans_point.end(), score_tile.begin(), match(thrust::raw_pointer_cast(&dev_map[0]), map_size));
        cudaDeviceSynchronize();
        thrust::transform(thrust::device, score_bins.begin(), score_bins.end(), score_tile.begin(), score_bins.begin(), thrust::plus<float>());
        cudaDeviceSynchronize();
    }
    thrust::transform(thrust::device, score_bins.begin(), score_bins.end(), score_bins.begin(), compute_score(scan_size));
    cudaDeviceSynchronize();

    thrust::device_vector<float>::iterator max_element_iter = thrust::max_element(score_bins.begin(), score_bins.end());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to compute opt pose:  %3.1f ms \n", time);

    int opt_pose_idx = max_element_iter - score_bins.begin();

    std::cout<<"opt pose index: "<<opt_pose_idx<<std::endl;
    std::cout<<"opt pose score: "<<score_bins[opt_pose_idx]<<std::endl;

    thrust::host_vector<Eigen::Matrix<float, 6, 1> > host_poses = poses;
    std::cout<<"opt pose: "<<host_poses[opt_pose_idx][0]<<", "<<host_poses[opt_pose_idx][1]<<", "<<host_poses[opt_pose_idx][2]<<", "
                           <<host_poses[opt_pose_idx][3]<<", "<<host_poses[opt_pose_idx][4]<<", "<<host_poses[opt_pose_idx][5]<<std::endl;

    return opt_pose_idx;

}