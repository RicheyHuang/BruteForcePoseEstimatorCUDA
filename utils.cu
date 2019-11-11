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
//                    -sinf(beta),                 sinf(gamma)*cosf(beta),                                        cosf(gamma)*cosf(beta),                                     z,
//                     0.0,                        0.0,                                                           0.0,                                                        1.0;

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
        transformed_point << roundf(homo_transformed_point[0]/_resolution), roundf(homo_transformed_point[1]/_resolution), roundf(homo_transformed_point[2]/_resolution);
        return transformed_point;
    }
};

struct sort_map_point
{
    __host__ __device__
    bool operator()(const Eigen::Vector4f& lhs, const Eigen::Vector4f& rhs)
    {
        return (lhs[0]<rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]<rhs[1])||(fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]<rhs[2])||(fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6&&lhs[3]<rhs[3]);
    }
};

__host__ __device__ bool operator==(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
{
    return fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6;
}

__host__ __device__ bool operator>(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
{
    return (lhs[0]>rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]>rhs[1])||(fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]>rhs[2]);
}

__host__ __device__ int BinarySearchRecursive(const Eigen::Vector4f* points, int low, int high, Eigen::Vector3f point)
{
    if (low > high)
        return -1;
    int mid = low + (high - low) / 2;

    Eigen::Vector3f mid_point;
    mid_point << points[mid][0], points[mid][1], points[mid][2];

    if (mid_point == point)
        return mid;
    else if (mid_point > point)
        return BinarySearchRecursive(points, low, mid - 1, point);
    else
        return BinarySearchRecursive(points, mid + 1, high, point);
}

struct match
{
    const Eigen::Vector4f* _map;
    const int _size;

    explicit match(Eigen::Vector4f* map, int size):_map(map), _size(size){}

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
            return _map[idx][3];
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

struct get_pose
{
    const int _offset;
    const float _resolution;
    const float _init_pose;
    explicit get_pose(const int& offset, const float& resolution, const float& init_pose):_offset(offset), _resolution(resolution), _init_pose(init_pose){}

    __host__ __device__
    float operator()(int index)
    {
        return float((index-_offset)*_resolution+_init_pose);
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


    const float* _roll_angles;
    const float* _pitch_angles;
    const float* _yaw_angles;
    const float* _x_displacements;
    const float* _y_displacements;
    const float* _z_displacements;


    explicit get_6dof(const int& loop_size_rpyxyz,
                      const int& loop_size_pyxyz,
                      const int& loop_size_yxyz,
                      const int& loop_size_xyz,
                      const int& loop_size_yz,
                      const int& loop_size_z,
                      const float* roll_angles,
                      const float* pitch_angles,
                      const float* yaw_angles,
                      const float* x_displacements,
                      const float* y_displacements,
                      const float* z_displacements
                      ):
                      _loop_size_rpyxyz(loop_size_rpyxyz),
                      _loop_size_pyxyz(loop_size_pyxyz),
                      _loop_size_yxyz(loop_size_yxyz),
                      _loop_size_xyz(loop_size_xyz),
                      _loop_size_yz(loop_size_yz),
                      _loop_size_z(loop_size_z),
                      _roll_angles(roll_angles),
                      _pitch_angles(pitch_angles),
                      _yaw_angles(yaw_angles),
                      _x_displacements(x_displacements),
                      _y_displacements(y_displacements),
                      _z_displacements(z_displacements)
                      {}

    __host__ __device__
    Eigen::Matrix<float, 6, 1> operator()(int pose_index)
    {
        Eigen::Matrix<float, 6, 1> pose;

        pose(0, 0) = _roll_angles[int(pose_index/_loop_size_pyxyz)];
        pose(1, 0) = _pitch_angles[int(pose_index%_loop_size_pyxyz/_loop_size_yxyz)];
        pose(2, 0) = _yaw_angles[int(pose_index%_loop_size_pyxyz%_loop_size_yxyz/_loop_size_xyz)];
        pose(3, 0) = _x_displacements[int(pose_index%_loop_size_pyxyz%_loop_size_yxyz%_loop_size_xyz/_loop_size_yz)];
        pose(4, 0) = _y_displacements[int(pose_index%_loop_size_pyxyz%_loop_size_yxyz%_loop_size_xyz%_loop_size_yz/_loop_size_z)];
        pose(5, 0) = _z_displacements[int(pose_index%_loop_size_pyxyz%_loop_size_yxyz%_loop_size_xyz%_loop_size_yz%_loop_size_z)];

        return pose;
    }
};

thrust::device_vector<Eigen::Matrix<float, 6, 1> > GeneratePoses(const Eigen::Vector3f& angular_init_pose, const int& angular_winsize, const float& angular_step, const Eigen::Vector3f& linear_init_pose, const int& linear_winsize, const float& linear_step)
{
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int angular_space_size = 2*angular_winsize+1;
    thrust::device_vector<int> angular_indices(angular_space_size);
    thrust::sequence(angular_indices.begin(), angular_indices.end());
    cudaDeviceSynchronize();

    float roll = angular_init_pose[0];
    thrust::device_vector<float> roll_angles(angular_space_size);
    thrust::transform(angular_indices.begin(), angular_indices.end(), roll_angles.begin(), get_pose(angular_winsize, angular_step, roll));
    cudaDeviceSynchronize();

    float pitch = angular_init_pose[1];
    thrust::device_vector<float> pitch_angles(angular_space_size);
    thrust::transform(angular_indices.begin(), angular_indices.end(), pitch_angles.begin(), get_pose(angular_winsize, angular_step, pitch));
    cudaDeviceSynchronize();

    float yaw = angular_init_pose[2];
    thrust::device_vector<float> yaw_angles(angular_space_size);
    thrust::transform(angular_indices.begin(), angular_indices.end(), yaw_angles.begin(), get_pose(angular_winsize, angular_step, yaw));
    cudaDeviceSynchronize();


    int linear_space_size = 2*linear_winsize+1;
    thrust::device_vector<int> linear_indices(linear_space_size);
    thrust::sequence(linear_indices.begin(), linear_indices.end());
    cudaDeviceSynchronize();

    float x = linear_init_pose[0];
    thrust::device_vector<float> x_displacements(linear_space_size);
    thrust::transform(linear_indices.begin(), linear_indices.end(), x_displacements.begin(), get_pose(linear_winsize, linear_step, x));
    cudaDeviceSynchronize();

    float y = linear_init_pose[1];
    thrust::device_vector<float> y_displacements(linear_space_size);
    thrust::transform(linear_indices.begin(), linear_indices.end(), y_displacements.begin(), get_pose(linear_winsize, linear_step, y));
    cudaDeviceSynchronize();

    float z = linear_init_pose[2];
    thrust::device_vector<float> z_displacements(linear_space_size);
    thrust::transform(linear_indices.begin(), linear_indices.end(), z_displacements.begin(), get_pose(linear_winsize, linear_step, z));
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

    thrust::transform(thrust::device, pose_indices.begin(), pose_indices.end(), poses.begin(),
             get_6dof(loop_size_rpyxyz, loop_size_pyxyz, loop_size_yxyz, loop_size_xyz, loop_size_yz, loop_size_z,
                      thrust::raw_pointer_cast(&roll_angles[0]),
                      thrust::raw_pointer_cast(&pitch_angles[0]),
                      thrust::raw_pointer_cast(&yaw_angles[0]),
                      thrust::raw_pointer_cast(&x_displacements[0]),
                      thrust::raw_pointer_cast(&y_displacements[0]),
                      thrust::raw_pointer_cast(&z_displacements[0])
                     ));
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate pose: %3.1f ms \n", time);

    return poses;
}

struct compute_point_score:public thrust::unary_function<Eigen::Vector3f, float> // <arg, result>
{
    Eigen::Matrix4f _transform;
    int _map_size;
    float _map_resolution;
    Eigen::Vector4f* _map;

    explicit compute_point_score(Eigen::Matrix4f transform, Eigen::Vector4f* map, int& map_size, float& map_resolution):_transform(transform), _map(map), _map_size(map_size), _map_resolution(map_resolution){}

    __host__ __device__

    float operator()(Eigen::Vector3f& point)
    {
        float score;

        Eigen::Vector4f homo_point;
        homo_point << point[0], point[1], point[2], 1.0;
        Eigen::Vector4f homo_transformed_point =  homo_point[0] * _transform.col(0) + homo_point[1] * _transform.col(1) + homo_point[2] * _transform.col(2) + homo_point[3] * _transform.col(3);
        Eigen::Vector3f transformed_point;
        transformed_point << roundf(homo_transformed_point[0]/_map_resolution), roundf(homo_transformed_point[1]/_map_resolution), roundf(homo_transformed_point[2]/_map_resolution);

        Eigen::Vector3f mid_point;

        if (_map_size <= 0)
        {
            return 0.0;
        }
        else
        {
            int low = 0;
            int high = _map_size - 1;
            while (low <= high)
            {
                int mid = low + (high - low) / 2;
                mid_point << _map[mid][0], _map[mid][1], _map[mid][2];

                if (mid_point == transformed_point)
                {
                    score = _map[mid][3];
                    return score;
                }
                else if (mid_point > transformed_point)
                {
                    high = mid - 1;
                }
                else
                {
                    low = mid + 1;
                }
            }
            return 0.0;
        }
    }
};

struct compute_cloud_score:public thrust::unary_function<Eigen::Matrix4f, float> // <arg, result>
{
    Eigen::Vector3f* _scan;
    int _scan_size;

    Eigen::Vector4f* _map;
    int _map_size;
    float _map_resolution;


    compute_cloud_score(Eigen::Vector3f* scan, int& scan_size, Eigen::Vector4f* map, int& map_size, float& map_resolution):_scan(scan), _scan_size(scan_size), _map(map), _map_size(map_size), _map_resolution(map_resolution){}

    __host__ __device__
    float operator()(const Eigen::Matrix4f& transform)
    {
        thrust::device_ptr<Eigen::Vector3f> dev_scan = thrust::device_pointer_cast(_scan);

        float sum = thrust::transform_reduce(thrust::device, dev_scan, dev_scan+_scan_size, compute_point_score(transform, thrust::raw_pointer_cast(_map), _map_size, _map_resolution), 0.0, thrust::plus<float>());

        float score = float(sum/_scan_size);
        return score;
    }
};


struct find_point
{
    Eigen::Vector4f _tgt;
    explicit find_point(Eigen::Vector4f tgt):_tgt(tgt){}

    __host__ __device__
    bool operator()(const Eigen::Vector4f& src)
    {
        return fabs(src[0]-_tgt[0])<1e-6&&fabs(src[1]-_tgt[1])<1e-6&&fabs(src[2]-_tgt[2])<1e-6;
    }
};

__host__ __device__ bool operator==(const Eigen::Vector4f& lhs, const Eigen::Vector4f& rhs)
{
    return fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6;
}

__host__ __device__ bool operator>(const Eigen::Vector4f& lhs, const Eigen::Vector4f& rhs)
{
    return (lhs[0]>rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]>rhs[1])||(fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]>rhs[2])||(fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6&&lhs[3]>rhs[3]);
}

struct fast_compute_point_score:public thrust::unary_function<Eigen::Vector3f, float> // <arg, result>
{
    Eigen::Matrix4f _transform;
    int _map_size;
    float _map_resolution;
    Eigen::Vector4f* _map;

    explicit fast_compute_point_score(Eigen::Matrix4f transform, Eigen::Vector4f* map, int& map_size, float& map_resolution):_transform(transform), _map(map), _map_size(map_size), _map_resolution(map_resolution){}

    __host__ __device__
    float operator()(Eigen::Vector3f& point)
    {
        float score;

        Eigen::Vector4f homo_point;
        homo_point << point[0], point[1], point[2], 1.0;
        Eigen::Vector4f homo_transformed_point =  homo_point[0] * _transform.col(0) + homo_point[1] * _transform.col(1) + homo_point[2] * _transform.col(2) + homo_point[3] * _transform.col(3);

        Eigen::Vector4f transformed_point;
        transformed_point << roundf(homo_transformed_point[0]/_map_resolution), roundf(homo_transformed_point[1]/_map_resolution), roundf(homo_transformed_point[2]/_map_resolution), roundf(homo_transformed_point[3]/_map_resolution);

        thrust::device_ptr<Eigen::Vector4f> map_ptr = thrust::device_pointer_cast(_map);
        int index = thrust::find_if(thrust::device, map_ptr, map_ptr+_map_size, find_point(transformed_point))-map_ptr;

        return _map[index][3];
    }
};

struct fast_compute_cloud_score:public thrust::unary_function<Eigen::Matrix4f, float> // <arg, result>
{
    Eigen::Vector3f* _scan;
    int _scan_size;

    Eigen::Vector4f* _map;
    int _map_size;
    float _map_resolution;


    fast_compute_cloud_score(Eigen::Vector3f* scan, int& scan_size, Eigen::Vector4f* map, int& map_size, float& map_resolution):_scan(scan), _scan_size(scan_size), _map(map), _map_size(map_size), _map_resolution(map_resolution){}

    __host__ __device__
    float operator()(const Eigen::Matrix4f& transform)
    {
        thrust::device_ptr<Eigen::Vector3f> dev_scan = thrust::device_pointer_cast(_scan);

        float sum = thrust::transform_reduce(thrust::device, dev_scan, dev_scan+_scan_size, fast_compute_point_score(transform, thrust::raw_pointer_cast(_map), _map_size, _map_resolution), 0.0, thrust::plus<float>());

        float score = float(sum/_scan_size);
        return score;
    }
};

void ComputeOptimalPoseV1(const std::vector<Eigen::Vector3f>& scan, const std::vector<Eigen::Vector4f>& map,
                          const Eigen::Vector3f& angular_init_pose, const int& angular_window_size, const float& angular_step_size,
                          const Eigen::Vector3f& linear_init_pose,  const int& linear_window_size,  const float& linear_step_size,
                          const float& map_resolution)
{
    thrust::device_vector<Eigen::Matrix<float, 6, 1> > poses = GeneratePoses(angular_init_pose, angular_window_size, angular_step_size, linear_init_pose, linear_window_size, linear_step_size);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<Eigen::Matrix4f> transforms(poses.size());
    thrust::transform(thrust::device, poses.begin(), poses.end(), transforms.begin(), get_transform());
    cudaDeviceSynchronize();

    std::cout<<"Number of generated poses: "<<transforms.size()<<std::endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate transforms: %3.1f ms \n", time);

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

    thrust::device_vector<Eigen::Vector4f> dev_map = map;
    thrust::sort(thrust::device, dev_map.begin(), dev_map.end(), sort_map_point());
    cudaDeviceSynchronize();


    std::cout<<"Number of points in scan: "<<scan_size<<std::endl;
    std::cout<<"Number of points in map: "<<map_size<<std::endl;

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
    printf("Time to compute optimal pose: %3.1f ms \n", time);

    int opt_pose_idx = max_element_iter - score_bins.begin();

    std::cout<<"Optimal Pose Index: "<<opt_pose_idx<<std::endl;
    std::cout<<"Optimal Pose Score: "<<score_bins[opt_pose_idx]<<std::endl;

    thrust::host_vector<Eigen::Matrix<float, 6, 1> > host_poses = poses;
    std::cout<<"Optimal Pose: (roll)"<<host_poses[opt_pose_idx][0]<<" rad, (pitch)"
                                     <<host_poses[opt_pose_idx][1]<<" rad, (yaw)"
                                     <<host_poses[opt_pose_idx][2]<<" rad, (x)"
                                     <<host_poses[opt_pose_idx][3]<<" m, (y)"
                                     <<host_poses[opt_pose_idx][4]<<" m, (z)"
                                     <<host_poses[opt_pose_idx][5]<<" m"<<std::endl;
}

void ComputeOptimalPoseV2(const std::vector<Eigen::Vector3f>& scan, const std::vector<Eigen::Vector4f>& map,
                          const Eigen::Vector3f& angular_init_pose, const int& angular_window_size, const float& angular_step_size,
                          const Eigen::Vector3f& linear_init_pose,  const int& linear_window_size,  const float& linear_step_size,
                          float& map_resolution)
{
    thrust::device_vector<Eigen::Matrix<float, 6, 1> > poses = GeneratePoses(angular_init_pose, angular_window_size, angular_step_size, linear_init_pose, linear_window_size, linear_step_size);
    int pose_num = poses.size();

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<Eigen::Matrix4f> transforms(pose_num);
    thrust::transform(thrust::device, poses.begin(), poses.end(), transforms.begin(), get_transform());
    cudaDeviceSynchronize();

    std::cout<<"Number of generated poses: "<<transforms.size()<<std::endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate transforms: %3.1f ms \n", time);


    thrust::device_vector<Eigen::Vector3f> dev_scan = scan;
    int scan_size = scan.size();

    thrust::device_vector<Eigen::Vector4f> dev_map = map;
    int map_size = map.size();

    thrust::sort(thrust::device, dev_map.begin(), dev_map.end(), sort_map_point());
    cudaDeviceSynchronize();

    std::cout<<"Number of points in scan: "<<scan_size<<std::endl;
    std::cout<<"Number of points in map: "<<map_size<<std::endl;

//    create thrust vector of thrust vector
//    thrust::device_vector<Eigen::Vector3f> trans_scans[pose_num];

    thrust::device_vector<float> scores(pose_num);
    thrust::transform(thrust::device, transforms.begin(), transforms.end(), scores.begin(), compute_cloud_score(thrust::raw_pointer_cast(dev_scan.data()), scan_size,
                                                                                                                thrust::raw_pointer_cast(dev_map.data()), map_size, map_resolution));
    cudaDeviceSynchronize();

    thrust::device_vector<float>::iterator max_element_iter = thrust::max_element(scores.begin(), scores.end());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to compute optimal pose: %3.1f ms \n", time);

    int opt_pose_idx = max_element_iter - scores.begin();

    std::cout<<"Optimal Pose Index: "<<opt_pose_idx<<std::endl;
    std::cout<<"Optimal Pose Score: "<<scores[opt_pose_idx]<<std::endl;

    thrust::host_vector<Eigen::Matrix<float, 6, 1> > host_poses = poses;
    std::cout<<"Optimal Pose: (roll)"<<host_poses[opt_pose_idx][0]<<" rad, (pitch)"
             <<host_poses[opt_pose_idx][1]<<" rad, (yaw)"
             <<host_poses[opt_pose_idx][2]<<" rad, (x)"
             <<host_poses[opt_pose_idx][3]<<" m, (y)"
             <<host_poses[opt_pose_idx][4]<<" m, (z)"
             <<host_poses[opt_pose_idx][5]<<" m"<<std::endl;
}

void ComputeOptimalPoseTest(const std::vector<Eigen::Vector3f>& scan, const std::vector<Eigen::Vector4f>& map,
                           const Eigen::Vector3f& angular_init_pose, const int& angular_window_size, const float& angular_step_size,
                           const Eigen::Vector3f& linear_init_pose,  const int& linear_window_size,  const float& linear_step_size,
                           float& map_resolution)
{
    thrust::device_vector<Eigen::Matrix<float, 6, 1> > poses = GeneratePoses(angular_init_pose, angular_window_size, angular_step_size, linear_init_pose, linear_window_size, linear_step_size);
    int pose_num = poses.size();

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<Eigen::Matrix4f> transforms(pose_num);
    thrust::transform(thrust::device, poses.begin(), poses.end(), transforms.begin(), get_transform());
    cudaDeviceSynchronize();

    std::cout<<"Number of generated poses: "<<transforms.size()<<std::endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate transforms: %3.1f ms \n", time);


    thrust::device_vector<Eigen::Vector3f> dev_scan = scan;
    int scan_size = scan.size();

    thrust::device_vector<Eigen::Vector4f> dev_map = map;
    int map_size = map.size();

    thrust::sort(thrust::device, dev_map.begin(), dev_map.end(), sort_map_point());
    cudaDeviceSynchronize();

    std::cout<<"Number of points in scan: "<<scan_size<<std::endl;
    std::cout<<"Number of points in map: "<<map_size<<std::endl;

//    create thrust vector of thrust vector
//    thrust::device_vector<Eigen::Vector3f> trans_scans[pose_num];

    thrust::device_vector<float> scores(pose_num);
    thrust::transform(thrust::device, transforms.begin(), transforms.end(), scores.begin(), fast_compute_cloud_score(thrust::raw_pointer_cast(dev_scan.data()), scan_size,
                                                                                                                thrust::raw_pointer_cast(dev_map.data()), map_size, map_resolution));
    cudaDeviceSynchronize();

    thrust::device_vector<float>::iterator max_element_iter = thrust::max_element(scores.begin(), scores.end());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to compute optimal pose: %3.1f ms \n", time);

    int opt_pose_idx = max_element_iter - scores.begin();

    std::cout<<"Optimal Pose Index: "<<opt_pose_idx<<std::endl;
    std::cout<<"Optimal Pose Score: "<<scores[opt_pose_idx]<<std::endl;

    thrust::host_vector<Eigen::Matrix<float, 6, 1> > host_poses = poses;
    std::cout<<"Optimal Pose: (roll)"<<host_poses[opt_pose_idx][0]<<" rad, (pitch)"
             <<host_poses[opt_pose_idx][1]<<" rad, (yaw)"
             <<host_poses[opt_pose_idx][2]<<" rad, (x)"
             <<host_poses[opt_pose_idx][3]<<" m, (y)"
             <<host_poses[opt_pose_idx][4]<<" m, (z)"
             <<host_poses[opt_pose_idx][5]<<" m"<<std::endl;
}