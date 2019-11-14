#include "utils.cuh"


const float NULL_ODDS = 10.0;
const float ODDS_LIMIT = 5.0;
//V1
struct get_pose
{
    const int _offset;
    const float _resolution;
    const float _init_pose;
    explicit get_pose(const int& offset, const float& resolution, const float& init_pose):
    _offset(offset), _resolution(resolution), _init_pose(init_pose){}

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
thrust::device_vector<Eigen::Matrix<float, 6, 1> > GeneratePoses(const Eigen::Vector3f& angular_init_pose,
        const int& angular_winsize, const float& angular_step, const Eigen::Vector3f& linear_init_pose,
        const int& linear_winsize, const float& linear_step)
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
    thrust::transform(angular_indices.begin(), angular_indices.end(), roll_angles.begin(),
            get_pose(angular_winsize, angular_step, roll));
    cudaDeviceSynchronize();

    float pitch = angular_init_pose[1];
    thrust::device_vector<float> pitch_angles(angular_space_size);
    thrust::transform(angular_indices.begin(), angular_indices.end(), pitch_angles.begin(),
            get_pose(angular_winsize, angular_step, pitch));
    cudaDeviceSynchronize();

    float yaw = angular_init_pose[2];
    thrust::device_vector<float> yaw_angles(angular_space_size);
    thrust::transform(angular_indices.begin(), angular_indices.end(), yaw_angles.begin(),
            get_pose(angular_winsize, angular_step, yaw));
    cudaDeviceSynchronize();


    int linear_space_size = 2*linear_winsize+1;
    thrust::device_vector<int> linear_indices(linear_space_size);
    thrust::sequence(linear_indices.begin(), linear_indices.end());
    cudaDeviceSynchronize();

    float x = linear_init_pose[0];
    thrust::device_vector<float> x_displacements(linear_space_size);
    thrust::transform(linear_indices.begin(), linear_indices.end(), x_displacements.begin(),
            get_pose(linear_winsize, linear_step, x));
    cudaDeviceSynchronize();

    float y = linear_init_pose[1];
    thrust::device_vector<float> y_displacements(linear_space_size);
    thrust::transform(linear_indices.begin(), linear_indices.end(), y_displacements.begin(),
            get_pose(linear_winsize, linear_step, y));
    cudaDeviceSynchronize();

    float z = linear_init_pose[2];
    thrust::device_vector<float> z_displacements(linear_space_size);
    thrust::transform(linear_indices.begin(), linear_indices.end(), z_displacements.begin(),
            get_pose(linear_winsize, linear_step, z));
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
                      get_6dof(loop_size_rpyxyz, loop_size_pyxyz, loop_size_yxyz,
                              loop_size_xyz, loop_size_yz, loop_size_z,
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
//    printf("Time to generate pose: %3.1f ms \n", time);

    return poses;
}


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


struct sort_map_point
{
    __host__ __device__
    bool operator()(const Eigen::Vector4f& lhs, const Eigen::Vector4f& rhs)
    {
        return (lhs[0]<rhs[0])||
               (fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]<rhs[1])||
               (fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]<rhs[2])||
               (fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6&&lhs[3]<rhs[3]);
    }
};

struct point_transform

{
    const Eigen::Vector3f _point;
    const float _resolution;

    explicit point_transform(const Eigen::Vector3f& point, const float& resolution):
            _point(point), _resolution(resolution){}

    __host__ __device__

    Eigen::Vector3f operator()(const Eigen::Matrix4f& transform)
    {
        Eigen::Vector4f homo_point;
        homo_point << _point[0], _point[1], _point[2], 1.0;

        Eigen::Vector4f homo_transformed_point =  homo_point[0] * transform.col(0) +
                                                  homo_point[1] * transform.col(1) +
                                                  homo_point[2] * transform.col(2) +
                                                  homo_point[3] * transform.col(3);

        Eigen::Vector3f transformed_point;
        transformed_point << roundf(homo_transformed_point[0]/_resolution),
                roundf(homo_transformed_point[1]/_resolution),
                roundf(homo_transformed_point[2]/_resolution);
        return transformed_point;

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


//V2
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

//V3
struct assign_value_
{
    float* _map_odds;
    explicit assign_value_(float* map_odds):_map_odds(map_odds){}

    __host__ __device__
    void operator()(Eigen::Vector4f map_element)
    {
        int key = (int(map_element[0])+500)*1000*1000+(int(map_element[1])+500)*1000+(int(map_element[2])+500);
        _map_odds[key] = map_element[3];
    }
};

struct get_key
{
    __host__ __device__
    int operator()(Eigen::Vector4f map_element)
    {
        int key = (int(map_element[0])+500)*1000*1000+(int(map_element[1])+500)*1000+(int(map_element[2])+500);
        return key;
    }
};

struct faster_compute_point_score:public thrust::unary_function<Eigen::Vector3f, float> // <arg, result>
{
    Eigen::Matrix4f _transform;
    int _map_size;
    float _map_resolution;
    float* _map_odds;

    explicit faster_compute_point_score(Eigen::Matrix4f transform, float* map_odds, int& map_size, float& map_resolution):_transform(transform), _map_odds(map_odds), _map_size(map_size), _map_resolution(map_resolution){}


    __host__ __device__
    float operator()(Eigen::Vector3f& point)
    {
        float score;

        Eigen::Vector4f homo_point;
        homo_point << point[0], point[1], point[2], 1.0;
        Eigen::Vector4f homo_transformed_point =  homo_point[0] * _transform.col(0) + homo_point[1] * _transform.col(1) + homo_point[2] * _transform.col(2) + homo_point[3] * _transform.col(3);

        Eigen::Vector3f transformed_point;
        transformed_point << roundf(homo_transformed_point[0]/_map_resolution), roundf(homo_transformed_point[1]/_map_resolution), roundf(homo_transformed_point[2]/_map_resolution);

        int key = (int(transformed_point[0])+500)*1000000+(int(transformed_point[1])+500)*1000+(int(transformed_point[2])+500);

        if(_map_odds[key] > ODDS_LIMIT)
        {
            return 0.0;
        }
        else
        {
            return _map_odds[key];
        }
    }
};
struct faster_compute_cloud_score:public thrust::unary_function<Eigen::Matrix4f, float> // <arg, result>
{
    Eigen::Vector3f* _scan;
    int _scan_size;

    float* _map_odds;
    int _map_size;
    float _map_resolution;

    explicit faster_compute_cloud_score(Eigen::Vector3f* scan, int& scan_size, float* map_odds, int& map_size, float& map_resolution):_scan(scan), _scan_size(scan_size), _map_odds(map_odds), _map_size(map_size), _map_resolution(map_resolution){}

    __host__ __device__
    float operator()(const Eigen::Matrix4f& transform)
    {

        thrust::device_ptr<Eigen::Vector3f> dev_scan = thrust::device_pointer_cast(_scan);

        float sum = thrust::transform_reduce(thrust::device, dev_scan, dev_scan+_scan_size, faster_compute_point_score(transform, thrust::raw_pointer_cast(_map_odds), _map_size, _map_resolution), 0.0, thrust::plus<float>());

        float score = float(sum/_scan_size);
        return score;

    }
};

//V4
struct cal_transform{
    int* _loop_size;
    Eigen::Vector3f _angular_init_pose;
    int _angular_winsize;
    float _angular_step;
    Eigen::Vector3f _linear_init_pose;
    int _linear_winsize;
    float _linear_step;
    explicit cal_transform(int* loop_size,const Eigen::Vector3f& angular_init_pose, const int& angular_winsize, const float& angular_step,
                           const Eigen::Vector3f& linear_init_pose, const int& linear_winsize, const float& linear_step):
                           _loop_size(loop_size),_angular_init_pose(angular_init_pose),_angular_winsize(angular_winsize),_angular_step(angular_step),
                           _linear_init_pose(linear_init_pose),_linear_winsize(linear_winsize),_linear_step(linear_step){}
    __host__ __device__
    Eigen::Matrix4f operator()(const int& pose_seq){
        Eigen::Matrix4f transform;
        int roll = pose_seq/_loop_size[0];
        int pit  = pose_seq%_loop_size[0]/_loop_size[1];
        int yaw = pose_seq%_loop_size[0]%_loop_size[1]/_loop_size[2];
        int x1 = pose_seq%_loop_size[0]%_loop_size[1]%_loop_size[2]/_loop_size[3];
        int y1 = pose_seq%_loop_size[0]%_loop_size[1]%_loop_size[2]%_loop_size[3]/_loop_size[4];
        int z1 = pose_seq%_loop_size[0]%_loop_size[1]%_loop_size[2]%_loop_size[3]%_loop_size[4];

        float alpha = (roll-_angular_winsize)*_angular_step + _angular_init_pose[0];
        float beta = (pit-_angular_winsize)*_angular_step + _angular_init_pose[1];
        float gamma = (yaw-_angular_winsize)*_angular_step + _angular_init_pose[2];
        float x = (x1-_linear_winsize)*_linear_step + _linear_init_pose[0];
        float y = (y1-_linear_winsize)*_linear_step + _linear_init_pose[1];
        float z = (z1-_linear_winsize)*_linear_step + _linear_init_pose[2];

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
thrust::device_vector<Eigen::Matrix4f> GenerateTransform(const Eigen::Vector3f& angular_init_pose, const int& angular_window_size, const float& angular_step_size,
                                                         const Eigen::Vector3f& linear_init_pose, const int& linear_window_size, const float& linear_step_size){
    int angular_space_size = 2*angular_window_size+1;
    int linear_space_size = 2*linear_window_size+1;
    int pose_num = int(pow(angular_space_size,3)*pow(linear_space_size, 3));
    thrust::device_vector<Eigen::Matrix4f> transforms(pose_num);
    int loop_size_rpyxyz = pose_num;
    int loop_size_pyxyz = loop_size_rpyxyz/angular_space_size;
    int loop_size_yxyz = loop_size_pyxyz/angular_space_size;
    int loop_size_xyz = loop_size_yxyz/angular_space_size;
    int loop_size_yz = loop_size_xyz/linear_space_size;
    int loop_size_z = loop_size_yz/linear_space_size;
    thrust::device_vector<int> loop_gap(6);
    loop_gap[0] = loop_size_pyxyz;
    loop_gap[1] = loop_size_yxyz;
    loop_gap[2]=loop_size_xyz;
    loop_gap[3]=loop_size_yz;
    loop_gap[4]=loop_size_z;
    thrust::device_vector<int> pose_seq(pose_num);
    thrust::sequence(pose_seq.begin(), pose_seq.end());
    cudaDeviceSynchronize();
    thrust::transform(thrust::device, pose_seq.begin(), pose_seq.end(), transforms.begin(), cal_transform(
            thrust::raw_pointer_cast(loop_gap.data()),angular_init_pose,angular_window_size,angular_step_size,
            linear_init_pose, linear_window_size, linear_step_size));
    cudaDeviceSynchronize();
    return transforms;
}

struct get_element{
    int x_;
    explicit get_element(int x):x_(x){}

    __host__ __device__
    float operator()(const Eigen::Vector4f& point){
        return int(point[x_]);
    }
};
std::vector<int> MinMaxXYZ(thrust::device_vector<Eigen::Vector4f>& dev_map, int map_size){
    thrust::device_vector<int> point_x(map_size);
    thrust::device_vector<int> point_y(map_size);
    thrust::device_vector<int> point_z(map_size);
    thrust::transform(thrust::device, dev_map.begin(), dev_map.end(), point_x.begin(), get_element(0));
    thrust::transform(thrust::device, dev_map.begin(), dev_map.end(), point_y.begin(), get_element(1));
    thrust::transform(thrust::device, dev_map.begin(), dev_map.end(), point_z.begin(), get_element(2));
    cudaDeviceSynchronize();
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> resx = thrust::minmax_element(thrust::device, point_x.begin(), point_x.end());
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> resy = thrust::minmax_element(thrust::device, point_y.begin(), point_y.end());
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> resz = thrust::minmax_element(thrust::device, point_z.begin(), point_z.end());
    cudaDeviceSynchronize();
//    int min_x = *(thrust::min_element(thrust::device, point_x.begin(), point_x.end()));
//    int max_x = *(thrust::max_element(thrust::device, point_x.begin(), point_x.end()));
//    int min_y = *(thrust::min_element(thrust::device, point_y.begin(), point_y.end()));
//    int max_y = *(thrust::max_element(thrust::device, point_y.begin(), point_y.end()));
//    int min_z = *(thrust::min_element(thrust::device, point_z.begin(), point_z.end()));
//    int max_z = *(thrust::max_element(thrust::device, point_z.begin(), point_z.end()));
    std::vector<int> result;
    result.push_back(*resx.first);
    result.push_back(*resy.first);
    result.push_back(*resz.first);
    result.push_back(*resx.second);
    result.push_back(*resy.second);
    result.push_back(*resz.second);
    return result;
}

struct assign_valueV2_
{

    float* _map_odds;
    int* _offset;
    int* _map_length;
    explicit assign_valueV2_(float* map_odds, int* offset, int* map_length):_map_odds(map_odds),_offset(offset),_map_length(map_length){}


    __host__ __device__
    void operator()(Eigen::Vector4f map_element)
    {
        int key = (int(map_element[0])+_offset[0])*_map_length[1]*_map_length[2]+(int(map_element[1])+_offset[1])*_map_length[2]+(int(map_element[2])+_offset[2]);
        _map_odds[key] = map_element[3];
    }
};


struct faster_compute_point_scoreV2:public thrust::unary_function<Eigen::Vector3f, float> // <arg, result>
{
    Eigen::Matrix4f _transform;
    int _map_size;
    float _map_resolution;
    float* _map_odds;
    int* _offset;
    int* _map_length;
    int _map_odds_size;


    explicit faster_compute_point_scoreV2(Eigen::Matrix4f transform, float* map_odds, int& map_size, float& map_resolution, int* offset, int* map_length, int map_odds_size):
            _transform(transform), _map_odds(map_odds), _map_size(map_size), _map_resolution(map_resolution), _offset(offset), _map_length(map_length),_map_odds_size(map_odds_size){}


    __host__ __device__
    float operator()(Eigen::Vector3f& point)
    {
        float score;

        Eigen::Vector4f homo_point;
        homo_point << point[0], point[1], point[2], 1.0;
        Eigen::Vector4f homo_transformed_point =  homo_point[0] * _transform.col(0) + homo_point[1] * _transform.col(1) + homo_point[2] * _transform.col(2) + homo_point[3] * _transform.col(3);

        Eigen::Vector3f transformed_point;
        transformed_point << roundf(homo_transformed_point[0]/_map_resolution), roundf(homo_transformed_point[1]/_map_resolution), roundf(homo_transformed_point[2]/_map_resolution);

        int tp_x = int(transformed_point[0])+_offset[0];
        int tp_y = int(transformed_point[1])+_offset[1];
        int tp_z = int(transformed_point[2])+_offset[2];
        if(tp_x<0 || tp_x>=_map_length[0] || tp_y<0 || tp_y>=_map_length[1] || tp_z<0 || tp_z>=_map_length[2]   ){
            return 0.0;
        }
        int key = tp_x*_map_length[1]*_map_length[2]+tp_y*_map_length[2]+tp_z;
        if(_map_odds[key] > ODDS_LIMIT)
        {
            return 0.0;
        }
        else
        {
            return _map_odds[key];
        }
    }
};

struct faster_compute_cloud_scoreV2:public thrust::unary_function<Eigen::Matrix4f, float> // <arg, result>
{
    Eigen::Vector3f* _scan;
    int _scan_size;

    float* _map_odds;
    int _map_size;
    float _map_resolution;

    int* _offset;
    int* _map_length;
    int _map_odds_size;

    explicit faster_compute_cloud_scoreV2(Eigen::Vector3f* scan, int& scan_size, float* map_odds, int& map_size, float& map_resolution, int* offset, int* map_length, int map_odds_size):
            _scan(scan), _scan_size(scan_size), _map_odds(map_odds), _map_size(map_size), _map_resolution(map_resolution),_offset(offset),_map_length(map_length),_map_odds_size(map_odds_size){}


    __host__ __device__
    float operator()(const Eigen::Matrix4f& transform)
    {
        thrust::device_ptr<Eigen::Vector3f> dev_scan = thrust::device_pointer_cast(_scan);


//        float sum = thrust::transform_reduce(thrust::device, dev_scan, dev_scan+_scan_size, faster_compute_point_score(transform, thrust::raw_pointer_cast(_map_odds), _map_size, _map_resolution), 0.0, thrust::plus<float>());
        float sum = thrust::transform_reduce(thrust::device, dev_scan, dev_scan+_scan_size, faster_compute_point_scoreV2(
                transform, thrust::raw_pointer_cast(_map_odds), _map_size, _map_resolution,thrust::raw_pointer_cast(_offset),thrust::raw_pointer_cast(_map_length),_map_odds_size), 0.0, thrust::plus<float>());


        float score = float(sum/_scan_size);
        return score;
    }
};



//__host__ __device__ bool operator==(const Eigen::Vector4f& lhs, const Eigen::Vector4f& rhs)
//{
//    return fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6;
//}
//
//__host__ __device__ bool operator>(const Eigen::Vector4f& lhs, const Eigen::Vector4f& rhs)
//{
//    return (lhs[0]>rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]>rhs[1])||(fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]>rhs[2])||(fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6&&lhs[3]>rhs[3]);
//}


void ComputeOptimalPoseV1(const std::vector<Eigen::Vector3f>& scan, const std::vector<Eigen::Vector4f>& map,
                          const Eigen::Vector3f& angular_init_pose, const int& angular_window_size, const float& angular_step_size,
                          const Eigen::Vector3f& linear_init_pose,  const int& linear_window_size,  const float& linear_step_size,
                          const float& map_resolution)

{
    thrust::device_vector<Eigen::Matrix<float, 6, 1> > poses = GeneratePoses(
            angular_init_pose, angular_window_size, angular_step_size,
            linear_init_pose, linear_window_size, linear_step_size);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    thrust::device_vector<Eigen::Matrix4f> transforms(poses.size());
    thrust::transform(thrust::device, poses.begin(), poses.end(), transforms.begin(), get_transform());
    cudaDeviceSynchronize();

//    std::cout<<"Number of generated poses: "<<transforms.size()<<std::endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
//    printf("Time to generate transforms: %3.1f ms \n", time);

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


//    std::cout<<"Number of points in scan: "<<scan_size<<std::endl;
//    std::cout<<"Number of points in map: "<<map_size<<std::endl;

    for(int i = 0 ; i < scan.size(); i++)
    {
        thrust::transform(thrust::device, transforms.begin(), transforms.end(), trans_point.begin(),
                point_transform(scan[i], map_resolution));
        cudaDeviceSynchronize();
        thrust::transform(thrust::device, trans_point.begin(), trans_point.end(), score_tile.begin(),
                match(thrust::raw_pointer_cast(&dev_map[0]), map_size));
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
//    printf("Time to compute optimal pose: %3.1f ms \n", time);

    int opt_pose_idx = max_element_iter - score_bins.begin();

//    std::cout<<"Optimal Pose Index: "<<opt_pose_idx<<std::endl;
    std::cout<<"Optimal Pose Score: "<<score_bins[opt_pose_idx]<<std::endl;

//    thrust::host_vector<Eigen::Matrix<float, 6, 1> > host_poses = poses;
//    std::cout<<"Optimal Pose: (roll)"<<host_poses[opt_pose_idx][0]<<" rad, (pitch)"
//                                     <<host_poses[opt_pose_idx][1]<<" rad, (yaw)"
//                                     <<host_poses[opt_pose_idx][2]<<" rad, (x)"
//                                     <<host_poses[opt_pose_idx][3]<<" m, (y)"
//                                     <<host_poses[opt_pose_idx][4]<<" m, (z)"
//                                     <<host_poses[opt_pose_idx][5]<<" m"<<std::endl;
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

//    std::cout<<"Number of generated poses: "<<transforms.size()<<std::endl;


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
//    printf("Time to generate transforms: %3.1f ms \n", time);


    thrust::device_vector<Eigen::Vector3f> dev_scan = scan;
    int scan_size = scan.size();

    thrust::device_vector<Eigen::Vector4f> dev_map = map;
    int map_size = map.size();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::sort(thrust::device, dev_map.begin(), dev_map.end(), sort_map_point());
    cudaDeviceSynchronize();


//    std::cout<<"Number of points in scan: "<<scan_size<<std::endl;
//    std::cout<<"Number of points in map: "<<map_size<<std::endl;

    thrust::device_vector<float> scores(pose_num);
    thrust::transform(thrust::device, transforms.begin(), transforms.end(), scores.begin(), compute_cloud_score(thrust::raw_pointer_cast(dev_scan.data()), scan_size,
                                                                                                                thrust::raw_pointer_cast(dev_map.data()), map_size, map_resolution));

    cudaDeviceSynchronize();

    thrust::device_vector<float>::iterator max_element_iter = thrust::max_element(scores.begin(), scores.end());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

//    printf("Time to compute optimal pose: %3.1f ms \n", time);

    int opt_pose_idx = max_element_iter - scores.begin();

//    std::cout<<"Optimal Pose Index: "<<opt_pose_idx<<std::endl;
    std::cout<<"Optimal Pose Score: "<<scores[opt_pose_idx]<<std::endl;

//    thrust::host_vector<Eigen::Matrix<float, 6, 1> > host_poses = poses;
//    std::cout<<"Optimal Pose: (roll)"<<host_poses[opt_pose_idx][0]<<" rad, (pitch)"
//             <<host_poses[opt_pose_idx][1]<<" rad, (yaw)"
//             <<host_poses[opt_pose_idx][2]<<" rad, (x)"
//             <<host_poses[opt_pose_idx][3]<<" m, (y)"
//             <<host_poses[opt_pose_idx][4]<<" m, (z)"
//             <<host_poses[opt_pose_idx][5]<<" m"<<std::endl;

}

void ComputeOptimalPoseV3(const std::vector<Eigen::Vector3f>& scan, const std::vector<Eigen::Vector4f>& map,
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

//    std::cout<<"Number of generated poses: "<<transforms.size()<<std::endl;


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

//    printf("Generate transformations: %3.1f ms \n", time);


    thrust::device_vector<Eigen::Vector3f> dev_scan = scan;
    int scan_size = scan.size();
    thrust::device_vector<Eigen::Vector4f> dev_map = map;
    int map_size = map.size();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int map_odds_size = pow(1000,3);
    thrust::device_vector<float> map_odds(map_odds_size);

    thrust::fill(thrust::device, map_odds.begin(), map_odds.end(), NULL_ODDS);

    cudaDeviceSynchronize();

//    method 1
//    thrust::device_vector<int> indices(map.size());
//    thrust::transform(thrust::device, dev_map.begin(), dev_map.end(), indices.begin(), get_key());
//    cudaDeviceSynchronize();
//    thrust::permutation_iterator<thrust::device_vector<float>::iterator, thrust::device_vector<int>::iterator> iter(map_odds.begin(), indices.begin());
//    thrust::transform(thrust::device, dev_map.begin(), dev_map.end(), iter, assign_value());
//    cudaDeviceSynchronize();

//    method 2
    thrust::for_each(thrust::device, dev_map.begin(), dev_map.end(), assign_value_(thrust::raw_pointer_cast(map_odds.data())));
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
//    printf("Generate hashmap: %3.1f ms \n", time);


//    std::cout<<"Number of points in scan: "<<scan_size<<std::endl;
//    std::cout<<"Number of points in map: "<<map_size<<std::endl;


//    create thrust vector of thrust vector
//    thrust::device_vector<Eigen::Vector3f> trans_scans[pose_num];

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<float> scores(pose_num);
    thrust::transform(thrust::device, transforms.begin(), transforms.end(), scores.begin(), faster_compute_cloud_score(thrust::raw_pointer_cast(dev_scan.data()), scan_size,
                                                                                                                       thrust::raw_pointer_cast(map_odds.data()), map_size, map_resolution));
    cudaDeviceSynchronize();
    thrust::device_vector<float>::iterator max_element_iter = thrust::max_element(scores.begin(), scores.end());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

//    printf("Calculate optimal pose: %3.1f ms \n", time);

    int opt_pose_idx = max_element_iter - scores.begin();
//    std::cout<<"Optimal Pose Index: "<<opt_pose_idx<<std::endl;
    std::cout<<"Optimal Pose Score: "<<scores[opt_pose_idx]<<std::endl;

//    thrust::host_vector<Eigen::Matrix<float, 6, 1> > host_poses = poses;
//    std::cout<<"Optimal Pose: (roll)"<<host_poses[opt_pose_idx][0]<<" rad, (pitch)"
//             <<host_poses[opt_pose_idx][1]<<" rad, (yaw)"
//             <<host_poses[opt_pose_idx][2]<<" rad, (x)"
//             <<host_poses[opt_pose_idx][3]<<" m, (y)"
//             <<host_poses[opt_pose_idx][4]<<" m, (z)"
//             <<host_poses[opt_pose_idx][5]<<" m"<<std::endl;
}

void ComputeOptimalPoseV4(const std::vector<Eigen::Vector3f>& scan, const std::vector<Eigen::Vector4f>& map,
                          const Eigen::Vector3f& angular_init_pose, const int& angular_window_size, const float& angular_step_size,
                          const Eigen::Vector3f& linear_init_pose,  const int& linear_window_size,  const float& linear_step_size,
                          float& map_resolution)
{


    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<Eigen::Matrix4f> transforms = GenerateTransform(angular_init_pose, angular_window_size, angular_step_size,
            linear_init_pose, linear_window_size, linear_step_size);
//    std::cout<<"Number of generated poses: "<<transforms.size()<<std::endl;
    int pose_num = transforms.size();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
//    printf("Time to generate transforms: %3.1f ms \n", time);

    thrust::device_vector<Eigen::Vector3f> dev_scan = scan;
    int scan_size = scan.size();

    thrust::device_vector<Eigen::Vector4f> dev_map = map;
    int map_size = map.size();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::vector<int> minmax = MinMaxXYZ(dev_map, map_size);

    thrust::device_vector<int> dev_offset(3);
    dev_offset[0] = -minmax[0];
    dev_offset[1] = -minmax[1];
    dev_offset[2] = -minmax[2];
    thrust::device_vector<int> dev_maplength(3);
    dev_maplength[0] = minmax[3]-minmax[0]+1;
    dev_maplength[1] = minmax[4]-minmax[1]+1;
    dev_maplength[2] = minmax[5]-minmax[2]+1;
    int map_odds_size = dev_maplength[0]*dev_maplength[1]*dev_maplength[2];
    thrust::device_vector<float> map_odds(map_odds_size);
    thrust::fill(thrust::device, map_odds.begin(), map_odds.end(), NULL_ODDS);
    cudaDeviceSynchronize();

    thrust::for_each(thrust::device, dev_map.begin(), dev_map.end(), assign_valueV2_(thrust::raw_pointer_cast(map_odds.data()),
                                                                                     thrust::raw_pointer_cast(dev_offset.data()), thrust::raw_pointer_cast(dev_maplength.data())));
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
//    printf("Time to generate hashmap: %3.1f ms \n", time);

//    std::cout<<"Number of points in scan: "<<scan_size<<std::endl;
//    std::cout<<"Number of points in map: "<<map_size<<std::endl;


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    thrust::device_vector<float> scores(pose_num);
    thrust::transform(thrust::device, transforms.begin(), transforms.end(), scores.begin(), faster_compute_cloud_scoreV2(
            thrust::raw_pointer_cast(dev_scan.data()), scan_size,thrust::raw_pointer_cast(map_odds.data()), map_size, map_resolution,
            thrust::raw_pointer_cast(dev_offset.data()), thrust::raw_pointer_cast(dev_maplength.data()),map_odds_size));
    cudaDeviceSynchronize();

    thrust::device_vector<float>::iterator max_element_iter = thrust::max_element(scores.begin(), scores.end());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

//    printf("Time to compute optimal pose: %3.1f ms \n", time);

    int opt_pose_idx = max_element_iter - scores.begin();
//    std::cout<<"Optimal Pose Index: "<<opt_pose_idx<<std::endl;
    std::cout<<"Optimal Pose Score: "<<scores[opt_pose_idx]<<std::endl;

}



