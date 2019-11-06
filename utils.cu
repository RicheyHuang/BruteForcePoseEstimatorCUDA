#include "utils.cuh"



struct point_transform1
{
    const Eigen::Matrix3f _rotation;

    explicit point_transform1(Eigen::Matrix3f rotation):_rotation(rotation){}

    __host__ __device__

    Eigen::Vector3f operator()(const Eigen::Vector3f& dev_point)
    {
        return (dev_point[0] * _rotation.col(0) / 0.02 + dev_point[1] * _rotation.col(1) / 0.02 + dev_point[2] * _rotation.col(2) / 0.02);
    }
};


thrust::device_vector<Eigen::Vector3f> CloudTransform(thrust::device_vector<Eigen::Vector3f> cloud, const Eigen::Matrix3f& rotation)
{
    thrust::device_vector<Eigen::Vector3f> transformed_cloud(cloud.size());
    thrust::transform(thrust::device, cloud.begin(), cloud.end(), transformed_cloud.begin(), point_transform1(rotation));
    return transformed_cloud;
}



struct point_transform
{
    const Eigen::Vector3f _point;

    explicit point_transform(Eigen::Vector3f point):_point(point){}

    __host__ __device__
    Eigen::Vector3f operator()(const Eigen::Matrix3f& rotation)
    {
        return (_point[0] * rotation.col(0) + _point[1] * rotation.col(1) + _point[2] * rotation.col(2));
//        return (_point[0] * rotation.col(0) / 0.02 + _point[1] * rotation.col(1) / 0.02 + _point[2] * rotation.col(2) / 0.02);
    }
};

struct eigen_compare
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
    bool operator()(const Eigen::Vector3f& point)
    {
        int idx = BinarySearchRecursive(_map, 0, _size-1, point);
        printf("%f, %f, %f: %d\n", point[0], point[1], point[2], idx);
        return idx > 0;
    }
};



//struct cloud_transform
//{
//    const thrust::device_vector<Eigen::Vector3f> _cloud;
//
//    explicit cloud_transform(thrust::device_vector<Eigen::Vector3f> cloud):_cloud(cloud){}
//
//    __host__ __device__ thrust::device_vector<Eigen::Vector3f> operator()(const Eigen::Matrix3f& dev_rotation)
//    {
//        thrust::device_vector<Eigen::Vector3f> transformed_cloud(_cloud.size());
//        thrust::transform(thrust::device, _cloud.begin(), _cloud.end(), transformed_cloud.begin(), point_transform(dev_rotation));
//        return transformed_cloud;
//    }
//};
//
//
//thrust::device_vector< thrust::device_vector<Eigen::Vector3f> > MultiCloudsTransform(thrust::device_vector<Eigen::Matrix3f> rotations, const thrust::device_vector<Eigen::Vector3f>& cloud)
//{
//    thrust::device_vector< thrust::device_vector<Eigen::Vector3f> > transformed_clouds(rotations.size());
//    thrust::transform(thrust::device, rotations.begin(), rotations.end(), transformed_clouds.begin(), cloud_transform(cloud));
//    return transformed_clouds;
//}



//struct exp_functor
//{
//    template<typename T>
//    __host__ __device__
//    thrust::complex<T> operator()(const thrust::complex<T> &x)
//    {
//        return exp(x);
//    } // end operator()()
//}; // end make_pair_functor




//__global__ void Kernel(Eigen::Vector3f* dev_submap, Eigen::Vector3f* dev_target, float* dev_scores)
//{
//    int i = blockIdx.x;
//    dev_scores[i] = dev_submap->col(0)[0];
//}





int kernelTest(const std::vector<Eigen::Vector3f>& submap, const std::vector<Eigen::Vector3f>& map)
{
//    thrust::device_vector<Eigen::Vector3f> dev_submap = submap;
//
//    float scores[100];
//    float* dev_scores;
//
//    cudaMalloc(&dev_scores,100* sizeof(float));
//
//    Kernel<<<100,1>>>(thrust::raw_pointer_cast(&dev_submap[0]), thrust::raw_pointer_cast(&dev_submap[0]), dev_scores);
//
//    cudaMemcpy(scores, dev_scores, 100* sizeof(float), cudaMemcpyDeviceToHost);
//
//    cudaFree(dev_scores);
//
//    cudaDeviceSynchronize();
//
//    for(int i = 0; i < 100; i++)
//    {
//        std::cout<<scores[i]<<std::endl;
//    }

    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
//    thrust::device_vector<Eigen::Vector3f> dev_submap = submap;
//    thrust::device_vector<Eigen::Vector3f> dev_trans_submap = CloudTransform(dev_submap, rotation);

    thrust::device_vector<Eigen::Matrix3f> poses(1000);
    thrust::fill(thrust::device, poses.begin(), poses.end(), rotation);

    thrust::device_vector<Eigen::Vector3f> trans_point(1000);
    thrust::device_vector<bool> stat_bins(1000);

    int map_size = map.size();
    thrust::device_vector<Eigen::Vector3f> dev_map = map;

    thrust::sort(thrust::device, dev_map.begin(), dev_map.end(), eigen_compare());


//    thrust::host_vector<Eigen::Vector3f> sorted_map = dev_map;
//    for(int i = 0; i < sorted_map.size(); i++)
//    {
//        std::cout<<sorted_map[i]<<std::endl;
//        std::cout<<std::endl;
//    }


//    thrust::device_ptr<Eigen::Vector3f> dev_map_ptr = &dev_map[0];



//    clock_t start = clock();

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i = 0 ; i < submap.size(); i++)
    {
        thrust::transform(thrust::device, poses.begin(), poses.end(), trans_point.begin(), point_transform(submap[i]));
        cudaDeviceSynchronize();
        thrust::transform(thrust::device, trans_point.begin(), trans_point.end(), stat_bins.begin(), match(thrust::raw_pointer_cast(&dev_map[0]), map_size));
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.1f ms \n", time);

//    clock_t during = clock() - start;
//    std::cout<<"GPU: "<<double(during / 1000.0)<<std::endl;



    return 0;
}