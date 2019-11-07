#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <bitset>
#include "utils.cuh"

//class Rigid3f
//{
//public:
//    Rigid3f()
//    {
//        _translation = {0, 0, 0};
//        _rotation = {0, 0, 0};
//    }
//    Rigid3f(const Eigen::Vector3f& translation, const Eigen::Vector3f& rotation)
//    {
//        _translation = translation;
//        _rotation = rotation;
//    }
//
//    Eigen::Vector3f _translation;
//    Eigen::Vector3f _rotation;
//};

using KeyType = std::bitset<3 * 32>;
using uint32 = uint32_t;
KeyType IndexToKey(Eigen::Vector3i& index){
    KeyType k_0(static_cast<uint32>(index[0]));
    KeyType k_1(static_cast<uint32>(index[1]));
    KeyType k_2(static_cast<uint32>(index[2]));
    return (k_0 << 2 * 32) | (k_1 << 1 * 32) | k_2;
}

void ReadTxt(std::vector<Eigen::Vector3f>& pcd_point, std::string& path){
    std::ifstream infile;
    std::string line;
    infile.open(path);
    while(std::getline(infile,line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> point;
        while(std::getline(ss,cell,' '))
        {
            point.push_back(std::stod(cell));
        }
        auto p = Eigen::Vector3f(point[0],point[1],point[2]);
        pcd_point.emplace_back(p);
    }
    infile.close();
}

void PcdToMap(std::vector<Eigen::Vector3i>& pcd_map, std::vector<Eigen::Vector3f>& pcd_point)
{
    pcd_map.clear();
    std::unordered_set<KeyType> point_set;
    for(auto it:pcd_point){
        auto p = Eigen::Vector3i(int(std::round(it.x()/0.02)),
                                 int(std::round(it.y()/0.02)),
                                 int(std::round(it.z()/0.02)));
        auto inserted = point_set.insert(IndexToKey(p));
        if(inserted.second){
            pcd_map.push_back(p);
        }
    }
}

void GenerateSearchPose(std::vector<Rigid3f>& pose, int linear_window_size,
                        int angular_window_size, float angular_step_size){
    float resolution = 0.02;
    for (int z = -linear_window_size; z <= linear_window_size; ++z) {
        for (int y = -linear_window_size; y <= linear_window_size; ++y) {
            for (int x = -linear_window_size; x <= linear_window_size; ++x) {
                for (int rz = -angular_window_size; rz <= angular_window_size; ++rz) {
                    for (int ry = -angular_window_size; ry <= angular_window_size; ++ry) {
                        for (int rx = -angular_window_size; rx <= angular_window_size;
                             ++rx) {
                            const Eigen::Vector3f angle_axis(rx * angular_step_size,
                                                             ry * angular_step_size,
                                                             rz * angular_step_size);
                            pose.emplace_back(
                                    Eigen::Vector3f(x * resolution, y * resolution,
                                                    z * resolution),
                                    angle_axis);
                        }
                    }
                }
            }
        }
    }
}

Eigen::Matrix4f RigidToMatrix(const Rigid3f& trans)
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

    Eigen::Vector3f xyz_vec = trans._translation;
    Eigen::Vector3f rpy_vec = trans._rotation;

    Eigen::Matrix3f rotation = Eigen::Matrix3f(Eigen::AngleAxisf(rpy_vec[0], Eigen::Vector3f::UnitZ())
                                              *Eigen::AngleAxisf(rpy_vec[1], Eigen::Vector3f::UnitY())
                                              *Eigen::AngleAxisf(rpy_vec[2], Eigen::Vector3f::UnitX()));
    transform.block<3,3>(0, 0) = rotation;
    transform.block<3,1>(0, 3) = xyz_vec;

    return transform;
}




//bool operator==(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
//{
//    return fabs(lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&fabs(lhs[2]-rhs[2])<1e-6;
//}
//
//bool operator>(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
//{
//    return (lhs[0]>rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]>rhs[1])||((lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]>rhs[2]);
//}
//
//bool operator<(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
//{
//    return (lhs[0]<rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]<rhs[1])||((lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]<rhs[2]);
//}
//
//struct eigen_compare
//{
//    bool operator()(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
//    {
//        return (lhs[0]<rhs[0])||(fabs(lhs[0]-rhs[0])<1e-6&&lhs[1]<rhs[1])||((lhs[0]-rhs[0])<1e-6&&fabs(lhs[1]-rhs[1])<1e-6&&lhs[2]<rhs[2]);
//    }
//};
//
//int BinarySearchRecursive(std::vector<Eigen::Vector3f> points, int low, int high, Eigen::Vector3f point)
//{
//    if (low > high)
//        return -1;
//    int mid = low + (high - low) / 2;
//    if (points[mid] == point)
//        return mid;
//    else if (points[mid] > point)
//        return BinarySearchRecursive(points, low, mid - 1, point);
//    else
//        return BinarySearchRecursive(points, mid + 1, high, point);
//}




int main(){
    std::string submap_pcd = "../map.txt";
    std::string target_pcd = "../scan.txt";
    std::vector<Eigen::Vector3f> submap_point;
    std::vector<Eigen::Vector3i> submap;
    std::vector<Eigen::Vector3f> target_point;
    std::cout<<"read txt"<<std::endl;
    ReadTxt(submap_point, submap_pcd);
    ReadTxt(target_point, target_pcd);
    std::cout<<"pcd to map"<<std::endl;
    PcdToMap(submap, submap_point);

    int linear_window_size = int(std::round(0.04/0.02));
    float angular_step_size = 0.01;
    int angular_window_size = int(std::round(0.04/angular_step_size));
    std::vector<Rigid3f> pose;

//    std::cout<<"GenerateSearchPose"<<std::endl;
//    clock_t start = clock();
//    GenerateSearchPose(pose, linear_window_size, angular_window_size, angular_step_size);
//    std::cout<<sizeof(Eigen::Vector3i) * submap.size()<<std::endl;

//    std::cout<<submap.size()<<std::endl;
//    std::cout<<target_point.size()<<std::endl;
//    std::cout<<pose.size()<<std::endl;

    GetOptPoseIndex(submap_point, submap_point, pose);

//    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
//
//    std::vector<Eigen::Vector3f> trans_point(1000);
//    clock_t start = clock();
//    for(int i = 0 ; i < submap.size(); i++)
//    {
//        for(int j = 0 ; j < 1000; j++)
//        {
//            trans_point[j] = submap[i][0] * rotation.col(0) / 0.02 +submap[i][1] * rotation.col(1) / 0.02 +submap[i][2] * rotation.col(2) / 0.02;
//        }
//    }
//    clock_t during = clock() - start;
//    std::cout<<"CPU: "<<double(during / 1000.0)<<std::endl;




//    std::sort(submap_point.begin(), submap_point.end(), eigen_compare());
//
//    for(auto point:submap_point)
//    {
//        int idx = BinarySearchRecursive(submap_point, 0, submap_point.size()-1, point);
//        printf("%f, %f, %f: %d\n", point[0], point[1], point[2], idx);
//    }


    return 0;
}