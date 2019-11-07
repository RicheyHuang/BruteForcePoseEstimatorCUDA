#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <bitset>
#include "utils.cuh"

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





int main(){
    std::string submap_pcd = "../points/pcd_34.txt";
    std::string target_pcd = "../points/pcd_36.txt";

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

    std::cout<<"GenerateSearchPose"<<std::endl;
    GenerateSearchPose(pose, linear_window_size, angular_window_size, angular_step_size);
    GetOptPoseIndex(submap_point, submap_point, pose);

    return 0;
}