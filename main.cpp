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



int main(){
    std::string map_pcd = "../points/pcd_34.txt";
    std::string scan_pcd = "../points/pcd_34.txt";

    std::vector<Eigen::Vector3f> map;
    std::vector<Eigen::Vector3f> scan;
    std::cout<<"read txt"<<std::endl;
    ReadTxt(map, map_pcd);
    ReadTxt(scan, scan_pcd);

    float linear_step_size = 0.02;
    int linear_window_size = int(std::round(0.04/linear_step_size));
    float angular_step_size = 0.01;
    int angular_window_size = int(std::round(0.04/angular_step_size));
    float map_resolution = 1.0;

    GetOptPoseIndex(scan, map, linear_window_size, linear_step_size, angular_window_size, angular_step_size, map_resolution);

    return 0;
}