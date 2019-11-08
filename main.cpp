#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <bitset>
#include "utils.cuh"


void ReadScan(std::vector<Eigen::Vector3f>& scan, std::string& file_path){
    std::ifstream infile;
    std::string line;
    infile.open(file_path);
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
        scan.emplace_back(p);
    }
    infile.close();
}

void ReadMap(std::vector<Eigen::Vector4f>& map, std::string& file_path){
    std::ifstream infile;
    std::string line;
    infile.open(file_path);
    while(std::getline(infile,line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> point;
        while(std::getline(ss,cell,' '))
        {
            point.push_back(std::stod(cell));
        }
        auto p = Eigen::Vector4f(point[0],point[1],point[2],point[3]);
        map.emplace_back(p);
    }
    infile.close();
}


int main(){
    std::string map_file_path = "../map.txt";
    std::string scan_file_path = "../scan.txt";

    std::vector<Eigen::Vector4f> map;
    std::vector<Eigen::Vector3f> scan;

    ReadMap(map, map_file_path);
    ReadScan(scan, scan_file_path);

    std::cout<<"Data loaded."<<std::endl;

    float angular_range = 0.04;
    float angular_step_size = 0.01;
    int angular_window_size = int(std::round(angular_range/angular_step_size));

    float linear_range = 0.04;
    float linear_step_size = 0.02;
    int linear_window_size = int(std::round(linear_range/linear_step_size));


    float map_resolution = 0.02;
    Eigen::Vector3f linear_init_pose = {0, 0, 0};
    Eigen::Vector3f angular_init_pose = {0, 0, 0};

    ComputeOptimalPose(scan, map, angular_init_pose, angular_window_size, angular_step_size, linear_init_pose, linear_window_size, linear_step_size, map_resolution);

    return 0;
}