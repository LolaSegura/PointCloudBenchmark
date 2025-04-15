#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <functional>

// PCL includes
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>

// Open3D includes
#include <open3d/Open3D.h>

// Struct to hold benchmark results
struct BenchmarkResult {
    double executionTime; // in milliseconds
    size_t memoryUsage;   // in KB
    int numOutputPoints;  // for voxelization
    int numClusters;      // for clustering
};

// Helper function to measure memory usage
size_t getCurrentMemoryUsage() {
    // Note: This is a Linux-specific implementation
    // For Windows or other platforms, you'll need to adapt this
    FILE* file = fopen("/proc/self/statm", "r");
    if (file == NULL)
        return 0;
    
    unsigned long size;
    fscanf(file, "%lu", &size);
    fclose(file);
    
    return size * getpagesize() / 1024; // Convert to KB
}

// PCL Benchmarks
BenchmarkResult benchmarkPCLVoxelization(const std::string& filename, float leafSize) {
    BenchmarkResult result;
    
    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        std::cerr << "Failed to load PCL point cloud: " << filename << std::endl;
        return result;
    }
    
    std::cout << "Loaded PCL cloud with " << cloud->size() << " points." << std::endl;
    
    // Start timing and memory measurement
    size_t memBefore = getCurrentMemoryUsage();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create voxel grid filter
    pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;
    voxelGrid.setInputCloud(cloud);
    voxelGrid.setLeafSize(leafSize, leafSize, leafSize);
    
    // Apply voxelization
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
    voxelGrid.filter(*cloudFiltered);
    
    // End timing and measure memory
    auto end = std::chrono::high_resolution_clock::now();
    size_t memAfter = getCurrentMemoryUsage();
    
    // Calculate results
    result.executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    result.memoryUsage = memAfter - memBefore;
    result.numOutputPoints = cloudFiltered->size();
    
    return result;
}

BenchmarkResult benchmarkPCLClustering(const std::string& filename, float clusterTolerance, int minClusterSize, int maxClusterSize) {
    BenchmarkResult result;
    
    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        std::cerr << "Failed to load PCL point cloud: " << filename << std::endl;
        return result;
    }
    
    // Start timing and memory measurement
    size_t memBefore = getCurrentMemoryUsage();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create KdTree for search method
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    
    // Extract clusters
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minClusterSize);
    ec.setMaxClusterSize(maxClusterSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);
    
    // End timing and measure memory
    auto end = std::chrono::high_resolution_clock::now();
    size_t memAfter = getCurrentMemoryUsage();
    
    // Calculate results
    result.executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    result.memoryUsage = memAfter - memBefore;
    result.numClusters = clusterIndices.size();
    
    return result;
}

// Open3D Benchmarks
BenchmarkResult benchmarkOpen3DVoxelization(const std::string& filename, float voxelSize) {
    BenchmarkResult result;
    
    // Load point cloud
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *cloud)) {
        std::cerr << "Failed to load Open3D point cloud: " << filename << std::endl;
        return result;
    }
    
    std::cout << "Loaded Open3D cloud with " << cloud->points_.size() << " points." << std::endl;
    
    // Start timing and memory measurement
    size_t memBefore = getCurrentMemoryUsage();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform voxel downsampling
    auto downsampled = cloud->VoxelDownSample(voxelSize);
    
    // End timing and measure memory
    auto end = std::chrono::high_resolution_clock::now();
    size_t memAfter = getCurrentMemoryUsage();
    
    // Calculate results
    result.executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    result.memoryUsage = memAfter - memBefore;
    result.numOutputPoints = downsampled->points_.size();
    
    return result;
}

BenchmarkResult benchmarkOpen3DClustering(const std::string& filename, float eps, int minPoints) {
    BenchmarkResult result;
    
    // Load point cloud
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *cloud)) {
        std::cerr << "Failed to load Open3D point cloud: " << filename << std::endl;
        return result;
    }
    
    // Start timing and memory measurement
    size_t memBefore = getCurrentMemoryUsage();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform DBSCAN clustering
    auto clusters = cloud->ClusterDBSCAN(eps, minPoints, true);
    
    // End timing and measure memory
    auto end = std::chrono::high_resolution_clock::now();
    size_t memAfter = getCurrentMemoryUsage();
    
    // Find number of clusters (max label + 1, excluding noise which has label -1)
    int maxLabel = -1;
    for (const auto& label : clusters) {
        maxLabel = std::max(maxLabel, label);
    }
    int numClusters = maxLabel + 1;
    
    // Calculate results
    result.executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    result.memoryUsage = memAfter - memBefore;
    result.numClusters = numClusters;
    
    return result;
}

// Utility function to convert PCL pointcloud to Open3D
std::shared_ptr<open3d::geometry::PointCloud> PCLToOpen3D(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pclCloud) {
    auto o3dCloud = std::make_shared<open3d::geometry::PointCloud>();
    
    // Reserve space
    o3dCloud->points_.reserve(pclCloud->size());
    
    // Copy points
    for (const auto& point : pclCloud->points) {
        o3dCloud->points_.push_back(Eigen::Vector3d(point.x, point.y, point.z));
    }
    
    return o3dCloud;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pointcloud.pcd>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    
    // Parameters for voxelization
    float leafSize = 0.1f;  // 10cm voxel size
    
    // Parameters for clustering
    float clusterTolerance = 0.02f;  // 2cm
    int minClusterSize = 100;
    int maxClusterSize = 25000;
    float dbscanEps = 0.3f;  // 30cm
    int dbscanMinPoints = 10;
    
    std::cout << "=== Point Cloud Processing Benchmark ===" << std::endl;
    std::cout << "Input file: " << filename << std::endl;
    std::cout << std::endl;
    
    // Run PCL voxelization benchmark
    std::cout << "Running PCL voxelization (leaf size = " << leafSize << ")..." << std::endl;
    auto pclVoxelResult = benchmarkPCLVoxelization(filename, leafSize);
    std::cout << "  Time: " << pclVoxelResult.executionTime << " ms" << std::endl;
    std::cout << "  Memory: " << pclVoxelResult.memoryUsage << " KB" << std::endl;
    std::cout << "  Output points: " << pclVoxelResult.numOutputPoints << std::endl;
    
    // Run Open3D voxelization benchmark
    std::cout << "\nRunning Open3D voxelization (voxel size = " << leafSize << ")..." << std::endl;
    auto o3dVoxelResult = benchmarkOpen3DVoxelization(filename, leafSize);
    std::cout << "  Time: " << o3dVoxelResult.executionTime << " ms" << std::endl;
    std::cout << "  Memory: " << o3dVoxelResult.memoryUsage << " KB" << std::endl;
    std::cout << "  Output points: " << o3dVoxelResult.numOutputPoints << std::endl;
    
    // Run PCL clustering benchmark
    std::cout << "\nRunning PCL clustering (tolerance = " << clusterTolerance << ")..." << std::endl;
    auto pclClusterResult = benchmarkPCLClustering(
        filename, clusterTolerance, minClusterSize, maxClusterSize);
    std::cout << "  Time: " << pclClusterResult.executionTime << " ms" << std::endl;
    std::cout << "  Memory: " << pclClusterResult.memoryUsage << " KB" << std::endl;
    std::cout << "  Clusters found: " << pclClusterResult.numClusters << std::endl;
    
    // Run Open3D clustering benchmark
    std::cout << "\nRunning Open3D clustering (eps = " << dbscanEps << ")..." << std::endl;
    auto o3dClusterResult = benchmarkOpen3DClustering(filename, dbscanEps, dbscanMinPoints);
    std::cout << "  Time: " << o3dClusterResult.executionTime << " ms" << std::endl;
    std::cout << "  Memory: " << o3dClusterResult.memoryUsage << " KB" << std::endl;
    std::cout << "  Clusters found: " << o3dClusterResult.numClusters << std::endl;
    
    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    if(o3dVoxelResult.executionTime != 0)
    {
        std::cout << "Voxelization speed ratio (PCL/Open3D): "
            << static_cast<double>(pclVoxelResult.executionTime) / o3dVoxelResult.executionTime << std::endl;
    }
    std::cout << "Clustering speed ratio (PCL/Open3D): "
              << static_cast<double>(pclClusterResult.executionTime) / o3dClusterResult.executionTime << std::endl;
    
    return 0;
}
