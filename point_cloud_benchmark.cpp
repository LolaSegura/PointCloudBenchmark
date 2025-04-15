#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <sys/resource.h> // For CPU utilization

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
    double cpuUtilization; // in milliseconds of CPU time
};

// Struct to hold clustering parameters
struct ClusteringParameters {
    float pclTolerance;
    float o3dEpsilon;
    int minClusterSize;
    int maxClusterSize;
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

// Helper function to get CPU usage time
double getCPUTime() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // Return the sum of user and system time in milliseconds
        return (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000.0 +
               (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000.0;
    }
    return 0.0;
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
    
    // Start timing, memory and CPU measurement
    size_t memBefore = getCurrentMemoryUsage();
    double cpuTimeBefore = getCPUTime();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create voxel grid filter
    pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;
    voxelGrid.setInputCloud(cloud);
    voxelGrid.setLeafSize(leafSize, leafSize, leafSize);
    
    // Apply voxelization
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
    voxelGrid.filter(*cloudFiltered);
    
    // End timing and measure memory and CPU
    auto end = std::chrono::high_resolution_clock::now();
    size_t memAfter = getCurrentMemoryUsage();
    double cpuTimeAfter = getCPUTime();
    
    // Calculate results
    result.executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    result.memoryUsage = memAfter - memBefore;
    result.numOutputPoints = cloudFiltered->size();
    result.cpuUtilization = cpuTimeAfter - cpuTimeBefore;
    
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
    
    // Start timing, memory and CPU measurement
    size_t memBefore = getCurrentMemoryUsage();
    double cpuTimeBefore = getCPUTime();
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
    
    // End timing and measure memory and CPU
    auto end = std::chrono::high_resolution_clock::now();
    size_t memAfter = getCurrentMemoryUsage();
    double cpuTimeAfter = getCPUTime();
    
    // Calculate results
    result.executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    result.memoryUsage = memAfter - memBefore;
    result.numClusters = clusterIndices.size();
    result.cpuUtilization = cpuTimeAfter - cpuTimeBefore;
    
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
    
    // Start timing, memory and CPU measurement
    size_t memBefore = getCurrentMemoryUsage();
    double cpuTimeBefore = getCPUTime();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform voxel downsampling
    auto downsampled = cloud->VoxelDownSample(voxelSize);
    
    // End timing and measure memory and CPU
    auto end = std::chrono::high_resolution_clock::now();
    size_t memAfter = getCurrentMemoryUsage();
    double cpuTimeAfter = getCPUTime();
    
    // Calculate results
    result.executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    result.memoryUsage = memAfter - memBefore;
    result.numOutputPoints = downsampled->points_.size();
    result.cpuUtilization = cpuTimeAfter - cpuTimeBefore;
    
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
    
    // Start timing, memory and CPU measurement
    size_t memBefore = getCurrentMemoryUsage();
    double cpuTimeBefore = getCPUTime();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform DBSCAN clustering
    auto clusters = cloud->ClusterDBSCAN(eps, minPoints, true);
    
    // End timing and measure memory and CPU
    auto end = std::chrono::high_resolution_clock::now();
    size_t memAfter = getCurrentMemoryUsage();
    double cpuTimeAfter = getCPUTime();
    
    // Find number of clusters (max label + 1, excluding noise which has label -1)
    int maxLabel = -1;
    for (const auto& label : clusters) {
        maxLabel = std::max(maxLabel, label);
    }
    int numClusters = maxLabel + 1;
    
    // Calculate results
    result.executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    result.memoryUsage = memAfter - memBefore;
    result.numClusters = numClusters;
    result.cpuUtilization = cpuTimeAfter - cpuTimeBefore;
    
    return result;
}

// Function to perform both clusterings with different parameters and find similar results
ClusteringParameters findSimilarClusteringParameters(const std::string& filename) {
    std::cout << "\n=== Finding similar clustering parameters between PCL and Open3D ===" << std::endl;

    ClusteringParameters params;
    params.pclTolerance = 0.1f;    // Default values
    params.o3dEpsilon = 0.2f;
    params.minClusterSize = 50;
    params.maxClusterSize = 25000;

    // Load point cloud for PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *pclCloud) == -1) {
        std::cerr << "Failed to load PCL point cloud: " << filename << std::endl;
        return params;
    }

    // Load point cloud for Open3D
    auto o3dCloud = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *o3dCloud)) {
        std::cerr << "Failed to load Open3D point cloud: " << filename << std::endl;
        return params;
    }

    // Parameters to test
    std::vector<float> pclTolerances = {0.02f, 0.05f, 0.1f, 0.2f, 0.3f};
    std::vector<float> o3dEpsilons = {0.05f, 0.1f, 0.2f, 0.3f, 0.5f};
    std::vector<int> minClusterSizes = {10, 20, 50, 100};

    // Best match found
    int bestMatchDiff = std::numeric_limits<int>::max();
    float bestPCLTolerance = 0.0f;
    float bestO3DEpsilon = 0.0f;
    int bestMinClusterSize = 0;
    int bestPCLClusters = 0;
    int bestO3DClusters = 0;

    // Test different combinations
    for (auto tolerance : pclTolerances) {
        for (auto epsilon : o3dEpsilons) {
            for (auto minSize : minClusterSizes) {
                // Run PCL clustering
                pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
                tree->setInputCloud(pclCloud);

                std::vector<pcl::PointIndices> pclClusterIndices;
                pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
                ec.setClusterTolerance(tolerance);
                ec.setMinClusterSize(minSize);
                ec.setMaxClusterSize(pclCloud->size());
                ec.setSearchMethod(tree);
                ec.setInputCloud(pclCloud);
                ec.extract(pclClusterIndices);

                int pclClusters = pclClusterIndices.size();

                // Run Open3D clustering
                auto o3dClusters = o3dCloud->ClusterDBSCAN(epsilon, minSize, true);

                int maxLabel = -1;
                for (const auto& label : o3dClusters) {
                    maxLabel = std::max(maxLabel, label);
                }
                int o3dClusterCount = maxLabel + 1;

                // Calculate difference in cluster counts
                int diff = std::abs(pclClusters - o3dClusterCount);

                // Update best match if closer
                if (diff < bestMatchDiff) {
                    bestMatchDiff = diff;
                    bestPCLTolerance = tolerance;
                    bestO3DEpsilon = epsilon;
                    bestMinClusterSize = minSize;
                    bestPCLClusters = pclClusters;
                    bestO3DClusters = o3dClusterCount;
                }

                std::cout << "Testing - PCL tolerance: " << tolerance
                          << ", O3D epsilon: " << epsilon
                          << ", Min size: " << minSize
                          << " -> PCL: " << pclClusters
                          << ", O3D: " << o3dClusterCount
                          << ", Diff: " << diff << std::endl;
            }
        }
    }

    std::cout << "\nBest matching parameters:" << std::endl;
    std::cout << "PCL Euclidean Cluster Tolerance: " << bestPCLTolerance << std::endl;
    std::cout << "Open3D DBSCAN Epsilon: " << bestO3DEpsilon << std::endl;
    std::cout << "Minimum Cluster Size: " << bestMinClusterSize << std::endl;
    std::cout << "Resulting clusters - PCL: " << bestPCLClusters
              << ", Open3D: " << bestO3DClusters
              << ", Difference: " << bestMatchDiff << std::endl;

    // Update and return the parameters
    params.pclTolerance = bestPCLTolerance;
    params.o3dEpsilon = bestO3DEpsilon;
    params.minClusterSize = bestMinClusterSize;
    params.maxClusterSize = 25000;  // Keep this large by default

    return params;
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

// Function to display help information
void displayHelp(const char* programName) {
    std::cout << "Usage: " << programName << " <pointcloud.pcd> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --find-params           Find similar clustering parameters between PCL and Open3D" << std::endl;
    std::cout << "  --help                  Display this help information" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        displayHelp(argv[0]);
        return 1;
    }
    
    // Check for help flag
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help") {
            displayHelp(argv[0]);
            return 0;
        }
    }

    std::string filename = argv[1];
    
    // Default parameters
    ClusteringParameters params;
    params.pclTolerance = 0.1f;
    params.o3dEpsilon = 0.2f;
    params.minClusterSize = 50;
    params.maxClusterSize = 25000;
    float leafSize = 0.1f;
    
    // Check if we should find parameters
    bool findParams = false;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--find-params") {
            findParams = true;
            break;
        }
    }

      // Find similar parameters if requested
    if (findParams) {
        params = findSimilarClusteringParameters(filename);
    }

    std::cout << "\n=== Point Cloud Processing Benchmark ===" << std::endl;
    std::cout << "Input file: " << filename << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Voxel leaf size: " << leafSize << std::endl;
    std::cout << "  PCL cluster tolerance: " << params.pclTolerance << std::endl;
    std::cout << "  Open3D DBSCAN epsilon: " << params.o3dEpsilon << std::endl;
    std::cout << "  Minimum cluster size: " << params.minClusterSize << std::endl;
    std::cout << "  Maximum cluster size (PCL): " << params.maxClusterSize << std::endl;
    std::cout << std::endl;
    
    // Run PCL voxelization benchmark
    std::cout << "Running PCL voxelization (leaf size = " << leafSize << ")..." << std::endl;
    auto pclVoxelResult = benchmarkPCLVoxelization(filename, leafSize);
    std::cout << "  Time: " << pclVoxelResult.executionTime << " us" << std::endl;
    std::cout << "  CPU Time: " << pclVoxelResult.cpuUtilization << " ms" << std::endl;
    std::cout << "  CPU Utilization: " << (pclVoxelResult.cpuUtilization * 1000 / pclVoxelResult.executionTime * 100)
              << "% (" << pclVoxelResult.cpuUtilization * 1000 << "us CPU / " << pclVoxelResult.executionTime << "us wall)" << std::endl;
    std::cout << "  Memory: " << pclVoxelResult.memoryUsage << " KB" << std::endl;
    std::cout << "  Output points: " << pclVoxelResult.numOutputPoints << std::endl;
    
    // Run Open3D voxelization benchmark
    std::cout << "\nRunning Open3D voxelization (voxel size = " << leafSize << ")..." << std::endl;
    auto o3dVoxelResult = benchmarkOpen3DVoxelization(filename, leafSize);
    std::cout << "  Time: " << o3dVoxelResult.executionTime << " us" << std::endl;
    std::cout << "  CPU Time: " << o3dVoxelResult.cpuUtilization << " ms" << std::endl;
    std::cout << "  CPU Utilization: " << (o3dVoxelResult.cpuUtilization * 1000 / o3dVoxelResult.executionTime * 100)
              << "% (" << o3dVoxelResult.cpuUtilization * 1000 << "us CPU / " << o3dVoxelResult.executionTime << "us wall)" << std::endl;
    std::cout << "  Memory: " << o3dVoxelResult.memoryUsage << " KB" << std::endl;
    std::cout << "  Output points: " << o3dVoxelResult.numOutputPoints << std::endl;
    
    // Run PCL clustering benchmark
    std::cout << "\nRunning PCL clustering (tolerance = " << params.pclTolerance << ")..." << std::endl;
    auto pclClusterResult = benchmarkPCLClustering(
        filename, params.pclTolerance, params.minClusterSize, params.maxClusterSize);
    std::cout << "  Time: " << pclClusterResult.executionTime << " us" << std::endl;
    std::cout << "  CPU Time: " << pclClusterResult.cpuUtilization << " ms" << std::endl;
    std::cout << "  CPU Utilization: " << (pclClusterResult.cpuUtilization * 1000 / pclClusterResult.executionTime * 100)
              << "% (" << pclClusterResult.cpuUtilization * 1000 << "us CPU / " << pclClusterResult.executionTime << "us wall)" << std::endl;
    std::cout << "  Memory: " << pclClusterResult.memoryUsage << " KB" << std::endl;
    std::cout << "  Clusters found: " << pclClusterResult.numClusters << std::endl;
    
    // Run Open3D clustering benchmark
    std::cout << "\nRunning Open3D clustering (eps = " << params.o3dEpsilon << ")..." << std::endl;
    auto o3dClusterResult = benchmarkOpen3DClustering(filename, params.o3dEpsilon, params.minClusterSize);
    std::cout << "  Time: " << o3dClusterResult.executionTime << " us" << std::endl;
    std::cout << "  CPU Time: " << o3dClusterResult.cpuUtilization << " ms" << std::endl;
    std::cout << "  CPU Utilization: " << (o3dClusterResult.cpuUtilization * 1000 / o3dClusterResult.executionTime * 100)
              << "% (" << o3dClusterResult.cpuUtilization * 1000 << "us CPU / " << o3dClusterResult.executionTime << "us wall)" << std::endl;
    std::cout << "  Memory: " << o3dClusterResult.memoryUsage << " KB" << std::endl;
    std::cout << "  Clusters found: " << o3dClusterResult.numClusters << std::endl;
    
    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    if(o3dVoxelResult.executionTime != 0)
    {
        std::cout << "Voxelization speed ratio (PCL/Open3D): "
            << static_cast<double>(pclVoxelResult.executionTime) / o3dVoxelResult.executionTime << std::endl;
        std::cout << "Voxelization CPU utilization - PCL: "
            << (pclVoxelResult.cpuUtilization * 1000 / pclVoxelResult.executionTime * 100) << "%, Open3D: "
            << (o3dVoxelResult.cpuUtilization * 1000 / o3dVoxelResult.executionTime * 100) << "%" << std::endl;
    }
    std::cout << "Clustering speed ratio (PCL/Open3D): "
              << static_cast<double>(pclClusterResult.executionTime) / o3dClusterResult.executionTime << std::endl;
    std::cout << "Clustering CPU utilization - PCL: "
            << (pclClusterResult.cpuUtilization * 1000 / pclClusterResult.executionTime * 100) << "%, Open3D: "
            << (o3dClusterResult.cpuUtilization * 1000 / o3dClusterResult.executionTime * 100) << "%" << std::endl;
    
    return 0;
}
