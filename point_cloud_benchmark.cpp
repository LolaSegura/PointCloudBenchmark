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

// Structure to hold all matched parameters
struct MatchingParameters {
    float pclLeafSize;
    float pclClusterTolerance;
    // Common parameters
    float o3dVoxelSize;
    float o3dEpsilon;
    int minClusterSize;
    int maxClusterSize;
};

struct Parameters {
    float pclLeafSize;
    float pclClusterTolerance;
    float o3dVoxelSize;
    float o3dEpsilon;
    // Common parameters
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

BenchmarkResult benchmarkPCLVoxelizationClustering(const std::string& filename, float leafSize, float clusterTolerance, int minClusterSize, int maxClusterSize) {
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

    // Create KdTree for search method
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloudFiltered);

    // Extract clusters
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minClusterSize);
    ec.setMaxClusterSize(maxClusterSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloudFiltered);
    ec.extract(clusterIndices);

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

BenchmarkResult benchmarkOpen3DVoxelizationClustering(const std::string& filename, float voxelSize, float eps, int minPoints)
{
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
    // Perform DBSCAN clustering
    auto clusters = downsampled->ClusterDBSCAN(eps, minPoints, true);

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

// Function to perform both clusterings with different parameters and find similar results
Parameters findSimilarClusteringParameters(const std::string& filename) {
    std::cout << "\n=== Finding similar clustering parameters between PCL and Open3D ===" << std::endl;

    // Default values
    Parameters params;
    params.pclClusterTolerance = 0.1f;
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
    params.pclClusterTolerance = bestPCLTolerance;
    params.o3dEpsilon = bestO3DEpsilon;
    params.minClusterSize = bestMinClusterSize;
    params.maxClusterSize = 25000;  // Keep this large by default

    return params;
}

// Function to find a PCL leaf size that produces similar results to a given Open3D voxel size
void findMatchingPCLLeafSize(const std::string& filename, float referenceO3DVoxelSize, Parameters& params) {
    std::cout << "\n=== Finding PCL leaf size to match Open3D voxel size " << referenceO3DVoxelSize << " ===" << std::endl;

    // Load point cloud for PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *pclCloud) == -1) {
        std::cerr << "Failed to load PCL point cloud: " << filename << std::endl;
        params.pclLeafSize = referenceO3DVoxelSize; // Return reference as default
        return;
    }

    // Load point cloud for Open3D
    auto o3dCloud = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *o3dCloud)) {
        std::cerr << "Failed to load Open3D point cloud: " << filename << std::endl;
        params.pclLeafSize = referenceO3DVoxelSize; // Return reference as default
        return;
    }

    // Create Open3D reference result with the given voxel size
    auto o3dDownsampled = o3dCloud->VoxelDownSample(referenceO3DVoxelSize);
    int o3dPointCount = o3dDownsampled->points_.size();

    std::cout << "Open3D voxel size " << referenceO3DVoxelSize << " produces " << o3dPointCount << " points." << std::endl;

    // Parameters to test - different PCL leaf sizes
    std::vector<float> pclLeafSizes = {
        referenceO3DVoxelSize * 0.5f,
        referenceO3DVoxelSize * 0.75f,
        referenceO3DVoxelSize * 0.9f,
        referenceO3DVoxelSize,
        referenceO3DVoxelSize * 1.1f,
        referenceO3DVoxelSize * 1.25f,
        referenceO3DVoxelSize * 1.5f
    };

    // Best match found
    int bestMatchDiff = std::numeric_limits<int>::max();
    float bestPCLLeafSize = referenceO3DVoxelSize;  // Default to same value
    int bestPCLPoints = 0;

    // Test different PCL leaf sizes
    for (auto leafSize : pclLeafSizes) {
        // Apply PCL voxelization
        pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;
        voxelGrid.setInputCloud(pclCloud);
        voxelGrid.setLeafSize(leafSize, leafSize, leafSize);

        pcl::PointCloud<pcl::PointXYZ>::Ptr pclDownsampled(new pcl::PointCloud<pcl::PointXYZ>);
        voxelGrid.filter(*pclDownsampled);
        int pclPointCount = pclDownsampled->size();

        // Calculate difference in point counts
        int diff = std::abs(pclPointCount - o3dPointCount);

        // Update best match if closer
        if (diff < bestMatchDiff) {
            bestMatchDiff = diff;
            bestPCLLeafSize = leafSize;
            bestPCLPoints = pclPointCount;
        }

        std::cout << "Testing PCL leaf size: " << leafSize
                  << " -> PCL points: " << pclPointCount
                  << ", Diff: " << diff << std::endl;
    }

    std::cout << "\nBest matching PCL leaf size: " << bestPCLLeafSize << std::endl;
    std::cout << "Resulting point counts - PCL: " << bestPCLPoints
              << ", Open3D: " << o3dPointCount
              << ", Difference: " << bestMatchDiff << std::endl;

    params.pclLeafSize = bestPCLLeafSize;
}

// Function to find PCL clustering parameters that match given Open3D parameters
void findMatchingPCLClusteringParams(const std::string& filename, float o3dEpsilon, int minClusterSize, Parameters& params) {
    std::cout << "\n=== Finding PCL clustering params to match Open3D epsilon " << o3dEpsilon
              << ", min size " << minClusterSize << " ===" << std::endl;

    params.o3dEpsilon = o3dEpsilon;
    params.minClusterSize = minClusterSize;
    params.maxClusterSize = 25000;  // Default large value

    // Load point cloud for PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *pclCloud) == -1) {
        std::cerr << "Failed to load PCL point cloud: " << filename << std::endl;
        params.pclClusterTolerance = o3dEpsilon * 0.5f;  // Default estimation
        return;
    }

    // Load point cloud for Open3D
    auto o3dCloud = std::make_shared<open3d::geometry::PointCloud>();
    if (!open3d::io::ReadPointCloud(filename, *o3dCloud)) {
        std::cerr << "Failed to load Open3D point cloud: " << filename << std::endl;
        params.pclClusterTolerance = o3dEpsilon * 0.5f;  // Default estimation
        return;
    }

    // Run Open3D clustering with the given parameters
    auto o3dClusters = o3dCloud->ClusterDBSCAN(o3dEpsilon, minClusterSize, true);

    // Count clusters in Open3D result
    int maxLabel = -1;
    for (const auto& label : o3dClusters) {
        maxLabel = std::max(maxLabel, label);
    }
    int o3dClusterCount = maxLabel + 1;

    std::cout << "Open3D clustering with epsilon " << o3dEpsilon << ", min size " << minClusterSize
              << " produces " << o3dClusterCount << " clusters." << std::endl;

    // Parameters to test - different PCL tolerance values
    std::vector<float> pclTolerances = {
        o3dEpsilon * 0.3f,
        o3dEpsilon * 0.4f,
        o3dEpsilon * 0.5f,
        o3dEpsilon * 0.6f,
        o3dEpsilon * 0.7f,
        o3dEpsilon * 0.8f
    };

    // Best match found
    int bestMatchDiff = std::numeric_limits<int>::max();
    float bestPCLTolerance = o3dEpsilon * 0.5f;  // Default estimate
    int bestPCLClusters = 0;

    // Test different PCL tolerance values
    for (auto tolerance : pclTolerances) {
        // Run PCL clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(pclCloud);

        std::vector<pcl::PointIndices> pclClusterIndices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(tolerance);
        ec.setMinClusterSize(minClusterSize);
        ec.setMaxClusterSize(params.maxClusterSize);
        ec.setSearchMethod(tree);
        ec.setInputCloud(pclCloud);
        ec.extract(pclClusterIndices);

        int pclClusters = pclClusterIndices.size();

        // Calculate difference in cluster counts
        int diff = std::abs(pclClusters - o3dClusterCount);

        // Update best match if closer
        if (diff < bestMatchDiff) {
            bestMatchDiff = diff;
            bestPCLTolerance = tolerance;
            bestPCLClusters = pclClusters;
        }

        std::cout << "Testing PCL tolerance: " << tolerance
                  << " -> PCL clusters: " << pclClusters
                  << ", Diff: " << diff << std::endl;
    }

    std::cout << "\nBest matching PCL cluster tolerance: " << bestPCLTolerance << std::endl;
    std::cout << "Resulting clusters - PCL: " << bestPCLClusters
              << ", Open3D: " << o3dClusterCount
              << ", Difference: " << bestMatchDiff << std::endl;

    params.pclClusterTolerance = bestPCLTolerance;
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

// Function to find PCL parameters that match given Open3D parameters
Parameters findMatchingPCLParameters(const std::string& filename,
                                           float o3dVoxelSize,
                                           float o3dEpsilon,
                                           int minClusterSize,
                                           int maxClusterSize = 25000) {
    Parameters params;
    // Store the reference Open3D parameters
    params.o3dVoxelSize = o3dVoxelSize;
    params.o3dEpsilon = o3dEpsilon;
    params.minClusterSize = minClusterSize;
    params.maxClusterSize = maxClusterSize;

    // Find matching PCL leaf size for voxelization
    findMatchingPCLLeafSize(filename, o3dVoxelSize, params);

    // Find matching PCL clustering parameters
    findMatchingPCLClusteringParams(filename, o3dEpsilon, minClusterSize, params);

    return params;
}
// Function to display help information
void displayHelp(const char* programName) {
    std::cout << "Usage: " << programName << " <pointcloud.pcd> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --compose                 Performs benchmark composing voxelization and clustering operations" << std::endl;
    std::cout << "  --find-params             Find similar clustering parameters between PCL and Open3D" << std::endl;
    std::cout << "  --find-matching-pcl       Find PCL parameters that match given Open3D parameters" << std::endl;
    std::cout << "  --params-only             Only output the found parameters without running benchmarks" << std::endl;
    std::cout << "  --o3d-voxel-size <size>   Open3D voxel size (default: 0.1)" << std::endl;
    std::cout << "  --o3d-epsilon <eps>       Open3D DBSCAN epsilon (default: 0.2)" << std::endl;
    std::cout << "  --min-cluster-size <n>    Minimum cluster size (default: 50)" << std::endl;
    std::cout << "  --max-cluster-size <n>    Maximum cluster size (default: 25000)" << std::endl;
    std::cout << "  --help                    Display this help information" << std::endl;
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

    Parameters params;
    params.pclClusterTolerance = 0.1f;
    params.pclLeafSize = 0.1f;
    params.o3dEpsilon = 0.2f;
    params.o3dVoxelSize = 0.1f;
    params.minClusterSize = 50;
    params.maxClusterSize = 25000;
    

    bool findParams = false;          // Check if we should find matching parameters
    bool findMatchingPCL = false;     // Find PCL params to match given Open3D params
    bool paramsOnly = false;          // Only output params without running benchmarks
    bool compose = false;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--compose") {
            compose = true;
        }
        else if (std::string(argv[i]) == "--find-params") {
            findParams = true;
        }
        else if (std::string(argv[i]) == "--find-matching-pcl") {
            findMatchingPCL = true;
        }
        else if (std::string(argv[i]) == "--params-only") {
            paramsOnly = true;
        }
        else if (std::string(argv[i]) == "--o3d-voxel-size" && i+1 < argc) {
            params.o3dVoxelSize = std::stof(argv[i+1]);
            i++; // Skip the next argument
        }
        else if (std::string(argv[i]) == "--o3d-epsilon" && i+1 < argc) {
            params.o3dEpsilon = std::stof(argv[i+1]);
            i++; // Skip the next argument
        }
        else if (std::string(argv[i]) == "--min-cluster-size" && i+1 < argc) {
            params.minClusterSize = std::stoi(argv[i+1]);
            i++; // Skip the next argument
        }
        else if (std::string(argv[i]) == "--max-cluster-size" && i+1 < argc) {
            params.maxClusterSize = std::stoi(argv[i+1]);
            i++; // Skip the next argument
        }
    }

    // Find similar parameters if requested
    if (findParams) {
        params = findSimilarClusteringParameters(filename);
    }

    // Find matching PCL parameters given Open3D if requested
    if (findMatchingPCL) {
        Parameters matching_params = findMatchingPCLParameters(
            filename, params.o3dVoxelSize, params.o3dEpsilon, params.minClusterSize, params.maxClusterSize);

        params.pclClusterTolerance = matching_params.pclClusterTolerance;
        params.pclLeafSize = matching_params.pclLeafSize;

        // If params-only flag is set, just print the parameters and exit
        if (paramsOnly) {
            std::cout << "\n=== Matching PCL Parameters for Open3D ===" << std::endl;
            std::cout << "PCL leaf size: " << params.pclLeafSize
                      << " (matches Open3D voxel size: " << params.o3dVoxelSize << ")" << std::endl;
            std::cout << "PCL cluster tolerance: " << params.pclClusterTolerance
                      << " (matches Open3D epsilon: " << params.o3dEpsilon << ")" << std::endl;
            std::cout << "Common parameters:" << std::endl;
            std::cout << "  Min cluster size: " << params.minClusterSize << std::endl;
            std::cout << "  Max cluster size: " << params.maxClusterSize << std::endl;
            return 0;
        }
    }

    std::cout << "\n=== Point Cloud Processing Benchmark ===" << std::endl;
    std::cout << "Input file: " << filename << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Voxel leaf size: " << params.pclLeafSize << std::endl;
    std::cout << "  PCL cluster tolerance: " << params.pclClusterTolerance << std::endl;
    std::cout << "  Open3D DBSCAN epsilon: " << params.o3dEpsilon << std::endl;
    std::cout << "  Minimum cluster size: " << params.minClusterSize << std::endl;
    std::cout << "  Maximum cluster size (PCL): " << params.maxClusterSize << std::endl;
    std::cout << std::endl;
    
    if(compose)
    {
        // Run PCL voxelization and clustering benchmark
        std::cout << "Running PCL voxelization and clustering" << std::endl;
        auto pclResult = benchmarkPCLVoxelizationClustering(filename, params.pclLeafSize, params.pclClusterTolerance, params.minClusterSize, params.maxClusterSize);
        std::cout << "  Time: " << pclResult.executionTime << " us" << std::endl;
        std::cout << "  CPU Time: " << pclResult.cpuUtilization << " ms" << std::endl;
        std::cout << "  CPU Utilization: " << (pclResult.cpuUtilization * 1000 / pclResult.executionTime * 100)
                << "% (" << pclResult.cpuUtilization * 1000 << "us CPU / " << pclResult.executionTime << "us wall)" << std::endl;
        std::cout << "  Memory: " << pclResult.memoryUsage << " KB" << std::endl;
        std::cout << "  Output points: " << pclResult.numOutputPoints << std::endl;

        // Run Open3D voxelization and clustering benchmark
        std::cout << "\nRunning Open3D voxelization and clustering" << std::endl;
        auto o3dResult = benchmarkOpen3DVoxelizationClustering(filename, params.o3dVoxelSize, params.o3dEpsilon, params.minClusterSize);
        std::cout << "  Time: " << o3dResult.executionTime << " us" << std::endl;
        std::cout << "  CPU Time: " << o3dResult.cpuUtilization << " ms" << std::endl;
        std::cout << "  CPU Utilization: " << (o3dResult.cpuUtilization * 1000 / o3dResult.executionTime * 100)
                << "% (" << o3dResult.cpuUtilization * 1000 << "us CPU / " << o3dResult.executionTime << "us wall)" << std::endl;
        std::cout << "  Memory: " << o3dResult.memoryUsage << " KB" << std::endl;
        std::cout << "  Output points: " << o3dResult.numOutputPoints << std::endl;

        // Summary
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Speed ratio (PCL/Open3D): "
                << static_cast<double>(pclResult.executionTime) / o3dResult.executionTime << std::endl;
        std::cout << "CPU utilization - PCL: "
                << (pclResult.cpuUtilization * 1000 / pclResult.executionTime * 100) << "%, Open3D: "
                << (o3dResult.cpuUtilization * 1000 / o3dResult.executionTime * 100) << "%" << std::endl;

    } else {
        // Run PCL voxelization benchmark
        std::cout << "Running PCL voxelization (leaf size = " << params.pclLeafSize << ")..." << std::endl;
        auto pclVoxelResult = benchmarkPCLVoxelization(filename, params.pclLeafSize);
        std::cout << "  Time: " << pclVoxelResult.executionTime << " us" << std::endl;
        std::cout << "  CPU Time: " << pclVoxelResult.cpuUtilization << " ms" << std::endl;
        std::cout << "  CPU Utilization: " << (pclVoxelResult.cpuUtilization * 1000 / pclVoxelResult.executionTime * 100)
                << "% (" << pclVoxelResult.cpuUtilization * 1000 << "us CPU / " << pclVoxelResult.executionTime << "us wall)" << std::endl;
        std::cout << "  Memory: " << pclVoxelResult.memoryUsage << " KB" << std::endl;
        std::cout << "  Output points: " << pclVoxelResult.numOutputPoints << std::endl;

        // Run Open3D voxelization benchmark
        std::cout << "\nRunning Open3D voxelization (voxel size = " << params.o3dVoxelSize << ")..." << std::endl;
        auto o3dVoxelResult = benchmarkOpen3DVoxelization(filename, params.o3dVoxelSize);
        std::cout << "  Time: " << o3dVoxelResult.executionTime << " us" << std::endl;
        std::cout << "  CPU Time: " << o3dVoxelResult.cpuUtilization << " ms" << std::endl;
        std::cout << "  CPU Utilization: " << (o3dVoxelResult.cpuUtilization * 1000 / o3dVoxelResult.executionTime * 100)
                << "% (" << o3dVoxelResult.cpuUtilization * 1000 << "us CPU / " << o3dVoxelResult.executionTime << "us wall)" << std::endl;
        std::cout << "  Memory: " << o3dVoxelResult.memoryUsage << " KB" << std::endl;
        std::cout << "  Output points: " << o3dVoxelResult.numOutputPoints << std::endl;

        // Run PCL clustering benchmark
        std::cout << "\nRunning PCL clustering (tolerance = " << params.pclClusterTolerance << ")..." << std::endl;
        auto pclClusterResult = benchmarkPCLClustering(
            filename, params.pclClusterTolerance, params.minClusterSize, params.maxClusterSize);
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
    }
    
    
    return 0;
}
