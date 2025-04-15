# About
This repository contains a script that compares time consumption and memory usage between Open3d and PCLs voxelization and clustering methods.

The script takes a point cloud `.pcd` data type, and performs voxelization and clustering with both libraries. It measures the execution time and the memory consuption, and compares the results.

# Build the project
```
mkdir build && cd build
cmake ..
make
```

# Run the benchmark
```
./point_cloud_benchmark <path_to_pcd_data>
```

# Visualization
There is a script to visualize the results from the benchmark execution. For this, store the resuls on a .txt file
```
./point_cloud_benchmark point_cloud.pcd > benchmark_results.txt
```
and run the visualization script:
```
python3 benchmark_visualization.py benchmark_results.txt --output results
```
# Example

There are some result examples under `benchmark_results`   

![results_results](https://github.com/user-attachments/assets/b8a1d5ac-66c5-4c69-acf0-209dc936fab6)
![results_ratio](https://github.com/user-attachments/assets/b7a64166-5562-42e1-a813-998a2a52438e)
![results_clustering](https://github.com/user-attachments/assets/73ba7b36-d085-4363-b77e-bf0d56ae1c24)
