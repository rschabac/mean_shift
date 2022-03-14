
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "kdtree.h"
struct Point {
    float r, g, b, x, y;

    Point(float p1, float p2, float p3, float p4, float p5) {
        r = p1;
        g = p2;
        b = p3;
        x = p4;
        y = p5;
    }
	float* operator[](int i)  {
		switch(i) {
		case 0: return &this->r;
		case 1: return &this->g;
		case 2: return &this->b;
		case 3: return &this->x;
		case 4: return &this->y;
		default: assert(0);
		}
	}
	float distance_squared(Point *other) {
		float delta_squared = 0;
		for (int i = 5; i < 5; i++) {
			delta_squared += (*(*this)[i] - *(*other)[i]) * (*(*this)[i] - *(*other)[i]);
		}
		return delta_squared;
	}
    //somehow this makes Point be a POD type, which is important because c++ likes to do weird things
    Point() = default;
};
//simplest way to handle errors
void add_point(struct kdtree* kd, Point p) {
    assert(kd_insertf(kd, &p.r, NULL) == 0);
}
struct kdres* neighbors(struct kdtree* kd, Point p, float radius) {
    auto result = kd_nearest_rangef(kd, &p.r, radius);
    assert(result);
    return result;
}
//Still have to call neighbors before, and kd_res_free after this
#define KD_FOR(point, set) for (kd_res_itemf(set, &point.r); !kd_res_end(set); kd_res_next(set), kd_res_itemf(set, &point.r))

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

char * cpu_version(const char* image_data, int rows, int cols, float radius, float convergence_threshold) {
    char* result = (char*)malloc(rows * cols * 3);
    struct kdtree* kd = kd_create(5);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const char* const base_of_pixel = &image_data[(r * rows + cols) * 3];
            add_point(kd, Point(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], r, c));
        }
    }
	
	/*
	for each point, see what it converges to
	if its convergence point is not in the map/vector, it is a new cluster
	set its r value in result to the cluster number
	Using this scheme, max clusters is 256
	
	once all are clustered, find average rgb over each cluster, then update colors appropriately
	*/
	std::vector<Point> cluster_convergences;
	cluster_convergences.reserve(256);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			const char* const base_of_pixel = &image_data[(r * rows + cols) * 3];
			Point centroid(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], r, c);
			while(true) {
				Point new_centroid(0,0,0,0,0);
				struct kdres* near_points = neighbors(kd, centroid, radius);
				Point temp;
				KD_FOR(temp, near_points) {
					for (int i = 0; i < 5; i++) {
						*new_centroid[i] = *temp[i];
					}
				}
				int num_near_points = kd_res_size(near_points);
				kd_res_free(near_points);
				for (int i = 0; i < 5; i++) {
					*new_centroid[i] /= num_near_points;
				}
				float delta_squared = new_centroid.distance_squared(&centroid);
				centroid = new_centroid;
				if (delta_squared <= convergence_threshold * convergence_threshold) {
					break;
				}
			}
			int cluster_id = -1;
			for (int i = 0; i < cluster_convergences.size(); i++) {
				//two paths to the same center could have converged from opposite directions,
				//so using 2 * convergence_threshold here.
				if (cluster_convergences[i].distance_squared(&centroid) < 4 * convergence_threshold * convergence_threshold) {
					//this point conerges to a centroid that has already been seen before
					cluster_id = i;
					break;
				}
			}
			if (cluster_id == -1) {
				if (cluster_convergences.size() == 256) {
					fprintf(stderr, "ERROR: more than 256 clusters identified, try tweaking CONVERGENCE_THRESHOLD and SEARCH_RADIUS\n");
					assert(0);
				}
				cluster_convergences.push_back(centroid);
				cluster_id = cluster_convergences.size() - 1;
			}
			result[(r * rows + cols) * 3] = cluster_id;
		}
	}
	
	//now compute average rgb for each cluster
	const int num_clusters = cluster_convergences.size();
	std::vector<long long> average_rs(num_clusters),
							average_gs(num_clusters),
							average_bs(num_clusters);
	std::vector<int> cluster_sizes(num_clusters);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int cluster_num = result[(r * rows + c) * 3];
			cluster_sizes[cluster_num]++;
			const char * const base_of_pixel = &image_data[(r * rows + c) * 3];
			average_rs[cluster_num] += base_of_pixel[0];
			average_gs[cluster_num] += base_of_pixel[1];
			average_bs[cluster_num] += base_of_pixel[2];
		}
	}
	for (int i = 0; i < num_clusters; i++) {
		average_rs[i] /= cluster_sizes[i];
		average_gs[i] /= cluster_sizes[i];
		average_bs[i] /= cluster_sizes[i];
	}
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			char * const base_of_result_pixel = &result[(r * rows + c) * 3];
			int cluster_num = base_of_result_pixel[0];
			base_of_result_pixel[0] = average_rs[cluster_num];
			base_of_result_pixel[1] = average_gs[cluster_num];
			base_of_result_pixel[2] = average_bs[cluster_num];
		}
	}
	return result;
}
#define SEARCH_RADIUS 100
#define CONVERGENCE_THRESHOLD 0
int main()
{
    int rows, cols, channels;
    char* image_data = (char*)stbi_load("test_images/dapper_lad.jpg", &cols, &rows, &channels, 3);
    if (!image_data) {
        fprintf(stderr, "Error reading image: %s", stbi_failure_reason());
        return -1;
    }
    char * cpu_result = cpu_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD);
    stbi_write_png("output.png", cols, rows, 3, cpu_result, 0);
    free(cpu_result);
    
    stbi_image_free(image_data);
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
