
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
    float r, g, b, row, col;

    __host__ __device__ Point(float p1, float p2, float p3, float p4, float p5) {
        r = p1;
        g = p2;
        b = p3;
        row = p4;
        col = p5;
    }
	__host__ __device__ float* operator[](int i)  {
		switch(i) {
		case 0: return &this->r;
		case 1: return &this->g;
		case 2: return &this->b;
		case 3: return &this->row;
		case 4: return &this->col;
		default: assert(0);
		}
	}
	__host__ __device__ float distance_squared(Point *other) {
		float delta_squared = 0;
		for (int i = 0; i < 5; i++) {
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

void color_result(const unsigned char *, unsigned char *, int, int, int);
unsigned char * cpu_version(const unsigned char* image_data, int rows, int cols, float radius, float convergence_threshold) {
    unsigned char* result = (unsigned char*)malloc(rows * cols * 3);
    struct kdtree* kd = kd_create(5);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const unsigned char* const base_of_pixel = &image_data[(r * cols + c) * 3];
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
	int max_iters = 0;
	cluster_convergences.reserve(256);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			const unsigned char* const base_of_pixel = &image_data[(r * cols + c) * 3];
			Point centroid(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], r, c);
			int iters = 0;
			while(true) {
				iters++;
				Point new_centroid(0,0,0,0,0);
				struct kdres* near_points = neighbors(kd, centroid, radius);
				Point temp;
				KD_FOR(temp, near_points) {
					for (int i = 0; i < 5; i++) {
						*new_centroid[i] += *temp[i];
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
			if (iters > max_iters) {
				max_iters = iters;
			}
			int cluster_id = -1;
			for (int i = 0; i < cluster_convergences.size(); i++) {
				//two paths to the same center could have converged from opposite directions,
				//so using 2 * convergence_threshold here.
				if (cluster_convergences[i].distance_squared(&centroid) <= 4 * convergence_threshold * convergence_threshold) {
					//this point converges to a centroid that has already been seen before
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
			result[(r * cols + c) * 3] = cluster_id;
		}
	}
	fprintf(stderr, "Max iters in CPU version was %d\n", max_iters);
	
	//now compute average rgb for each cluster
	color_result(image_data, result, cluster_convergences.size(), rows, cols);
	return result;
}

void color_result(const unsigned char *image_data, unsigned char *result, int num_clusters, int rows, int cols) {
	std::vector<long long> average_rs(num_clusters),
							average_gs(num_clusters),
							average_bs(num_clusters);
	std::vector<int> cluster_sizes(num_clusters);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int cluster_num = result[(r * cols + c) * 3];
			cluster_sizes[cluster_num]++;
			const unsigned char * const base_of_pixel = &image_data[(r * cols + c) * 3];
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
			char unsigned * const base_of_result_pixel = &result[(r * cols + c) * 3];
			int cluster_num = base_of_result_pixel[0];
			base_of_result_pixel[0] = average_rs[cluster_num];
			base_of_result_pixel[1] = average_gs[cluster_num];
			base_of_result_pixel[2] = average_bs[cluster_num];
		}
	}
}



__host__ __device__ void naive_kernel_internals(const unsigned char* image, int rows, int cols, int r, int c, float radius, Point* centroids, float* deltas) {
	if (r >= rows || c >= cols) {
		return;
	}
	//if (deltas[r * cols + c] < 0.1f) {
	//	//HOST ONLY
	//	return;
	//}
	Point new_centroid(0, 0, 0, 0, 0);
	int num_neighbors = 0;
	Point* this_centroid = &centroids[r * cols + c];
	const float radius_squared = radius * radius;
	for (int d_r = floorf(-radius) - 1; d_r <= ceilf(radius) + 1; d_r++) {
		//float limit = sqrtf(radius * radius - d_r * d_r);
		//for (int d_c = floorf(-limit); d_c <= ceilf(limit); d_c++) {
		for (int d_c = floorf(-radius) - 1; d_c <= ceilf(radius) + 1; d_c++) {
			if (d_r * d_r + d_c * d_c > radius_squared) continue;
			int search_r = floorf(this_centroid->row + d_r), search_c = floorf(this_centroid->col + d_c);
			if (search_r < 0 || search_r >= rows || search_c < 0 || search_c >= cols) {
				continue;
			}
			const unsigned char* const base_of_pixel = &image[(search_r * cols + search_c) * 3];
			Point potential_neighbor(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], search_r, search_c);
			float distance_squared = potential_neighbor.distance_squared(this_centroid);
			if (distance_squared <= radius_squared) {
				num_neighbors++;
				//fprintf(stderr, "pixel at r=%d, c=%d is a neighbor of centroid r=%f, c=%f\n", search_r, search_c, this_centroid->row, this_centroid->col);
				for (int i = 0; i < 5; i++) {
					*new_centroid[i] += *potential_neighbor[i];
				}
			}
		}
	}
	for (int i = 0; i < 5; i++) {
		*new_centroid[i] /= num_neighbors;
	}
	float distance_squared = new_centroid.distance_squared(this_centroid);
	*this_centroid = new_centroid;
	/*if (deltas[r * cols + c] > distance_squared) {
		fprintf(stderr, "delta for r=%d, c=%d got bigger, was %f, now is %f\n", r, c, deltas[r * cols + c], distance_squared);
	}*/
	deltas[r * cols + c] = distance_squared;
}

__global__ void naive_kernel(const unsigned char *image, int rows, int cols, float radius, Point *centroids, float *deltas) {
	int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
	if (r >= rows || c >= cols) {
		return;
	}
	//if (deltas[r * cols + c] < 0.1f) {
	//	//HOST ONLY
	//	return;
	//}
	Point new_centroid(0, 0, 0, 0, 0);
	int num_neighbors = 0;
	Point* this_centroid = &centroids[r * cols + c];
	const float radius_squared = radius * radius;
	for (int d_r = floorf(-radius) - 1; d_r <= ceilf(radius) + 1; d_r++) {
		//float limit = sqrtf(radius * radius - d_r * d_r);
		//for (int d_c = floorf(-limit); d_c <= ceilf(limit); d_c++) {
		for (int d_c = floorf(-radius) - 1; d_c <= ceilf(radius) + 1; d_c++) {
			if (d_r * d_r + d_c * d_c > radius_squared) continue;
			int search_r = floorf(this_centroid->row + d_r), search_c = floorf(this_centroid->col + d_c);
			if (search_r < 0 || search_r >= rows || search_c < 0 || search_c >= cols) {
				continue;
			}
			const unsigned char* const base_of_pixel = &image[(search_r * cols + search_c) * 3];
			Point potential_neighbor(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], search_r, search_c);
			float distance_squared = potential_neighbor.distance_squared(this_centroid);
			if (distance_squared <= radius_squared) {
				num_neighbors++;
				//fprintf(stderr, "pixel at r=%d, c=%d is a neighbor of centroid r=%f, c=%f\n", search_r, search_c, this_centroid->row, this_centroid->col);
				for (int i = 0; i < 5; i++) {
					*new_centroid[i] += *potential_neighbor[i];
				}
			}
		}
	}
	for (int i = 0; i < 5; i++) {
		*new_centroid[i] /= num_neighbors;
	}
	float distance_squared = new_centroid.distance_squared(this_centroid);
	*this_centroid = new_centroid;
	/*if (deltas[r * cols + c] > distance_squared) {
		fprintf(stderr, "delta for r=%d, c=%d got bigger, was %f, now is %f\n", r, c, deltas[r * cols + c], distance_squared);
	}*/
	deltas[r * cols + c] = distance_squared;
}

unsigned char* sequential_gpu_version(const unsigned char* image_data, int rows, int cols, float radius, float convergence_threshold) {
	unsigned char *result = (unsigned char*)malloc(rows * cols * 3);
	Point *centroids = (Point*)malloc(rows * cols * sizeof(Point));
	for (int r = 0; r < rows; r++){
		for (int c = 0; c < cols; c++) {
			const unsigned char* const base_of_pixel = &image_data[(r * cols + c) * 3];
			centroids[r * cols + c] = Point(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], r, c);
		}
	}
	
	float *deltas = (float*)malloc(rows * cols * sizeof(float));
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			deltas[r * cols + c] = INFINITY;
		}
	}
	
	int iters = 0;
	while (true) {
		fprintf(stderr, "starting iter %d\n", iters++);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				naive_kernel_internals(image_data, rows, cols, r, c, radius, centroids, deltas);
			}
		}
		float max_delta = 0.0f;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (deltas[r * cols + c] > max_delta) {
					max_delta = deltas[r * cols + c];
				}
			}
		}
		if (max_delta <= convergence_threshold) {
			break;
		}
	}
	
	std::vector<Point> cluster_convergences;
	cluster_convergences.reserve(256);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int cluster_id = -1;
			for (int i = 0; i < cluster_convergences.size(); i++) {
				if (cluster_convergences[i].distance_squared(&centroids[r * cols + c]) <= 4 * convergence_threshold * convergence_threshold) {
					cluster_id = i;
					break;
				}
			}
			if (cluster_id == -1) {
				if (cluster_convergences.size() == 256) {
					fprintf(stderr, "ERROR: more than 256 clusters identified, try tweaking CONVERGENCE_THRESHOLD and SEARCH_RADIUS\n");
					assert(0);
				}
				cluster_convergences.push_back(centroids[r * cols + c]);
				cluster_id = cluster_convergences.size() - 1;
			}
			result[(r * cols + c) * 3] = cluster_id;
		}
	}
	color_result(image_data, result, cluster_convergences.size(), rows, cols);
	
	free(deltas);
	free(centroids);
	return result;
}

__global__ void max_in_groups(float* data, int n) {
	__shared__ float block_data[2048];
	unsigned short t_id = threadIdx.x;
	unsigned global_block_start = blockIdx.x * 2048;
	unsigned short local_offset = 2 * t_id;
	//pad out the last block's block_data with zeros
	if (global_block_start + local_offset < n) {
		block_data[local_offset] = data[global_block_start + local_offset];
		if (global_block_start + local_offset + 1 < n) {
			block_data[local_offset + 1] = data[global_block_start + local_offset + 1];
		}
		else {
			block_data[local_offset + 1] = 0.0f;
		}
	}
	else {
		block_data[local_offset] = 0.0f;
		block_data[local_offset + 1] = 0.0f;
	}

	for (unsigned short stride = 1024; stride >= 1; stride >>= 1) {
		__syncthreads();
		if (t_id < stride) {
			block_data[t_id] += max(block_data[t_id + stride], block_data[t_id]);
		}
	}

	__syncthreads();
	if (t_id == 0) {
		data[global_block_start] = block_data[0];
	}
}

unsigned char * naive_GPU_version(const unsigned char *image_data, int rows, int cols, float radius, float convergence_threshold) {
	unsigned char *result = (unsigned char*)malloc(rows * cols * 3);
	cudaError_t err_code = cudaSuccess;
	err_code = cudaSetDevice(0);
	Point *host_centroids = (Point*)malloc(rows * cols * sizeof(Point));
	for (int r = 0; r < rows; r++){
		for (int c = 0; c < cols; c++) {
			const unsigned char* const base_of_pixel = &image_data[(r * cols + c) * 3];
			host_centroids[r * cols + c] = Point(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], r, c);
		}
	}
	Point *dev_centroids = NULL;
	err_code = cudaMalloc((void**)&dev_centroids, rows * cols * sizeof(Point));
	err_code = cudaMemcpy(dev_centroids, host_centroids, rows * cols * sizeof(Point), cudaMemcpyHostToDevice);
	
	float *host_deltas = (float*)malloc(rows * cols * sizeof(float));
	float *dev_deltas = NULL;
	err_code = cudaMalloc((void**)&dev_deltas, rows * cols * sizeof(float));
	
	unsigned char *dev_image = NULL;
	err_code = cudaMalloc((void**)&dev_image, rows * cols * 3);
	err_code = cudaMemcpy(dev_image, image_data, rows * cols * 3, cudaMemcpyHostToDevice);
	
	/*Point* seq_centroids = (Point*)malloc(rows * cols * sizeof(Point));
	float* seq_deltas = (float*)malloc(rows * cols * sizeof(Point));
	{
		memcpy(seq_centroids, host_centroids, rows * cols * sizeof(Point));
	}*/

	int iters = 0;
	while (true) {
		fprintf(stderr, "now starting iter %d\n", iters++);
		//My device (NVIDIA GeForce GTX 1660) has a max of 1024 threads per block
		dim3 block_dims(32, 32);
		dim3 grid_dims((int)ceil(cols / (float) 32), (int)ceil(rows / (float)32));
		naive_kernel<<<grid_dims, block_dims>>>(dev_image, rows, cols, radius, dev_centroids, dev_deltas);
		
		err_code = cudaGetLastError(); //errors from launching the kernel
		err_code = cudaDeviceSynchronize(); //errors that happened during the kernel launch
		
		//err_code = cudaMemcpy(host_deltas, dev_deltas, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
		//load-bearing memcpy, do not remove
		err_code = cudaMemcpy(host_centroids, dev_centroids, rows * cols * sizeof(Point), cudaMemcpyDeviceToHost);
		/*{ //just for comparing against sequential "kernel"
			cudaMemcpy(host_centroids, dev_centroids, rows * cols * sizeof(Point), cudaMemcpyDeviceToHost);
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					naive_kernel_internals(image_data, rows, cols, r, c, radius, seq_centroids, seq_deltas);
					if (memcmp(&host_centroids[r * cols + c], &seq_centroids[r * cols + c], sizeof(Point))) {
						fprintf(stderr, "centroid for r=%d, c=%d does not match\n", r, c);
					}
					if (host_deltas[r * cols + c] != seq_deltas[r * cols + c]) {
						fprintf(stderr, "delta for r=%d, c=%d does not match, seq = %f, dev = %f\n", r, c, seq_deltas[r * cols + c], host_deltas[r * cols + c]);
					}
				}
			}
		}*/
		int blocks_necessary = (int)ceil((float)(rows * cols) / 2048.0f); //1024 is max threads per block
		max_in_groups<<<blocks_necessary, 2048>>>(dev_deltas, rows * cols);
		float max_delta = 0.0f;
		//Maybe it would be faster to make one big array here and just do one memcpy into that
		for (int i = 0; i < rows * cols; i++) {
			float temp;
			cudaMemcpy(&temp, &dev_deltas[i], sizeof(float), cudaMemcpyDeviceToHost);
			if (temp > max_delta) {
				max_delta = temp;
			}
		}

		if (max_delta < convergence_threshold) {
			break;
		}
		
	}
	
	err_code = cudaMemcpy(host_centroids, dev_centroids, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
	
	std::vector<Point> cluster_convergences;
	cluster_convergences.reserve(256);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int cluster_id = -1;
			for (int i = 0; i < cluster_convergences.size(); i++) {
				if (cluster_convergences[i].distance_squared(&host_centroids[r * cols + c]) <= 4 * convergence_threshold * convergence_threshold) {
					cluster_id = i;
					break;
				}
			}
			if (cluster_id == -1) {
				if (cluster_convergences.size() == 256) {
					fprintf(stderr, "ERROR: more than 256 clusters identified, try tweaking CONVERGENCE_THRESHOLD and SEARCH_RADIUS\n");
					assert(0);
				}
				cluster_convergences.push_back(host_centroids[r * cols + c]);
				cluster_id = cluster_convergences.size() - 1;
			}
			result[(r * cols + c) * 3] = cluster_id;
		}
	}
	color_result(image_data, result, cluster_convergences.size(), rows, cols);
	
	free(host_centroids);
	cudaFree(dev_centroids);
	free(host_deltas);
	cudaFree(dev_deltas);
	cudaFree(dev_image);
	return result;
	
}
#define SEARCH_RADIUS 50
#define CONVERGENCE_THRESHOLD 10
int main()
{
    int rows, cols, channels;
    unsigned char* image_data = (unsigned char*)stbi_load("test_images/dapper_lad_smaller.jpg", &cols, &rows, &channels, 3);
    if (!image_data) {
        fprintf(stderr, "Error reading image: %s", stbi_failure_reason());
        return -1;
    }
    unsigned char * cpu_result = cpu_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD);
    stbi_write_png("cpu_output.png", cols, rows, 3, cpu_result, 0);
    free(cpu_result);

	/*unsigned char* sequential_kernel_result = sequential_gpu_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD);
	stbi_write_png("sequential_output.png", cols, rows, 3, sequential_kernel_result, 0);
	free(sequential_kernel_result);*/

	unsigned char * gpu_result = naive_GPU_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD);
	stbi_write_png("gpu_output.png", cols, rows, 3, gpu_result, 0);
	free(gpu_result);
	
	stbi_image_free(image_data);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}