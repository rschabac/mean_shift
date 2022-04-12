
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
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

void add_point(struct kdtree* kd, Point p) {
	//simplest way to handle errors
	auto result = kd_insertf(kd, &p.r, NULL);
	assert(result == 0);
}
struct kdres* neighbors(struct kdtree* kd, Point p, float radius) {
    auto result = kd_nearest_rangef(kd, &p.r, radius);
    assert(result);
    return result;
}
//Still have to call neighbors before, and kd_res_free after this
#define KD_FOR(point, set) for (kd_res_itemf(set, &point.r); !kd_res_end(set); kd_res_next(set), kd_res_itemf(set, &point.r))

void color_result(const unsigned char *, unsigned char *, int, int, int);
unsigned char * cpu_version(const unsigned char* image_data, int rows, int cols, float radius, float convergence_threshold, bool do_color) {
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
		//printf("now starting r = %d\n", r);
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
	kd_free(kd);
	//fprintf(stderr, "Max iters in CPU version was %d\n", max_iters);
	
	//now compute average rgb for each cluster
	if (do_color) {
		color_result(image_data, result, cluster_convergences.size(), rows, cols);
	}
	return result;
}

unsigned char *cpu_version_with_trajectories(const unsigned char *image_data, int rows, int cols, float radius, float convergence_threshold, bool do_color) {
	unsigned char* result = (unsigned char*)malloc(rows * cols * 3);
    struct kdtree* source_points = kd_create(5);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const unsigned char* const base_of_pixel = &image_data[(r * cols + c) * 3];
            add_point(source_points, Point(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], r, c));
        }
    }
	/*
	make a new kd tree that maps intermediate centroids to what centroid they eventually converge to
	when iterating a point, keep track of the path of the centroid in a vector
	after it has converged to centroid `k`, add all these intermediate centroids to the kd tree and have them map to (void*)`k`
	When iterating future points, if an intermediate centroid is within a certain radius of a point already in the kd tree,
	it already maps to that centroid.
	*/
	struct kdtree *endpoints = kd_create(5);
	std::vector<Point> cluster_convergences;
	cluster_convergences.reserve(256);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			const unsigned char* const base_of_pixel = &image_data[(r * cols + c) * 3];
			Point centroid(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], r, c);
			std::vector<Point> this_trajectory;
			int cluster_id = -1;
			while (true) {
				struct kdres *traj_points = kd_nearestf(endpoints, &centroid.r);
				//traj_points can be null if endpoints is empty
				if (traj_points && kd_res_size(traj_points) != 0) {
					Point traj_point;
					int possible_cluster_id = (int)kd_res_itemf(traj_points, &traj_point.r);
					//Might be worth trying different thresholds here
					//If centroid was close enough to an already seen traj point, it will not be added to endpoints using this method.
					if (traj_point.distance_squared(&centroid) <= convergence_threshold * convergence_threshold) {
						centroid = cluster_convergences[possible_cluster_id];
						kd_res_free(traj_points);
						cluster_id = possible_cluster_id;
						break;
					}
				}
				
				Point new_centroid(0,0,0,0,0);
				struct kdres* near_points = neighbors(source_points, centroid, radius);
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
				this_trajectory.push_back(new_centroid);
				float delta_squared = new_centroid.distance_squared(&centroid);
				centroid = new_centroid;
				if (delta_squared <= convergence_threshold * convergence_threshold) {
					break;
				}
			}
			if (cluster_id == -1) {
				//did not join with existing trajectory
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
			}
			result[(r * cols + c) * 3] = cluster_id;
			
			for (int i = 0; i < this_trajectory.size(); i++) {
				assert( kd_insertf(endpoints, &this_trajectory[i].r, (void*)cluster_id) == 0 );
			}
		}
	}
	kd_free(endpoints);
	kd_free(source_points);
	
	if (do_color) color_result(image_data, result, cluster_convergences.size(), rows, cols);
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



void naive_kernel_internals(const unsigned char* image, int rows, int cols, int r, int c, float radius, Point* centroids, float* deltas, float convergence_threshold) {
	if (r >= rows || c >= cols) {
		return;
	}
	if (deltas[r * cols + c] <= convergence_threshold) {
		//No need to keep iterating this centroid
		return;
	}
	Point new_centroid(0, 0, 0, 0, 0);
	int num_neighbors = 0;
	Point* this_centroid = &centroids[r * cols + c];
	const float radius_squared = radius * radius;
	for (int d_r = (int)floorf(-radius) - 1; d_r <= (int)ceilf(radius) + 1; d_r++) {
		//float limit = sqrtf(radius * radius - d_r * d_r);
		//for (int d_c = floorf(-limit); d_c <= ceilf(limit); d_c++) {
		for (int d_c = (int)floorf(-radius) - 1; d_c <= (int)ceilf(radius) + 1; d_c++) {
			if (d_r * d_r + d_c * d_c > radius_squared) continue;
			int search_r = (int)floorf(this_centroid->row + d_r), search_c = (int)floorf(this_centroid->col + d_c);
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

__global__ void naive_kernel(const unsigned char *image, int rows, int cols, float radius, Point *centroids, float *deltas, float convergence_threshold) {
	int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
	if (r >= rows || c >= cols) {
		return;
	}

	/*
	* Each block is 32x32
	* My initial guess for SEARCH_RADIUS is 50
	* can fit a max of 16384 pixels in shmem
	* Doesn't make sense for shared memory to be more than [132][132]
	* Maximum shared memory on my device is [128][128]
	* More shared mem means less L1, try varying shared mem dimensions
	* Should be at least 32x32?
	*/
//number of pixels beyond the 32x32 that should be in shared memory
#define SH_PAD 48
//total dimension length of shared memory
#define SH_DIM (32 + 2*SH_PAD)
//Each thread is responsible for loading this many (squared) pixels
#define RESP_DIM ((SH_DIM + 31) / 32)
#if 0
	__shared__ unsigned char shared[SH_DIM][SH_DIM][3];
	for (int r_offset = 0; r_offset < RESP_DIM; r_offset++) {
		for (int c_offset = 0; c_offset < RESP_DIM; c_offset++) {
			const int image_r = 32 * blockIdx.y - SH_PAD + threadIdx.y * RESP_DIM,
				image_c = 32 * blockIdx.x - SH_PAD + threadIdx.x * RESP_DIM;
			if (image_r < 0 || image_r >= rows || image_c < 0 || image_c >= cols) {
				continue;
			}
			const unsigned char* const base_of_pixel = &image[(image_r * cols + image_c) * 3];
			int dest_r = RESP_DIM * threadIdx.y + r_offset,
				dest_c = RESP_DIM * threadIdx.x + c_offset;
			if (dest_r >= 0 && dest_r < SH_DIM && dest_c >= 0 && dest_c < SH_DIM) {
				shared[RESP_DIM * threadIdx.y + r_offset][RESP_DIM + threadIdx.x + c_offset][0] = base_of_pixel[0];
				shared[RESP_DIM * threadIdx.y + r_offset][RESP_DIM + threadIdx.x + c_offset][1] = base_of_pixel[1];
				shared[RESP_DIM * threadIdx.y + r_offset][RESP_DIM + threadIdx.x + c_offset][2] = base_of_pixel[2];
			}
		}
	}
	__syncthreads();
#endif
	if (deltas[r * cols + c] <= convergence_threshold) {
		return;
	}
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

unsigned char* sequential_gpu_version(const unsigned char* image_data, int rows, int cols, float radius, float convergence_threshold, bool do_color) {
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
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				naive_kernel_internals(image_data, rows, cols, r, c, radius, centroids, deltas, convergence_threshold);
			}
		}
		bool found_greater_than_thresh = false;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (deltas[r * cols + c] > convergence_threshold) {
					found_greater_than_thresh = true;
					break;
				}
			}
			if (found_greater_than_thresh) {
				break;
			}
		}
		if (!found_greater_than_thresh) {
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
				cluster_id = (int)cluster_convergences.size() - 1;
			}
			result[(r * cols + c) * 3] = cluster_id;
		}
	}
	if (do_color) color_result(image_data, result, cluster_convergences.size(), rows, cols);
	
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
			block_data[t_id] = max(block_data[t_id + stride], block_data[t_id]);
		}
	}

	__syncthreads();
	if (t_id == 0) {
		data[global_block_start] = block_data[0];
	}
}

unsigned char * naive_GPU_version(const unsigned char *image_data, int rows, int cols, float radius, float convergence_threshold, bool do_color) {
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
	for (int i = 0; i < rows * cols; i++) {
		host_deltas[i] = INFINITY;
	}
	float *dev_deltas = NULL;
	err_code = cudaMalloc((void**)&dev_deltas, rows * cols * sizeof(float));
	err_code = cudaMemcpy(dev_deltas, host_deltas, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
	
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
		//fprintf(stderr, "now starting iter %d\n", iters++);
		//My device (NVIDIA GeForce GTX 1660) has a max of 1024 threads per block
		dim3 block_dims(32, 32);
		dim3 grid_dims((int)ceil(cols / (float) 32), (int)ceil(rows / (float)32));
		naive_kernel<<<grid_dims, block_dims>>>(dev_image, rows, cols, radius, dev_centroids, dev_deltas, convergence_threshold);
		
		//err_code = cudaGetLastError(); //errors from launching the kernel
		//err_code = cudaDeviceSynchronize(); //errors that happened during the kernel launch
		
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

		//Faster code to check for convergence
		//change this to do diagnostics on whole-warp convergence
#if 1
		int blocks_necessary = (int)ceil((float)(rows * cols) / 2048.0f); //1024 is max threads per block on my device
		max_in_groups<<<blocks_necessary, 1024>>>(dev_deltas, rows * cols);
		err_code = cudaGetLastError();
		err_code = cudaDeviceSynchronize();
		bool all_below_threshold = true;
		//Maybe it would be faster to make one big array here and just do one memcpy into that
		for (int i = 0; i < rows * cols; i += 2048) {
			float temp;
			cudaMemcpy(&temp, &dev_deltas[i], sizeof(float), cudaMemcpyDeviceToHost);
			if (temp > convergence_threshold) {
				all_below_threshold = false;
				break;
			}
		}
		if (all_below_threshold) {
			break;
		}
		
#else
		float max_delta = 0.0f;
		int amount_above_thresh = 0;
		err_code = cudaMemcpy(host_deltas, dev_deltas, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (host_deltas[r * cols + c] > convergence_threshold) {
					amount_above_thresh++;
				}
				if (host_deltas[r * cols + c] > max_delta) {
					max_delta = host_deltas[r * cols + c];
				}
			}
		}
		/*
		* If I make the blocks 32x32, each warp handles a row of threads
		* This checks if every thread in a warp would have converged
		*/
		int warps_converged = 0;
		int total_warps = 0;
		for (int r = 0; r < rows; r++) {
			for (int warp = 0; warp < (cols + 31) / 32; warp++) {
				total_warps++;
				bool warp_entirely_converged = true;
				for (int c = warp * 32; c < warp * 32 + 32 && c < cols; c++) {
					if (host_deltas[r * cols + c] > convergence_threshold) {
						warp_entirely_converged = false;
						break;
					}
				}
				if (warp_entirely_converged) {
					warps_converged++;
				}
			}
		}
		fprintf(stderr, "Iteration %d has %d deltas > %f, %d / %d warps converged (%f%%)\n", iters++, amount_above_thresh, convergence_threshold,
			warps_converged, total_warps, (float)warps_converged / (float)total_warps);
		if (max_delta < convergence_threshold) {
			break;
		}
#endif
		iters++;
	}
	
	err_code = cudaMemcpy(host_centroids, dev_centroids, rows * cols * sizeof(Point), cudaMemcpyDeviceToHost);
	
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
	if (do_color) color_result(image_data, result, cluster_convergences.size(), rows, cols);
	
	free(host_centroids);
	cudaFree(dev_centroids);
	free(host_deltas);
	cudaFree(dev_deltas);
	cudaFree(dev_image);
	return result;
	
}
#define SEARCH_RADIUS 50
#define CONVERGENCE_THRESHOLD 10

void timings(const char* filename) {
	std::cout << "Now timing on " << filename << std::endl;
	int rows, cols, channels;
	unsigned char* image_data = (unsigned char*)stbi_load(filename, &cols, &rows, &channels, 3);
	if (!image_data) {
		fprintf(stderr, "Error reading image: %s\n", stbi_failure_reason());
		return;
	}

	/*auto start = std::chrono::high_resolution_clock::now();
	cpu_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD, false);
	auto end = std::chrono::high_resolution_clock::now();
	printf("naive CPU: %10lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());*/

	/*start = std::chrono::high_resolution_clock::now();
	cpu_version_with_trajectories(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD, false);
	end = std::chrono::high_resolution_clock::now();
	printf("CPU w/trj: %10lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());*/

	//start = std::chrono::high_resolution_clock::now();
	//sequential_gpu_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD, false);
	//end = std::chrono::high_resolution_clock::now();
	//printf("seq GPU  : %10lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	auto start = std::chrono::high_resolution_clock::now();
	naive_GPU_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD, false);
	auto end = std::chrono::high_resolution_clock::now();
	printf("naive GPU: %10lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}
int main()
{
	//timings("test_images/dapper_lad_smaller.jpg");
	timings("test_images/dapper_lad.jpg");
	return;

    int rows, cols, channels;
    unsigned char* image_data = (unsigned char*)stbi_load("test_images/dapper_lad_smaller.jpg", &cols, &rows, &channels, 3);
    if (!image_data) {
        fprintf(stderr, "Error reading image: %s\n", stbi_failure_reason());
        return -1;
    }
    /*unsigned char * cpu_result = cpu_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD, true);
    stbi_write_png("cpu_output.png", cols, rows, 3, cpu_result, 0);
    free(cpu_result);*/

	/*unsigned char* cpu_traj_result = cpu_version_with_trajectories(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD, true);
	stbi_write_png("cpu_output_traj.png", cols, rows, 3, cpu_traj_result, 0);
	free(cpu_traj_result);*/

	unsigned char* sequential_kernel_result = sequential_gpu_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD, true);
	stbi_write_png("sequential_output.png", cols, rows, 3, sequential_kernel_result, 0);
	free(sequential_kernel_result);

	unsigned char * gpu_result = naive_GPU_version(image_data, rows, cols, SEARCH_RADIUS, CONVERGENCE_THRESHOLD, true);
	stbi_write_png("gpu_output.png", cols, rows, 3, gpu_result, 0);
	free(gpu_result);
	
	stbi_image_free(image_data);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    return 0;
}