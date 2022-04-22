﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
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

//CONFIG

//#define TIME_ITERS
#define RESTRICT __restrict
//#define RESTRICT

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

void color_result(const unsigned char *, unsigned char *, int *, int, int, int);
unsigned char * cpu_version(const unsigned char* image_data, int rows, int cols, float radius, float convergence_threshold, bool do_color) {
    unsigned char* result = (unsigned char*)malloc(rows * cols * 3);
	int* cluster_ids = (int*)malloc(rows * cols * sizeof(int));
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
				cluster_convergences.push_back(centroid);
				cluster_id = cluster_convergences.size() - 1;
			}
			cluster_ids[r * cols + c] = cluster_id;
		}
	}
	kd_free(kd);
	//fprintf(stderr, "Max iters in CPU version was %d\n", max_iters);
	
	//now compute average rgb for each cluster
	if (do_color) {
		color_result(image_data, result, cluster_ids, cluster_convergences.size(), rows, cols);
	}
	free(cluster_ids);
	return result;
}

unsigned char *cpu_version_with_trajectories(const unsigned char *image_data, int rows, int cols, float radius, float convergence_threshold, bool do_color) {
	unsigned char* result = (unsigned char*)malloc(rows * cols * 3);
	int* cluster_ids = (int*)malloc(rows * cols * sizeof(int));
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
					cluster_convergences.push_back(centroid);
					cluster_id = cluster_convergences.size() - 1;
				}
			}
			cluster_ids[r * cols + c] = cluster_id;
			
			for (int i = 0; i < this_trajectory.size(); i++) {
				assert( kd_insertf(endpoints, &this_trajectory[i].r, (void*)cluster_id) == 0 );
			}
		}
	}
	kd_free(endpoints);
	kd_free(source_points);
	
	if (do_color) color_result(image_data, result, cluster_ids, cluster_convergences.size(), rows, cols);
	free(cluster_ids);
	return result;
}

void color_result(const unsigned char *image_data, unsigned char *result, int *cluster_ids, int num_clusters, int rows, int cols) {
	std::vector<long long> average_rs(num_clusters),
							average_gs(num_clusters),
							average_bs(num_clusters);
	std::vector<int> cluster_sizes(num_clusters);
	//fprintf(stderr, "%d clusters\n", num_clusters);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int cluster_num = cluster_ids[r * cols + c];
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
			int cluster_num = cluster_ids[r * cols + c];
			base_of_result_pixel[0] = average_rs[cluster_num];
			base_of_result_pixel[1] = average_gs[cluster_num];
			base_of_result_pixel[2] = average_bs[cluster_num];
		}
	}
}

void naive_kernel_internals(const unsigned char* RESTRICT image, int rows, int cols, int r, int c, float radius, Point* centroids, float* deltas, float convergence_threshold) {
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

template <bool EarlyStop>
__global__ void first_kernel(const unsigned char * const RESTRICT image, const int rows, const int cols, const float radius, Point * const centroids, float * const deltas, const float convergence_threshold) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
	if (r >= rows || c >= cols) {
		return;
	}
	if constexpr (EarlyStop) {
		if (deltas[r * cols + c] <= convergence_threshold) {
			return;
		}
	}

	int num_neighbors = 0;
	Point *this_centroid = &centroids[r * cols + c];
	const float radius_squared = radius * radius;
	Point new_centroid(0, 0, 0, 0, 0);
	for (int d_r = (int)floorf(-radius) - 1; d_r <= (int)ceilf(radius) + 1; d_r++) {
		float limit = sqrtf(radius * radius - d_r * d_r);
		for (int d_c = floorf(-limit); d_c <= ceilf(limit); d_c++) {
		//for (int d_c = (int)floorf(-radius) - 1; d_c <= (int)ceilf(radius) + 1; d_c++) {
			if (d_r * d_r + d_c * d_c > radius_squared) continue;
			const int search_r = (int)floorf(this_centroid->row + d_r), search_c = (int)floorf(this_centroid->col + d_c);
			if (search_r < 0 || search_r >= rows || search_c < 0 || search_c >= cols) {
				continue;
			}
			const unsigned char* const base_of_pixel = &image[(search_r * cols + search_c) * 3];
			Point potential_neighbor(base_of_pixel[0], base_of_pixel[1], base_of_pixel[2], search_r, search_c);
			if (potential_neighbor.distance_squared(this_centroid) <= radius_squared) {
				num_neighbors++;
				for (int i = 0; i < 5; i++) {
					*new_centroid[i] += *potential_neighbor[i];
				}
			}
		}
	}
	for (int i = 0; i < 5; i++) {
		*new_centroid[i] /= num_neighbors;
	}
	const float distance_squared = new_centroid.distance_squared(this_centroid);
	*this_centroid = new_centroid;

	deltas[r * cols + c] = distance_squared;
}

__global__ void reg_points_kernel(const unsigned char * const RESTRICT image, const int rows, const int cols, const float radius, Point * const centroids, float * const deltas, const float convergence_threshold) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
	if (r >= rows || c >= cols) {
		return;
	}
	if (deltas[r * cols + c] <= convergence_threshold) {
		return;
	}

	int num_neighbors = 0;
	Point *this_centroid = &centroids[r * cols + c];
	const float radius_squared = radius * radius;
	float new_r = 0.0f, new_g = 0.0f, new_b = 0.0f, new_row = 0.0f, new_col = 0.0f;
	const float this_centroid_row = this_centroid->row,
				this_centroid_col = this_centroid->col,
				this_centroid_r = this_centroid->r,
				this_centroid_g = this_centroid->g,
				this_centroid_b = this_centroid->b;
	for (int d_r = (int)floorf(-radius) - 1; d_r <= (int)ceilf(radius) + 1; d_r++) {
		float limit = sqrtf(radius * radius - d_r * d_r);
		for (int d_c = floorf(-limit); d_c <= ceilf(limit); d_c++) {
			if (d_r * d_r + d_c * d_c > radius_squared) continue;
			const int search_r = (int)floorf(this_centroid_row + d_r), search_c = (int)floorf(this_centroid_col + d_c);
			if (search_r < 0 || search_r >= rows || search_c < 0 || search_c >= cols) {
				continue;
			}
			const unsigned char* const base_of_pixel = &image[(search_r * cols + search_c) * 3];
			float potential_r = base_of_pixel[0],
				potential_g = base_of_pixel[1],
				potential_b = base_of_pixel[2];
			const float delta_r = potential_r - this_centroid_r;
			const float delta_g = potential_g - this_centroid_g;
			const float delta_b = potential_b - this_centroid_b;
			const float delta_row = search_r - this_centroid_row;
			const float delta_col = search_c - this_centroid_col;
			if (delta_r * delta_r + delta_g * delta_g + delta_b * delta_b + delta_row * delta_row + delta_col * delta_col <= radius_squared) {
				num_neighbors++;
				//fprintf(stderr, "pixel at r=%d, c=%d is a neighbor of centroid r=%f, c=%f\n", search_r, search_c, this_centroid->row, this_centroid->col);
				new_r += potential_r;
				new_g += potential_g;
				new_b += potential_b;
				new_row += search_r;
				new_col += search_c;
			}
		}
	}
	new_r /= num_neighbors;
	new_g /= num_neighbors;
	new_b /= num_neighbors;
	new_row /= num_neighbors;
	new_col /= num_neighbors;
	const float delta_r = new_r - this_centroid_r;
	const float delta_g = new_g - this_centroid_g;
	const float delta_b = new_b - this_centroid_b;
	const float delta_row = new_row - this_centroid_row;
	const float delta_col = new_col - this_centroid_col;
	const float distance_squared = delta_r * delta_r + delta_g * delta_g + delta_b * delta_b + delta_row * delta_row + delta_col * delta_col;
	this_centroid->r = new_r;
	this_centroid->g = new_g;
	this_centroid->b = new_b;
	this_centroid->row = new_row;
	this_centroid->col = new_col;

	deltas[r * cols + c] = distance_squared;
}

template <size_t SH_PAD>
__global__ void shmem_kernel(const unsigned char * const RESTRICT image, const int rows, const int cols, const float radius, Point * const centroids, float * const deltas, const float convergence_threshold) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;

	/*
	* Each block is 32x32
	* My initial guess for SEARCH_RADIUS is 50
	* can fit a max of 16384 pixels in shmem
	* Doesn't make sense for shared memory to be more than [132][132]
	* Maximum shared memory on my device is [128][128]
	* More shared mem means less L1, try varying shared mem dimensions
	* Should be at least 32x32?
	*/
	//SH_PAD is number of pixels beyond the 32x32 that should be in shared memory
	//total dimension length of shared memory
	constexpr size_t SH_DIM = 32 + 2 * SH_PAD;
	//Each thread is responsible for loading this many (squared) pixels
	constexpr size_t RESP_DIM = (SH_DIM + 31) / 32;
	__shared__ unsigned char shared[SH_DIM][SH_DIM][3];
	for (int r_offset = 0; r_offset < RESP_DIM; r_offset++) {
		for (int c_offset = 0; c_offset < RESP_DIM; c_offset++) {
			const int image_r = 32 * blockIdx.y - SH_PAD + threadIdx.y * RESP_DIM + r_offset,
				image_c = 32 * blockIdx.x - SH_PAD + threadIdx.x * RESP_DIM + c_offset;
			if (image_r < 0 || image_r >= rows || image_c < 0 || image_c >= cols) {
				continue;
			}
			const unsigned char* const base_of_pixel = &image[(image_r * cols + image_c) * 3];
			int dest_r = RESP_DIM * threadIdx.y + r_offset,
				dest_c = RESP_DIM * threadIdx.x + c_offset;
			if (dest_r >= 0 && dest_r < SH_DIM && dest_c >= 0 && dest_c < SH_DIM) {
				shared[dest_r][dest_c][0] = base_of_pixel[0];
				shared[dest_r][dest_c][1] = base_of_pixel[1];
				shared[dest_r][dest_c][2] = base_of_pixel[2];
			}
		}
	}
	__syncthreads();
	//This needs to happen after shmem is populated because out of bounds threads
	//could still responsible for some in-bounds shmem items
	if (r >= rows || c >= cols) {
		return;
	}
	if (deltas[r * cols + c] <= convergence_threshold) {
		return;
	}

	int num_neighbors = 0;
	Point *this_centroid = &centroids[r * cols + c];
	const float radius_squared = radius * radius;
	float new_r = 0.0f, new_g = 0.0f, new_b = 0.0f, new_row = 0.0f, new_col = 0.0f;
	const float this_centroid_row = this_centroid->row,
				this_centroid_col = this_centroid->col,
				this_centroid_r = this_centroid->r,
				this_centroid_g = this_centroid->g,
				this_centroid_b = this_centroid->b;
	for (int d_r = (int)floorf(-radius) - 1; d_r <= (int)ceilf(radius) + 1; d_r++) {
		//float limit = sqrtf(radius * radius - d_r * d_r);
		//for (int d_c = floorf(-limit); d_c <= ceilf(limit); d_c++) {
		for (int d_c = (int)floorf(-radius) - 1; d_c <= (int)ceilf(radius) + 1; d_c++) {
			if (d_r * d_r + d_c * d_c > radius_squared) continue;
			const int search_r = (int)floorf(this_centroid_row + d_r), search_c = (int)floorf(this_centroid_col + d_c);
			if (search_r < 0 || search_r >= rows || search_c < 0 || search_c >= cols) {
				continue;
			}
			float potential_r, potential_g, potential_b;
			/*
			if blockidx = {2,2}, shared[0][0] is image[64-SH_PAD][64-SH_PAD]
			shared[SH_PAD][SH_PAD] is image[64][64]
			shared[SH_DIM-1][SH_DIM-1] is image[96+SH_PAD-1][96+SH_PAD-1]
			sharedIdx = imgIdx - 32*blkIdx + SH_PAD
			*/
			const int shared_r = search_r - 32 * blockIdx.y + SH_PAD,
				shared_c = search_c - 32 * blockIdx.x + SH_PAD;
			if (shared_r >= 0 && shared_r < SH_DIM && shared_c >= 0 && shared_c < SH_DIM) {
				potential_r = shared[shared_r][shared_c][0];
				potential_g = shared[shared_r][shared_c][1];
				potential_b = shared[shared_r][shared_c][2];
			} else {
				const unsigned char* const base_of_pixel = &image[(search_r * cols + search_c) * 3];
				potential_r = base_of_pixel[0];
				potential_g = base_of_pixel[1];
				potential_b = base_of_pixel[2];
			}
			const float delta_r = potential_r - this_centroid_r;
			const float delta_g = potential_g - this_centroid_g;
			const float delta_b = potential_b - this_centroid_b;
			const float delta_row = search_r - this_centroid_row;
			const float delta_col = search_c - this_centroid_col;
			if (delta_r * delta_r + delta_g * delta_g + delta_b * delta_b + delta_row * delta_row + delta_col * delta_col <= radius_squared) {
				num_neighbors++;
				new_r += potential_r;
				new_g += potential_g;
				new_b += potential_b;
				new_row += search_r;
				new_col += search_c;
			}
		}
	}
	new_r /= num_neighbors;
	new_g /= num_neighbors;
	new_b /= num_neighbors;
	new_row /= num_neighbors;
	new_col /= num_neighbors;
	const float delta_r = new_r - this_centroid_r;
	const float delta_g = new_g - this_centroid_g;
	const float delta_b = new_b - this_centroid_b;
	const float delta_row = new_row - this_centroid_row;
	const float delta_col = new_col - this_centroid_col;
	float distance_squared = delta_r * delta_r + delta_g * delta_g + delta_b * delta_b + delta_row * delta_row + delta_col * delta_col;
	this_centroid->r = new_r;
	this_centroid->g = new_g;
	this_centroid->b = new_b;
	this_centroid->row = new_row;
	this_centroid->col = new_col;

	deltas[r * cols + c] = distance_squared;
}

unsigned char* sequential_gpu_version(const unsigned char* image_data, int rows, int cols, float radius, float convergence_threshold, bool do_color) {
	unsigned char *result = (unsigned char*)malloc(rows * cols * 3);
	int* cluster_ids = (int*)malloc(rows * cols * sizeof(int));
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
	
	//int iters = 0;
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
				cluster_convergences.push_back(centroids[r * cols + c]);
				cluster_id = (int)cluster_convergences.size() - 1;
			}
			cluster_ids[r * cols + c] = cluster_id;
		}
	}
	if (do_color) color_result(image_data, result, cluster_ids, cluster_convergences.size(), rows, cols);
	
	free(cluster_ids);
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

__device__ int need_more_iter;

template <bool UseAtomics>
__global__ void kernel_without_deltas(const unsigned char* const RESTRICT image, const int rows, const int cols, const float radius, Point* const centroids, const float convergence_threshold) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
	/*
	If a centroid has a negative .r value, return
	Compute the delta for each centroid, as before
	if it is less than the threshold, .r -= 256
	otherwise, shared_flag = true;

	__syncthreads();
	thread 0 in each block checks shared_flag
	if it is true, atomicOr(should_continue, true)
	*/
	__shared__ int need_more_shared;
	if constexpr (UseAtomics) {
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			need_more_shared = 0;
		}
	}

	if (r >= rows || c >= cols) {
		return;
	}
	Point* const this_centroid = &centroids[r * cols + c];
	if (this_centroid->r < 0) {
		return;
	}
	if constexpr (UseAtomics) {
		__syncthreads();
	} else {
		need_more_shared = 0;
	}
	
	int num_neighbors = 0;
	const float radius_squared = radius * radius;
	float new_r = 0.0f, new_g = 0.0f, new_b = 0.0f, new_row = 0.0f, new_col = 0.0f;
	const float this_centroid_row = this_centroid->row,
		this_centroid_col = this_centroid->col,
		this_centroid_r = this_centroid->r,
		this_centroid_g = this_centroid->g,
		this_centroid_b = this_centroid->b;
	for (int d_r = (int)floorf(-radius) - 1; d_r <= (int)ceilf(radius) + 1; d_r++) {
		for (int d_c = (int)floorf(-radius) - 1; d_c <= (int)ceilf(radius) + 1; d_c++) {
			if (d_r * d_r + d_c * d_c > radius_squared) continue;
			const int search_r = (int)floorf(this_centroid_row + d_r), search_c = (int)floorf(this_centroid_col + d_c);
			if (search_r < 0 || search_r >= rows || search_c < 0 || search_c >= cols) {
				continue;
			}
			const unsigned char* const base_of_pixel = &image[(search_r * cols + search_c) * 3];
			const float potential_r = base_of_pixel[0],
				potential_g = base_of_pixel[1],
				potential_b = base_of_pixel[2];
			
			const float delta_r = potential_r - this_centroid_r;
			const float delta_g = potential_g - this_centroid_g;
			const float delta_b = potential_b - this_centroid_b;
			const float delta_row = search_r - this_centroid_row;
			const float delta_col = search_c - this_centroid_col;
			if (delta_r * delta_r + delta_g * delta_g + delta_b * delta_b + delta_row * delta_row + delta_col * delta_col <= radius_squared) {
				num_neighbors++;
				//fprintf(stderr, "pixel at r=%d, c=%d is a neighbor of centroid r=%f, c=%f\n", search_r, search_c, this_centroid->row, this_centroid->col);
				new_r += potential_r;
				new_g += potential_g;
				new_b += potential_b;
				new_row += search_r;
				new_col += search_c;
			}
		}
	}
	new_r /= num_neighbors;
	new_g /= num_neighbors;
	new_b /= num_neighbors;
	new_row /= num_neighbors;
	new_col /= num_neighbors;
	const float delta_r = new_r - this_centroid_r;
	const float delta_g = new_g - this_centroid_g;
	const float delta_b = new_b - this_centroid_b;
	const float delta_row = new_row - this_centroid_row;
	const float delta_col = new_col - this_centroid_col;
	float distance_squared = delta_r * delta_r + delta_g * delta_g + delta_b * delta_b + delta_row * delta_row + delta_col * delta_col;

	if (distance_squared <= convergence_threshold) {
		//this centroid has converged, do no further processing on it
		new_r -= 256;
	} else if (!need_more_shared) {
		if constexpr (UseAtomics) {
			atomicOr(&need_more_shared, 1);
		} else {
			need_more_shared = 1;
		}
	}
	this_centroid->r = new_r;
	this_centroid->g = new_g;
	this_centroid->b = new_b;
	this_centroid->row = new_row;
	this_centroid->col = new_col;

	/*
	Only one thread in each block needs to set the global flag
	Can't always choose thread (0,0) in the block, this may have already returned
	*/
	if (need_more_shared) {
		if constexpr (UseAtomics) {
			atomicOr(&need_more_iter, 1);
		} else {
			need_more_iter = 1;
		}
	}
}

enum KernelType {
	First,
	RegPoints,
	Shmem,
	NoDeltas
};

template <KernelType WhichKernel, bool EarlyStop = true, size_t SH_PAD = 48, bool UseAtomics = true>
unsigned char * GPU_driver(const unsigned char *image_data, int rows, int cols, float radius, float convergence_threshold, bool do_color) {
	unsigned char *result = (unsigned char*)malloc(rows * cols * 3);
	int* cluster_ids = (int*)malloc(rows * cols * sizeof(int));
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
	
	float *host_deltas = NULL, *dev_deltas = NULL;
	if constexpr (WhichKernel != NoDeltas) {
		host_deltas = (float*)malloc(rows * cols * sizeof(float));
		for (int i = 0; i < rows * cols; i++) {
			host_deltas[i] = INFINITY;
		}
		dev_deltas = NULL;
		err_code = cudaMalloc((void**)&dev_deltas, rows * cols * sizeof(float));
		err_code = cudaMemcpy(dev_deltas, host_deltas, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
	}
	
	
	unsigned char *dev_image = NULL;
	err_code = cudaMalloc((void**)&dev_image, rows * cols * 3);
	err_code = cudaMemcpy(dev_image, image_data, rows * cols * 3, cudaMemcpyHostToDevice);
	
#ifdef TIME_ITERS
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
#endif
	for (int iters = 0; ; iters++) {
#ifdef TIME_ITERS
		start = std::chrono::high_resolution_clock::now();
#endif
		//fprintf(stderr, "now starting iter %d\n", iters++);
		//My device (NVIDIA GeForce GTX 1660) has a max of 1024 threads per block
		dim3 block_dims(32, 32);
		dim3 grid_dims((cols + 31)/32, (rows + 31)/32);
		if constexpr (WhichKernel == First) {
			first_kernel<EarlyStop><<<grid_dims, block_dims>>>(dev_image, rows, cols, radius, dev_centroids, dev_deltas, convergence_threshold);
		} else if constexpr (WhichKernel == RegPoints) {
			reg_points_kernel<<<grid_dims, block_dims>>>(dev_image, rows, cols, radius, dev_centroids, dev_deltas, convergence_threshold);
		} else if constexpr (WhichKernel == Shmem) {
			shmem_kernel<SH_PAD><<<grid_dims, block_dims>>>(dev_image, rows, cols, radius, dev_centroids, dev_deltas, convergence_threshold);
		} else if constexpr (WhichKernel == NoDeltas) {
			int zero = 0;
			err_code = cudaMemcpyToSymbol(need_more_iter, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
			kernel_without_deltas<UseAtomics><<<grid_dims, block_dims>>>(dev_image, rows, cols, radius, dev_centroids, convergence_threshold);
		} else {
			fprintf(stderr, "Invalid Kernel specified\n");
			abort();
		}
		
#ifdef _DEBUG
		err_code = cudaGetLastError(); //errors from launching the kernel
		err_code = cudaDeviceSynchronize(); //errors that happened during the kernel launch
#endif
		if constexpr (WhichKernel != NoDeltas) {
			int blocks_necessary = (int)ceil((float)(rows * cols) / 2048.0f); //1024 is max threads per block on my device
			max_in_groups<<<blocks_necessary, 1024>>>(dev_deltas, rows * cols);
#ifdef _DEBUG
			err_code = cudaGetLastError();
			err_code = cudaDeviceSynchronize();
#endif
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
#ifdef TIME_ITERS
			end = std::chrono::high_resolution_clock::now();
			printf("iter %d took %5ld ms\n", iters, std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
#endif
			if (all_below_threshold) {
				break;
			}
		} else { //using a NoDeltas kernel, stopping condition is different
			int should_continue_host = 1;
			err_code = cudaMemcpyFromSymbol(&should_continue_host, need_more_iter, sizeof(int), 0, cudaMemcpyDeviceToHost);
			if (!should_continue_host) {
				break;
			}
		}
		
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
				cluster_convergences.push_back(host_centroids[r * cols + c]);
				cluster_id = cluster_convergences.size() - 1;
			}
			cluster_ids[r * cols + c] = cluster_id;
		}
	}
	if (do_color) color_result(image_data, result, cluster_ids, cluster_convergences.size(), rows, cols);
	
	free(host_centroids);
	cudaFree(dev_centroids);
	free(host_deltas);
	cudaFree(dev_deltas);
	cudaFree(dev_image);
	(void)err_code;
	return result;
	
}

void timings(const char* filename, float radius, float convergence_threshold) {
	cudaFree(0); //Force init CUDA runtime
	printf("Now timing on %s with radius %f and threshold %f\n", filename, radius, convergence_threshold);
	int rows, cols, channels;
	unsigned char* image_data = (unsigned char*)stbi_load(filename, &cols, &rows, &channels, 3);
	if (!image_data) {
		fprintf(stderr, "Error reading image: %s\n", stbi_failure_reason());
		return;
	}

	std::chrono::time_point < std::chrono::high_resolution_clock > start, end;
#define TIME(name, stmt) do { \
	start = std::chrono::high_resolution_clock::now(); \
	stmt; \
	end = std::chrono::high_resolution_clock::now(); \
	printf(name ": %10lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()); \
} while (0)
	
	//TIME("naive CPU            ", cpu_version(image_data, rows, cols, radius, convergence_threshold, false));
	//TIME("CPU with trajectories", cpu_version_with_trajectories(image_data, rows, cols, radius, convergence_threshold, false));
	//TIME("first kernel         ", (GPU_driver<First, false, 0, false>(image_data, rows, cols, radius, convergence_threshold, false)));
	//TIME("with early stop      ", (GPU_driver<First, true, 0, false>(image_data, rows, cols, radius, convergence_threshold, false)));
	TIME("points in regs       ", GPU_driver<RegPoints>(image_data, rows, cols, radius, convergence_threshold, false));
	TIME("Shmem, pad=48        ", (GPU_driver<Shmem, false, 48, false>(image_data, rows, cols, radius, convergence_threshold, false)));
	//TIME("Shmem, pad=32        ", (GPU_driver<Shmem, false, 32, false>(image_data, rows, cols, radius, convergence_threshold, false)));
	//TIME("Shmem, pad=16        ", (GPU_driver<Shmem, false, 16, false>(image_data, rows, cols, radius, convergence_threshold, false)));
	//TIME("Shmem, pad=8         ", (GPU_driver<Shmem, false, 8, false>(image_data, rows, cols, radius, convergence_threshold, false)));
	TIME("no deltas            ", (GPU_driver<NoDeltas, false, 0, false>(image_data, rows, cols, radius, convergence_threshold, false)));
	TIME("no deltas, no atomics", GPU_driver<NoDeltas>(image_data, rows, cols, radius, convergence_threshold, false));
}
int main()
{
	//timings("test_images/dapper_lad_smaller.jpg", 50, 10);
	//timings("test_images/dapper_lad.jpg", 50, 10);
	timings("test_images/campus.jpg", 10, 10);
	timings("test_images/campus.jpg", 50, 10);
	timings("test_images/campus.jpg", 100, 10);
	
	//timings("test_images/eas_1500x100.jpg", 50, 50);
	return;

    int rows, cols, channels;
    unsigned char* image_data = (unsigned char*)stbi_load("test_images/dapper_lad_smaller.jpg", &cols, &rows, &channels, 3);
    if (!image_data) {
        fprintf(stderr, "Error reading image: %s\n", stbi_failure_reason());
        return -1;
    }
#define WRITE_IMG(filename, stmt) do { \
	unsigned char * result = stmt; \
	stbi_write_png(filename, cols, rows, 3, result, 0); \
	free(result); \
} while (0)
	const float radius = 50, convergence_threshold = 10;
	WRITE_IMG("cpu_output.png", cpu_version(image_data, rows, cols, radius, convergence_threshold, true));
	WRITE_IMG("cpu_traj_output.png", cpu_version_with_trajectories(image_data, rows, cols, radius, convergence_threshold, true));

	WRITE_IMG("first_kernel_output.png", (GPU_driver<First, false, 0, false>(image_data, rows, cols, radius, convergence_threshold, true)));
	WRITE_IMG("early_stop_output.png", (GPU_driver<First, true, 0, false>(image_data, rows, cols, radius, convergence_threshold, true)));
	WRITE_IMG("reg_points_output.png", GPU_driver<RegPoints>(image_data, rows, cols, radius, convergence_threshold, true));
	WRITE_IMG("shmem_pad_48_output.png", (GPU_driver<Shmem, false, 48, false>(image_data, rows, cols, radius, convergence_threshold, true)));
	WRITE_IMG("Shmem_pad_32_output.png", (GPU_driver<Shmem, false, 32, false>(image_data, rows, cols, radius, convergence_threshold, true)));
	WRITE_IMG("Shmem_pad_16_output.png", (GPU_driver<Shmem, false, 16, false>(image_data, rows, cols, radius, convergence_threshold, true)));
	WRITE_IMG("Shmem_pad_8_output.png", (GPU_driver<Shmem, false, 8, false>(image_data, rows, cols, radius, convergence_threshold, true)));
	WRITE_IMG("no_deltas_output.png", (GPU_driver<NoDeltas, false, 0, false>(image_data, rows, cols, radius, convergence_threshold, true)));
	WRITE_IMG("no_atomics_output.png", GPU_driver<NoDeltas>(image_data, rows, cols, radius, convergence_threshold, true));
	
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