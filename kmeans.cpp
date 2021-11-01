#include "kmeans.hpp"
#include <queue>
#include <cassert>
#include <limits>
#include <thread>
#include <omp.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define THREAD_COUNT 8
#define ALLIGNMENT 32
#define NUM_POINTS 1000000

inline float Point::Distance(const Point &other) const noexcept {
  float a = x - other.x;
  float b = y - other.y;
  return a * a + b * b;
}

std::istream &operator>>(std::istream &is, Point &pt) {
  return is >> pt.x >> pt.y;
}

std::ostream &operator<<(std::ostream &os, Point &pt) {
  return os << pt.x << " " << pt.y;
}

Kmeans::Kmeans(const std::vector<Point> &points,
               const std::vector<Point> &init_centers) {
  //m_points = points;
  //m_centers = init_centers;
  m_numPoints = points.size();
  m_numCenters = init_centers.size();

  for (int i = 0; i < NUM_POINTS; i++) {
	  points_x[i] = points[i].x;
	  points_y[i] = points[i].y;
  }
  for (int i = 0; i < 15+1; i++) {
	  centers_x[i] = init_centers[i].x;
	  centers_y[i] = init_centers[i].y;
  }
  centers_x[15] = 4096.0f;	//maximize
  centers_y[15] = 4096.0f;

  assignment = (index_t*)calloc(m_numPoints, sizeof(index_t));
  p_cnt = (int*)calloc(m_numPoints, sizeof(int));
}

std::vector<index_t> Kmeans::Run(int max_iterations) {
	
	const float max_float = std::numeric_limits<float>::max();
	
	std::cout << "Running kmeans with num points = " << m_numPoints
		<< ", num centers = " << m_numCenters
		<< ", max iterations = " << max_iterations << "...\n";

	int curr_iteration = 0;

	while (max_iterations--) {
		++curr_iteration;
		bool flag = 0;
		//std::cout << curr_iteration << std::endl;
		__m256_centers[0][0] = _mm256_load_ps(centers_x);
		__m256_centers[0][1] = _mm256_load_ps(centers_x+8);
		__m256_centers[1][0] = _mm256_load_ps(centers_y);
		__m256_centers[1][1] = _mm256_load_ps(centers_y+8);

#		pragma omp parallel for num_threads(THREAD_COUNT)
		for (int i = 0; i < m_numPoints; i++) {
			
			//int temp = assignment[i];
			//float dis;
			//std::cout << "i=" << i << ' ' << std::endl;

			__attribute__((aligned(32))) __m256 __m256_x = _mm256_set1_ps(points_x[i]);
			__attribute__((aligned(32))) __m256 __m256_y = _mm256_set1_ps(points_y[i]);
			__attribute__((aligned(32))) __m256 __m256_tmp[2][2];
			__m256_tmp[0][0] = __m256_centers[0][0];
			__m256_tmp[0][1] = __m256_centers[0][1];
			__m256_tmp[1][0] = __m256_centers[1][0];
			__m256_tmp[1][1] = __m256_centers[1][1];

			__m256_tmp[0][0] = _mm256_sub_ps(__m256_tmp[0][0], __m256_x);
			__m256_tmp[0][0] = _mm256_mul_ps(__m256_tmp[0][0], __m256_tmp[0][0]);

			__m256_tmp[0][1] = _mm256_sub_ps(__m256_tmp[0][1], __m256_x);
			__m256_tmp[0][1] = _mm256_mul_ps(__m256_tmp[0][1], __m256_tmp[0][1]);

			__m256_tmp[1][0] = _mm256_sub_ps(__m256_tmp[1][0], __m256_y);
			__m256_tmp[1][0] = _mm256_mul_ps(__m256_tmp[1][0], __m256_tmp[1][0]);

			__m256_tmp[1][1] = _mm256_sub_ps(__m256_tmp[1][1], __m256_y);
			__m256_tmp[1][1] = _mm256_mul_ps(__m256_tmp[1][1], __m256_tmp[1][1]);
			
			__m256_tmp[1][0] = _mm256_add_ps(__m256_tmp[1][0], __m256_tmp[0][0]);
			__m256_tmp[1][1] = _mm256_add_ps(__m256_tmp[1][1], __m256_tmp[0][1]);
			__attribute__((aligned(32))) float tmp[16];
			_mm256_store_ps(tmp, __m256_tmp[1][0]);
			_mm256_store_ps(tmp + 8, __m256_tmp[1][1]);

			int min = 15;
			for (int k = 0; k < m_numCenters; ++k) {
				if (tmp[k] < tmp[min]) {
					min = k;
				}
			}
			if (!flag && assignment[i] != min) flag = 1;
			assignment[i] = min;
		}

		if (flag == 0) {
			goto converge;
		}

		memset(centers_x, 0, m_numCenters * sizeof(float));
		memset(centers_y, 0, m_numCenters * sizeof(float));
		memset(p_cnt, 0, m_numCenters * sizeof(index_t));

#		pragma omp parallel num_threads(THREAD_COUNT)
		{
			int* my_cnt = (int*)calloc(m_numCenters, sizeof(int));
			Point* my_centers = (Point*)calloc(m_numCenters, sizeof(Point));

			int my_rank = omp_get_thread_num();
			for (int i = m_numPoints * my_rank / THREAD_COUNT; i < m_numPoints * (my_rank+1) / THREAD_COUNT; ++i) {     //把多余的循环合并了
				index_t cluster_1 = assignment[i];
				my_centers[cluster_1].x += points_x[i];
				my_centers[cluster_1].y += points_y[i];
				my_cnt[cluster_1]++;
			}
#		pragma omp critical(add)
			{
				for (int i = 0; i < m_numCenters; i++) {
					centers_x[i] += my_centers[i].x;
					centers_y[i] += my_centers[i].y;
					p_cnt[i] += my_cnt[i];
				}
			}
			free(my_cnt);
			free(my_centers);
		}
		
#		pragma omp parallel for num_threads(THREAD_COUNT)
		for (int j = 0; j < m_numCenters; ++j) {
			centers_x[j] /= p_cnt[j];
			centers_y[j] /= p_cnt[j];
		}
	}
	
converge:
	std::cout << "Finished in " << curr_iteration << " iterations." << std::endl;
	std::vector<index_t> result(m_numPoints);
	for (int i = 0; i < m_numPoints; i++) result[i] = assignment[i];
	return result;
}
