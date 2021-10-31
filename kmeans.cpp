#include "kmeans.hpp"
#include <queue>
#include <cassert>
#include <limits>
#include <thread>
#include <omp.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define THREAD_COUNT 12
#define ALLIGNMENT 32

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
  m_points = points;
  m_centers = init_centers;
  m_numPoints = points.size();
  m_numCenters = init_centers.size();
}

std::vector<index_t> Kmeans::Run(int max_iterations) {
    std::vector<index_t> assignment(m_numPoints, 0); // the return vector
    std::vector<int> p_cnt(m_numCenters, 0);
	const float max_float = std::numeric_limits<float>::max();
	
	std::cout << "Running kmeans with num points = " << m_numPoints
		<< ", num centers = " << m_numCenters
		<< ", max iterations = " << max_iterations << "...\n";

	int curr_iteration = 0;

	while (max_iterations--) {
		++curr_iteration;
		bool flag = 0;

#		pragma omp parallel for num_threads(THREAD_COUNT)
		for (int i = 0; i < m_numPoints; ++i) {

			float min_dis = max_float;
			int temp = assignment[i];
			float dis;

			for (int k = 0; k < m_numCenters; ++k) {
				dis = m_points[i].Distance(m_centers[k]);
				if (dis < min_dis) {
					min_dis = dis;
					assignment[i] = k;
				}
			}
			if (assignment[i] != temp) flag = 1;
		}

		if (flag == 0) {
			goto converge;
		}

		memset(&m_centers[0], 0, m_numCenters * sizeof(Point));
		memset(&p_cnt[0], 0, m_numCenters * sizeof(index_t));

#		pragma omp parallel num_threads(THREAD_COUNT)
		{
			int* my_cnt = (int*)calloc(m_numCenters, sizeof(int));
			Point* my_centers = (Point*)calloc(m_numCenters, sizeof(Point));

			int my_rank = omp_get_thread_num();
			for (int i = m_numPoints * my_rank / THREAD_COUNT; i < m_numPoints * (my_rank+1) / THREAD_COUNT; ++i) {     //把多余的循环合并了
				index_t cluster_1 = assignment[i];
				my_centers[cluster_1].x += m_points[i].x;
				my_centers[cluster_1].y += m_points[i].y;
				my_cnt[cluster_1]++;
			}
#		pragma omp critical(add)
			{
				for (int i = 0; i < m_numCenters; i++) {
					m_centers[i].x += my_centers[i].x;
					m_centers[i].y += my_centers[i].y;
					p_cnt[i] += my_cnt[i];
				}
			}
			free(my_cnt);
			free(my_centers);
		}
		
#		pragma omp parallel for num_threads(THREAD_COUNT)
		for (int j = 0; j < m_numCenters; ++j) {
			m_centers[j].x /= p_cnt[j];
			m_centers[j].y /= p_cnt[j];
		}
	}

converge:
	std::cout << "Finished in " << curr_iteration << " iterations." << std::endl;
	return assignment;
}
