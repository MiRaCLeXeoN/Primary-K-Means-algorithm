#ifndef __KMEANS_HPP_
#define __KMEANS_HPP_

#include <vector>
#include <iostream>
#include <immintrin.h>
#define NUM_POINTS 1000000

using index_t = int;

struct Point {
	float x, y;

	Point() : x(0), y(0) {}
	Point(int _x, int _y) : x(_x), y(_y) {}
	Point(const Point& other) = default;
	~Point() = default;

	[[nodiscard]] float Distance(const Point& other) const noexcept;
};

class Kmeans {
public:
	Kmeans(const std::vector<Point>& points,
		const std::vector<Point>& init_centers);
	std::vector<index_t> Run(int max_iterations = 1000);
	~Kmeans(){
		free(p_cnt);
		free(assignment);
	}

private:
	//std::vector<Point> m_points;
	//std::vector<Point> m_centers;
	int m_numPoints;
	int m_numCenters;
	index_t* assignment;
	int* p_cnt;
	__attribute__((aligned(32))) float points_x[NUM_POINTS];
	__attribute__((aligned(32))) float points_y[NUM_POINTS];
	__attribute__((aligned(32))) float centers_x[15+1];
	__attribute__((aligned(32))) float centers_y[15+1];
	__attribute__((aligned(32))) __m256 __m256_centers[2][2];
};

std::istream& operator>>(std::istream& is, Point& pt);
std::ostream& operator<<(std::ostream& os, Point& pt);

#endif // __KMEANS_HPP_