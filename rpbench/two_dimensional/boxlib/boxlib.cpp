#include<cmath>
#include<algorithm>
#include<limits>
#include<vector>
#include<iostream>

extern "C" {
    void* create_parametric_maze_boxes(double* param, int n_dof, double wall_thickness, double hole_size, double y_length);
    void* make_boxes(double* xmin, double* xmax, double* ymin, double* ymax, int n);
    void delete_boxes(void* boxes);
    double signed_distance(double x, double y, void* boxes);
    void signed_distance_batch(double* x, double* y, double* dist, int n, void* boxes);
}

struct Box {
    double xmin, xmax, ymin, ymax;

    double signed_distance(double x, double y) {
        double dx = std::max({0.0, xmin - x, x - xmax});
        double dy = std::max({0.0, ymin - y, y - ymax});
        if (x >= xmin && x <= xmax && y >= ymin && y <= ymax) {
            return -std::min({x - xmin, xmax - x, y - ymin, ymax - y});
        }
        return std::sqrt(dx*dx + dy*dy);
    }
};

struct Boxes {
  Box* boxes;
  int n;
  ~Boxes() {delete[] boxes;}
};

void* make_boxes(double* xmin, double* xmax, double* ymin, double* ymax, int n) {
  auto bs = new Boxes;
  bs->boxes = new Box[n];
  bs->n = n;
  for (int i = 0; i < n; ++i) {
    bs->boxes[i] = {xmin[i], xmax[i], ymin[i], ymax[i]};
  }
  return bs;
}

double signed_distance(double x, double y, void* boxes) {
  Boxes* bs = static_cast<Boxes*>(boxes);
  double min_dist = std::numeric_limits<double>::infinity();
  for (int i = 0; i < bs->n; ++i) {
    min_dist = std::min(min_dist, bs->boxes[i].signed_distance(x, y));
  }
  return min_dist;
}

void signed_distance_batch(double* x, double* y, double* dist, int n, void* boxes) {
  Boxes* bs = static_cast<Boxes*>(boxes);
  for (int i = 0; i < n; ++i) {
    dist[i] = signed_distance(x[i], y[i], bs);
  }
}

void delete_boxes(void* boxes) {
  delete static_cast<Boxes*>(boxes);
}

void* create_parametric_maze_boxes(double* param, int n_dof, double wall_thickness, double hole_size, double y_length) {
  std::vector<double> xmins(2 * n_dof);
  std::vector<double> xmaxs(2 * n_dof);
  std::vector<double> ymins(2 * n_dof);
  std::vector<double> ymaxs(2 * n_dof);
  double half_wall_thickness = wall_thickness * 0.5;
  double half_hole_size = hole_size * 0.5;

  double interval = y_length / (n_dof + 1.0);
  for(int i = 0; i < n_dof; i++) {
    auto wall_y = interval * (i + 1);
    auto holl_x = param[i];

    // left
    xmins[2 * i + 0] = 0.0;
    xmaxs[2 * i + 0] = holl_x - half_hole_size;
    ymins[2 * i + 0] = wall_y - half_wall_thickness;
    ymaxs[2 * i + 0] = wall_y + half_wall_thickness;

    // right
    xmins[2 * i + 1] = holl_x + half_hole_size;
    xmaxs[2 * i + 1] = 1.0;
    ymins[2 * i + 1] = wall_y - half_wall_thickness;
    ymaxs[2 * i + 1] = wall_y + half_wall_thickness;
  }
  void* boxes = make_boxes(xmins.data(), xmaxs.data(), ymins.data(), ymaxs.data(), 2 * n_dof);
  return boxes;
}
