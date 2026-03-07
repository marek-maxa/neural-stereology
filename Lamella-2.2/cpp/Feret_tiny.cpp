#include </home/oleksandr/Documents/voro/src/voro++.cc>
#include <random>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <cmath>
#include <cstdlib>
#include <limits>

using namespace std;
using namespace voro;

typedef struct { double x, y, z; } point;

int N = 100;
float delta = 0.5;
bool periodic = false;
double x, y, z, xx, yy, zz, rr;

int izk, jzk, i, j, k, l, r, n, nx, ny, nz, en, id;


    // Looking for empty cells
int empty_cells(voro::container_poly &con) {

	voro::voronoicell c;
	int i, j, ni;
	ni = 0;
	bool cell;

	for (j = 0; j < con.nxyz; j++) { // loop over boxes
		for (i = 0; i < con.co[j]; i++) { // loop over particles in considered box

			cell = con.compute_cell(c, j, i);
			if (cell == false) { ni = ni + 1;

			}

		}
	}

	return ni;

}


    // Computing the orthogonal projection of a point onto a line
double closest_point_on_line(point a, point b, point p) {
    // a,b - point and vector determining the line, p - point to be projected on the line

	double apx = p.x - a.x;
	double apy = p.y - a.y;
	double apz = p.z - a.z;

	double s2 = b.x * b.x + b.y * b.y + b.z * b.z;
	double s1 = apx * b.x + apy * b.y + apz * b.z;

	return (s1 / s2);
}


    // Computing Feret projection
void compute_line_segments(voro::container_poly &conp, std::vector<point> &normals, std::vector<point> &generators, std::vector<double> &a, std::vector<double> &b, std::vector<double> &cvol) {
	//  [in]		conp		container with stored generators
	//	[in]		normals		normal vectors
	//	[in,out]	a,b			vectors of end points of the line segments
	//	[in,out]	cvol		vector of cell volumes

	int i, j;
	voro::voronoicell_neighbor c; // cell
	std::vector<double> v;
	double ml;
	point gen, vec;

	// loop over all generators:
	for (j = 0; j < conp.nxyz; j++) { // loop over boxes
		for (i = 0; i < conp.co[j]; i++) { // loop over generators in considered box

			if (conp.compute_cell(c, j, i)) { // if the grain (Laguerre cell) is nonempty

				cvol[conp.id[j][i] - 1] = c.volume(); // compute volume of the cell and store it into the vector `cvol'

				gen = generators[conp.id[j][i]-1];

				// compute vertices of the cell and store them into the vector `v'
				c.vertices(gen.x, gen.y, gen.z, v);

				// find the closest point on the line determined by generator position and normal vector to the first vertex
				vec = { v[0],v[1],v[2] };
				ml = closest_point_on_line(gen, normals[conp.id[j][i] - 1], vec);
				a[conp.id[j][i] - 1] = ml; b[conp.id[j][i] - 1] = ml; // actualize the line segment

				// loop over vertices (the first vertex is skipped)
				for (int k = 3; k < v.size(); k = k + 3) {

					// find the closest point on the line determined by generator position and normal vector to the vertex
					vec = { v[k],v[k + 1],v[k + 2] };
					ml = closest_point_on_line(gen, normals[conp.id[j][i] - 1], vec);

					// actualize the line segment
					if (ml < a[conp.id[j][i] - 1]) { a[conp.id[j][i] - 1] = ml; }
					if (ml > b[conp.id[j][i] - 1]) { b[conp.id[j][i] - 1] = ml; }
				}

			}
		}
	}

	return;
}



int main(int argc, char* argv[]) {
    // Check for correct argument count
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <conname_path> <nname_path>\n";
        return 1;
    }

    // Set conname and nname paths from command-line arguments
    const char* conname = argv[1];
    const char* nname = argv[2];

    // Precision for output
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(6);

    // Fundamental work with the container
    voro::pre_container_poly pconp(-delta, 1+delta, -delta, 1+delta, -delta, 1+delta, periodic, periodic, periodic);

    // Import conname data
    pconp.import(conname);
    pconp.guess_optimal(nx, ny, nz);
    voro::container_poly conp(-delta, 1+delta, -delta, 1+delta, -delta, 1+delta, nx, ny, nz, periodic, periodic, periodic, 8);
    pconp.setup(conp);

    wall_plane p1(1,0,0,1);
    conp.add_wall(p1);
    wall_plane p2(0,1,0,1);
    conp.add_wall(p2);
    wall_plane p3(0,0,1,1);
    conp.add_wall(p3);

    wall_plane p4(-1,0,0,0);
    conp.add_wall(p4);
    wall_plane p5(0,-1,0,0);
    conp.add_wall(p5);
    wall_plane p6(0,0,-1,0);
    conp.add_wall(p6);

    n = conp.total_particles();
    en = empty_cells(conp);

    cout << n << " " << en << "\n";

    // Import normals data
    std::vector<point> ns;
    std::ifstream infile(nname);
    while (infile >> id >> xx >> yy >> zz) {
        rr = sqrt(xx * xx + yy * yy + zz * zz);
        ns.push_back({ xx / rr, yy / rr, zz / rr }); }

    infile.close();

    // Import generators and radii from conname
    std::vector<point> generators;
    std::vector<double> radii;
    std::ifstream infile2(conname);
    while (infile2 >> id >> xx >> yy >> zz >> rr) {
        generators.push_back({ xx, yy, zz });
        radii.push_back(rr); }

    infile2.close();

    // Computing the line segment
    std::vector<double> a(generators.size()), b(generators.size()), cvol(generators.size());
    compute_line_segments(conp, ns, generators, a, b, cvol);

    // Computing the volume function approximation
    std::vector<std::vector<double>> funE(generators.size(), std::vector<double>(N + 1, 0));
    voro::voronoicell_neighbor v, d;

    for (int j = 0; j < conp.nxyz; j++) {
        for (int i = 0; i < conp.co[j]; i++) {
            funE[conp.id[j][i] - 1][0] = 0;
            for (int k = 1; k <= N; k++) {
                conp.compute_cell(d, j, i);
                v = d;
                v.plane(ns[conp.id[j][i] - 1].x, ns[conp.id[j][i] - 1].y, ns[conp.id[j][i] - 1].z,
                        2 * (a[conp.id[j][i] - 1] + (b[conp.id[j][i] - 1] - a[conp.id[j][i] - 1]) * double(k) / N));
                funE[conp.id[j][i] - 1][k] = v.volume();
            }
        }
    }

    /*
    // Save the volume function and Feret projection data
    std::ofstream outVF("/home/oleksandr//volQ");
    for (int i = 0; i < n; i++) {
        for (int k = 0; k <= N; k++) outVF << funE[i][k] << " ";
        outVF << "\n";
    }
    */
    std::ofstream outFeret("./data/feret_small");

    for (int j = 0; j < a.size(); j++) {

            outFeret << a[j] << " " << b[j] << "\n";
        }

    return 0;
}









