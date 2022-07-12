#include <stdint.h>
#include <math.h>
#include "bsg_manycore.h"
#include "appl.hpp"

const float L = 1, W = 1, dt = 1e-3, alpha = 0.25, V = 50, epsilon = 1e-1, grav = 0.04; //grav should be 100/N
float *x, *y, *u, *v, *force_x, *force_y, *mass;

/*
 * Struct that represents a node of the Barnes Hut quad tree.
 */
struct node_t
{
    int particle;
    int has_particle;
    int has_children;
    float min_x, max_x, min_y, max_y, total_mass, c_x, c_y;
    struct node_t *children;
};

//Functions for handling the placement of particles in the tree
void put_particle_in_tree(int new_particle, struct node_t *node);
void place_particle(int particle, struct node_t *node);
void set_node(struct node_t *node);
void free_node(struct node_t *node);
void display_tree(struct node_t *node);

//Functions for calculating the mass and centre of mass of the tree
float calculate_mass(struct node_t *node);
float calculate_center_of_mass_x(struct node_t *node);
float calculate_center_of_mass_y(struct node_t *node);

//Functions for the force calculations
void update_forces();
void update_forces_help(int particle, struct node_t *node);
void calculate_force(int particle, struct node_t *node, float r);

/*
 * Sets initial values for a new node
 */
void set_node(struct node_t *node)
{
    node->has_particle = 0;
    node->has_children = 0;
}

/*
 * Puts a particle recursively in the Barnes Hut quad-tree.
 */
void put_particle_in_tree(int new_particle, struct node_t *node)
{
    //If no particle is assigned to the node
    if (!node->has_particle)
    {
        node->particle = new_particle;
        node->has_particle = 1;
    }
    //If the node has no children
    else if (!node->has_children)
    {
        //Allocate and initiate children
        node->children = (struct node_t*)appl::appl_malloc(4 * sizeof(struct node_t));

        for (int i = 0; i < 4; i++)
        {
            set_node(&node->children[i]);
        }

        //Set boundaries for the children
        node->children[0].min_x = node->min_x;
        node->children[0].max_x = (node->min_x + node->max_x) / 2;
        node->children[0].min_y = node->min_y;
        node->children[0].max_y = (node->min_y + node->max_y) / 2;

        node->children[1].min_x = (node->min_x + node->max_x) / 2;
        node->children[1].max_x = node->max_x;
        node->children[1].min_y = node->min_y;
        node->children[1].max_y = (node->min_y + node->max_y) / 2;

        node->children[2].min_x = node->min_x;
        node->children[2].max_x = (node->min_x + node->max_x) / 2;
        node->children[2].min_y = (node->min_y + node->max_y) / 2;
        node->children[2].max_y = node->max_y;

        node->children[3].min_x = (node->min_x + node->max_x) / 2;
        node->children[3].max_x = node->max_x;
        node->children[3].min_y = (node->min_y + node->max_y) / 2;
        node->children[3].max_y = node->max_y;

        //Put old particle into the appropriate child
        place_particle(node->particle, node);

        //Put new particle into the appropriate child
        place_particle(new_particle, node);

        //It now has children
        node->has_children = 1;
    }
    //Add the new particle to the appropriate children
    else
    {
        //Put new particle into the appropriate child
        place_particle(new_particle, node);
    }
}

/*
 * Puts a particle in the right child of a node with children.
 */
void place_particle(int particle, struct node_t *node)
{
    if (x[particle] <= (node->min_x + node->max_x) / 2 && y[particle] <= (node->min_y + node->max_y) / 2)
    {
        put_particle_in_tree(particle, &node->children[0]);
    }
    else if (x[particle] > (node->min_x + node->max_x) / 2 && y[particle] < (node->min_y + node->max_y) / 2)
    {
        put_particle_in_tree(particle, &node->children[1]);
    }
    else if (x[particle] < (node->min_x + node->max_x) / 2 && y[particle] > (node->min_y + node->max_y) / 2)
    {
        put_particle_in_tree(particle, &node->children[2]);
    }
    else
    {
        put_particle_in_tree(particle, &node->children[3]);
    }
}

/*
 * Calculates the total mass for the node. It recursively updates the mass
 * of itself and all of its children.
 */
float calculate_mass(struct node_t *node)
{
    if (!node->has_particle)
    {
        node->total_mass = 0;
    }
    else if (!node->has_children)
    {
        node->total_mass = mass[node->particle];
    }
    else
    {
        int mass1, mass2, mass3, mass4;
        appl::parallel_invoke(
            [&mass1, node] { mass1 = calculate_mass(&node->children[0]); },
            [&mass2, node] { mass2 = calculate_mass(&node->children[1]); },
            [&mass3, node] { mass3 = calculate_mass(&node->children[2]); },
            [&mass4, node] { mass4 = calculate_mass(&node->children[3]); } );

        node->total_mass = (mass1 + mass2 + mass3 + mass4);
    }
    return node->total_mass;
}

void center_of_mass_x_helper(struct node_t *node, float& c_x, float& m_tot) {
  if (node->has_particle) {
    c_x = node->total_mass * calculate_center_of_mass_x(node);
    m_tot = node->total_mass;
  }
}

void center_of_mass_y_helper(struct node_t *node, float& c_y, float& m_tot) {
  if (node->has_particle) {
    c_y = node->total_mass * calculate_center_of_mass_y(node);
    m_tot = node->total_mass;
  }
}

/*
 * Calculates the x-position of the centre of mass for the
 * node. It recursively updates the position of itself and
 * all of its children.
 */
float calculate_center_of_mass_x(struct node_t *node)
{
    if (!node->has_children)
    {
        node->c_x = x[node->particle];
    }
    else
    {
        float tmp_c_x1 = 0;
        float tmp_c_x2 = 0;
        float tmp_c_x3 = 0;
        float tmp_c_x4 = 0;
        float tmp_m1   = 0;
        float tmp_m2   = 0;
        float tmp_m3   = 0;
        float tmp_m4   = 0;
        appl::parallel_invoke(
            [&tmp_c_x1, &tmp_m1, node] { center_of_mass_x_helper(&node->children[0], tmp_c_x1, tmp_m1); },
            [&tmp_c_x2, &tmp_m2, node] { center_of_mass_x_helper(&node->children[1], tmp_c_x2, tmp_m2); },
            [&tmp_c_x3, &tmp_m3, node] { center_of_mass_x_helper(&node->children[2], tmp_c_x3, tmp_m3); },
            [&tmp_c_x4, &tmp_m4, node] { center_of_mass_x_helper(&node->children[3], tmp_c_x4, tmp_m4); } );
        node->c_x = (tmp_c_x1 + tmp_c_x2 + tmp_c_x3 + tmp_c_x4);
        node->c_x /= (tmp_m1 + tmp_m2 + tmp_m3 + tmp_m4);
    }
    return node->c_x;
}

/*
 * Calculates the y-position of the centre of mass for the
 * node. It recursively updates the position of itself and
 * all of its children.
 */
float calculate_center_of_mass_y(struct node_t *node)
{
    if (!node->has_children)
    {
        node->c_y = y[node->particle];
    }
    else
    {
        float tmp_c_y1 = 0;
        float tmp_c_y2 = 0;
        float tmp_c_y3 = 0;
        float tmp_c_y4 = 0;
        float tmp_m1   = 0;
        float tmp_m2   = 0;
        float tmp_m3   = 0;
        float tmp_m4   = 0;
        appl::parallel_invoke(
            [&tmp_c_y1, &tmp_m1, node] { center_of_mass_y_helper(&node->children[0], tmp_c_y1, tmp_m1); },
            [&tmp_c_y2, &tmp_m2, node] { center_of_mass_y_helper(&node->children[1], tmp_c_y2, tmp_m2); },
            [&tmp_c_y3, &tmp_m3, node] { center_of_mass_y_helper(&node->children[2], tmp_c_y3, tmp_m3); },
            [&tmp_c_y4, &tmp_m4, node] { center_of_mass_y_helper(&node->children[3], tmp_c_y4, tmp_m4); } );
        node->c_y = (tmp_c_y1 + tmp_c_y2 + tmp_c_y3 + tmp_c_y4);
        node->c_y /= (tmp_m1 + tmp_m2 + tmp_m3 + tmp_m4);
    }
    return node->c_y;
}

/*
 * Calculates the forces in a time step of all particles in
 * the simulation using the Barnes Hut quad tree.
 */
void update_forces(int N, struct node_t* root)
{
    appl::parallel_for(0, N, [root](int i) {
          force_x[i] = 0;
          force_y[i] = 0;
          update_forces_help(i, root);
        } );
}

/*
 * Help function for calculating the forces recursively
 * using the Barnes Hut quad tree.
 */
void update_forces_help(int particle, struct node_t *node)
{
    //The node is a leaf node with a particle and not the particle itself
    if (!node->has_children && node->has_particle && node->particle != particle)
    {
        float r = sqrt((x[particle] - node->c_x) * (x[particle] - node->c_x) + (y[particle] - node->c_y) * (y[particle] - node->c_y));
        calculate_force(particle, node, r);
    }
    //The node has children
    else if (node->has_children)
    {
        //Calculate r and theta
        float r = sqrt((x[particle] - node->c_x) * (x[particle] - node->c_x) + (y[particle] - node->c_y) * (y[particle] - node->c_y));
        float theta = (node->max_x - node->min_x) / r;

        /* If the distance to the node's centre of mass is far enough, calculate the force,
     * otherwise traverse further down the tree
     */
        if (theta < 0.5)
        {
            calculate_force(particle, node, r);
        }
        else
        {
            appl::parallel_invoke(
                [particle, node] { update_forces_help(particle, &node->children[0]); },
                [particle, node] { update_forces_help(particle, &node->children[1]); },
                [particle, node] { update_forces_help(particle, &node->children[2]); },
                [particle, node] { update_forces_help(particle, &node->children[3]); } );
        }
    }
}

/*
 * Calculates and updates the force of a particle from a node.
 */
void calculate_force(int particle, struct node_t *node, float r)
{
    float temp = -grav * mass[particle] * node->total_mass / ((r + epsilon) * (r + epsilon) * (r + epsilon));
    force_x[particle] += (x[particle] - node->c_x) * temp;
    force_y[particle] += (y[particle] - node->c_y) * temp;
}

/*
 * If a particle moves beyond any of the boundaries then bounce it back
 */
void bounce(float *x, float *y, float *u, float *v)
{
    float W = 1.0f, H = 1.0f;
    if (*x > W)
    {
        *x = 2 * W - *x;
        *u = -*u;
    }

    if (*x < 0)
    {
        *x = -*x;
        *u = -*u;
    }

    if (*y > H)
    {
        *y = 2 * H - *y;
        *v = -*v;
    }

    if (*y < 0)
    {
        *y = -*y;
        *v = -*v;
    }
}

struct node_t* construct_tree(int N) {
  struct node_t* root = (struct node_t*)appl::appl_malloc(sizeof(struct node_t));
  set_node(root);
  root->min_x = 0;
  root->max_x = 1;
  root->min_y = 0;
  root->max_y = 1;
  for (int i = 0; i < N; i++) {
    bsg_print_int(i);
    put_particle_in_tree(i, root);
  }
  return root;
}

extern "C" __attribute__ ((noinline))
int kernel_appl_barnes(int* results, float* _x, float* _y, float* _u, float* _v,
                       float* _force_x, float* _force_y, float* _mass, int N,
                       int* dram_buffer) {

  x       = _x;
  y       = _y;
  u       = _u;
  v       = _v;
  force_x = _force_x;
  force_y = _force_y;
  mass    = _mass;
  // debug print
  if (__bsg_id == 0) {
    bsg_print_int(N);
  }

  // --------------------- kernel ------------------------
  appl::runtime_init(dram_buffer);

  // sync
  appl::sync();

  // construct the quadtree
  struct node_t* root;
  if (__bsg_id == 0) {
    root = construct_tree(N);
  }

  appl::sync();
  bsg_cuda_print_stat_kernel_start();

  if (__bsg_id == 0) {
    //Calculate mass and center of mass
    calculate_mass(root);
    calculate_center_of_mass_x(root);
    calculate_center_of_mass_y(root);

    //Calculate forces
    update_forces(N, root);

    //Update velocities and positions
    appl::parallel_for(0, N, [](int i) {
        float ax = force_x[i] / mass[i];
        float ay = force_y[i] / mass[i];
        u[i] += ax * dt;
        v[i] += ay * dt;
        x[i] += u[i] * dt;
        y[i] += v[i] * dt;

        /* This of course doesn't make any sense physically,
     * but makes sure that the particles stay within the
     * bounds. Normally the particles won't leave the
     * area anyway.
     */
        bounce(&x[i], &y[i], &u[i], &v[i]);
      } );
  } else {
    appl::worker_thread_init();
  }

  appl::runtime_end();
  // --------------------- end of kernel -----------------

  bsg_cuda_print_stat_kernel_end();

  appl::sync();
  return 0;
}
