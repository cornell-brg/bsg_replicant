#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <bsg_manycore_regression.h>

#define ALLOC_NAME "default_allocator"
#define MAX_WORKERS 128
#define HB_L2_CACHE_LINE_WORDS 16
#define BUF_FACTOR 16385
#define BUF_SIZE (MAX_WORKERS * HB_L2_CACHE_LINE_WORDS * BUF_FACTOR)

const int N = NBODY;
const float L = 1, W = 1, dt = 1e-3, alpha = 0.25, V = 50, epsilon = 1e-1, grav = 0.04; //grav should be 100/N
float *x, *y, *u, *v, *force_x, *force_y, *mass;
struct node_t *root;

eva_t device_x;
eva_t device_y;
eva_t device_u;
eva_t device_v;
eva_t device_force_x;
eva_t device_force_y;

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
 * Function to read a case
 */
int read_case(char *filename)
{
    int i, n;
    FILE *arq = fopen(filename, "r");

    if(arq == NULL)
    {
        printf("Error: The file %s could not be opened.\n", filename);
        return 1;
    }

    int max_N;
    n = fscanf(arq, "%d", &max_N);

    if (n != 1 || N > max_N)
    {
        printf("Error: The file %s could not be read for number of particles.\n", filename);
        fclose(arq);
        return 1;
    }

    //Initiate memory for the vectors
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    u = (float *)malloc(N * sizeof(float));
    v = (float *)malloc(N * sizeof(float));
    force_x = (float *)calloc(N, sizeof(float));
    force_y = (float *)calloc(N, sizeof(float));
    mass = (float *)malloc(N * sizeof(float));

    if (x == NULL || y == NULL || u == NULL || v == NULL || force_x == NULL || force_y == NULL || mass == NULL)
    {
        printf("Error: Some malloc won't work.\n");
        fclose(arq);
        return 1;
    }

    for (i = 0; i < N; i++)
    {
        n = fscanf(arq, "%f %f %f %f %f", &mass[i], &x[i], &y[i], &u[i], &v[i]);

        if (n != 5)
        {
            printf("Error: Some reading won't work at line %d (%d).\n", i + 1, n);
            fclose(arq);
            return 1;
        }
    }

    if (filename)
    {
        fclose(arq);
    }

    return 0;
}

/*
 * Function to free a case
 */
void free_case()
{
    free(x);
    free(y);
    free(u);
    free(v);
    free(force_x);
    free(force_y);
    free(mass);
}

/*
 * Prints statistics: time, N, final velocity, final center of mass
 */
void print_statistics(float ut, float vt, float xc, float xy)
{
    printf("%.5f %.5f\n", ut, vt);
    printf("%.5f %.5f\n", xc, xy);
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

/*
 * Updates the positions of the particles of a time step.
 */
void time_step(void)
{
    //Allocate memory for root
    root = malloc(sizeof(struct node_t));
    set_node(root);
    root->min_x = 0;
    root->max_x = 1;
    root->min_y = 0;
    root->max_y = 1;

    //Put particles in tree
    for (int i = 0; i < N; i++)
    {
        put_particle_in_tree(i, root);
    }

    //Calculate mass and center of mass
    calculate_mass(root);
    calculate_center_of_mass_x(root);
    calculate_center_of_mass_y(root);

    //Calculate forces
    update_forces();

    //Update velocities and positions
    for (int i = 0; i < N; i++)
    {
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
    }

    //Free memory
    free_node(root);
    free(root);
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
        node->children = malloc(4 * sizeof(struct node_t));

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
 * Sets initial values for a new node
 */
void set_node(struct node_t *node)
{
    node->has_particle = 0;
    node->has_children = 0;
}

/*
 * Frees memory for a node and its children recursively.
 */
void free_node(struct node_t *node)
{
    if (node->has_children)
    {
        free_node(&node->children[0]);
        free_node(&node->children[1]);
        free_node(&node->children[2]);
        free_node(&node->children[3]);
        free(node->children);
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
        node->total_mass = 0;

        for (int i = 0; i < 4; i++)
        {
            node->total_mass += calculate_mass(&node->children[i]);
        }
    }

    return node->total_mass;
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
        node->c_x = 0;
        float m_tot = 0;

        for (int i = 0; i < 4; i++)
        {
            if (node->children[i].has_particle)
            {
                node->c_x += node->children[i].total_mass * calculate_center_of_mass_x(&node->children[i]);
                m_tot += node->children[i].total_mass;
            }
        }

        node->c_x /= m_tot;
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
        node->c_y = 0;
        float m_tot = 0;

        for (int i = 0; i < 4; i++)
        {
            if (node->children[i].has_particle)
            {
                node->c_y += node->children[i].total_mass * calculate_center_of_mass_y(&node->children[i]);
                m_tot += node->children[i].total_mass;
            }
        }

        node->c_y /= m_tot;
    }

    return node->c_y;
}

/*
 * Calculates the forces in a time step of all particles in
 * the simulation using the Barnes Hut quad tree.
 */
void update_forces()
{
    for (int i = 0; i < N; i++)
    {
        force_x[i] = 0;
        force_y[i] = 0;
        update_forces_help(i, root);
    }
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
            update_forces_help(particle, &node->children[0]);
            update_forces_help(particle, &node->children[1]);
            update_forces_help(particle, &node->children[2]);
            update_forces_help(particle, &node->children[3]);
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

void hb_time_step(hb_mc_device_t *device) {
  /*****************************************************************************************************************
   * Allocate memory on the device.
   ******************************************************************************************************************/
  eva_t device_result;
  BSG_CUDA_CALL(hb_mc_device_malloc(device, 64 * sizeof(uint32_t), &device_result)); // buffer for return results
  eva_t dram_buffer;
  BSG_CUDA_CALL(hb_mc_device_malloc(device, BUF_SIZE * sizeof(uint32_t), &dram_buffer));

  BSG_CUDA_CALL(hb_mc_device_malloc(device, N * sizeof(float), &device_x));
  BSG_CUDA_CALL(hb_mc_device_malloc(device, N * sizeof(float), &device_y));
  BSG_CUDA_CALL(hb_mc_device_malloc(device, N * sizeof(float), &device_u));
  BSG_CUDA_CALL(hb_mc_device_malloc(device, N * sizeof(float), &device_v));
  BSG_CUDA_CALL(hb_mc_device_malloc(device, N * sizeof(float), &device_force_x));
  BSG_CUDA_CALL(hb_mc_device_malloc(device, N * sizeof(float), &device_force_y));
  eva_t device_mass;
  BSG_CUDA_CALL(hb_mc_device_malloc(device, N * sizeof(float), &device_mass));

  hb_mc_dma_htod_t htod[] = {{
    .d_addr = device_x,
    .h_addr = x,
    .size   = N * sizeof(float)
  }, {
    .d_addr = device_y,
    .h_addr = y,
    .size   = N * sizeof(float)
  }, {
    .d_addr = device_u,
    .h_addr = u,
    .size   = N * sizeof(float)
  }, {
    .d_addr = device_v,
    .h_addr = v,
    .size   = N * sizeof(float)
  }, {
    .d_addr = device_mass,
    .h_addr = mass,
    .size   = N * sizeof(float)
  }};
  BSG_CUDA_CALL(hb_mc_device_dma_to_device(device, htod, 5));

  /*****************************************************************************************************************
   * Define block_size_x/y: amount of work for each tile group
   * Define tg_dim_x/y: number of tiles in each tile group
   * Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y
   ******************************************************************************************************************/
  hb_mc_dimension_t tg_dim = { .x = bsg_tiles_X, .y = bsg_tiles_Y};
  hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};

  /*****************************************************************************************************************
   * Prepare list of input arguments for kernel.
   ******************************************************************************************************************/
  int cuda_argv[] = {device_result, device_x, device_y, device_u, device_v,
                     device_force_x, device_force_y, device_mass, N, dram_buffer};

  /*****************************************************************************************************************
   * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
   ******************************************************************************************************************/
  BSG_CUDA_CALL(hb_mc_kernel_enqueue (device, grid_dim, tg_dim, "kernel_appl_barnes", 10, cuda_argv));

  /*****************************************************************************************************************
   * Launch and execute all tile groups on device and wait for all to finish.
   ******************************************************************************************************************/
  BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(device));

  return;
}


int kernel_appl_barnes (int argc, char **argv) {
        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the Barnes-Hut on one %dx%d tile groups.\n\n", bsg_tiles_X, bsg_tiles_Y);

        srand(time);

        /*****************************************************************************************************************
        * Define path to binary.
        * Initialize device, load binary and unfreeze tiles.
        ******************************************************************************************************************/
        hb_mc_device_t device;
        BSG_CUDA_CALL(hb_mc_device_init(&device, test_name, 0));

        hb_mc_pod_id_t pod;
        hb_mc_device_foreach_pod_id(&device, pod)
        {
                bsg_pr_info("Loading program for test %s onto pod %d\n", test_name, pod);
                BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
                BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));
                /*****************************************************************************************************************
                 * Read data on from simulation setup
                 ******************************************************************************************************************/
                if ( read_case("../galaxy.in") == 1 ) {
                  printf("Error: Failed to read the case\n");
                  exit(1);
                }

                hb_time_step(&device);
                time_step();

                //Compute final statistics
                float vu = 0;
                float vv = 0;
                float sumx = 0;
                float sumy = 0;
                float total_mass = 0;

                for (int i = 0; i < N; i++)
                {
                    sumx += mass[i] * x[i];
                    sumy += mass[i] * y[i];
                    vu += u[i];
                    vv += v[i];
                    total_mass += mass[i];
                }

                float cx = sumx / total_mass;
                float cy = sumy / total_mass;

                printf("On Host:\n");
                print_statistics(vu, vv, cx, cy);

                // copy to host
                float* h_force_x = (float *)malloc(N * sizeof(float));
                float* h_force_y = (float *)malloc(N * sizeof(float));
                hb_mc_dma_dtoh_t dtoh[] = {{
                  .d_addr = device_force_x,
                  .h_addr = h_force_x,
                  .size   = N * sizeof(float)
                }, {
                  .d_addr = device_force_y,
                  .h_addr = h_force_y,
                  .size   = N * sizeof(float)
                }};
                BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, dtoh, 2));

                /*****************************************************************************************************************
                 * verification
                 ******************************************************************************************************************/
                double error = 0.0;
                for (int i = 0; i < N; i++) {
                  bsg_pr_info("particle %d with force_x = (%f, %f) and force_y = (%f, %f)\n", i, h_force_x[i], force_x[i],
                              h_force_y[i], force_y[i]);
                  error += (fabs(h_force_x[i] - force_x[i]));
                  error += (fabs(h_force_y[i] - force_y[i]));
                  if (error > 0.0001) {
                    return HB_MC_FAIL;
                  }
                }

                /*****************************************************************************************************************
                 * Freeze the tiles and memory manager cleanup.
                 ******************************************************************************************************************/
                BSG_CUDA_CALL(hb_mc_device_program_finish(&device));

        }
        BSG_CUDA_CALL(hb_mc_device_finish(&device));

        return HB_MC_SUCCESS;
}

declare_program_main("test_appl_barnes", kernel_appl_barnes);
