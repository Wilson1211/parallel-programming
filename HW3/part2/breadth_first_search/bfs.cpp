#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    //int *new_frontier_count = &(new_frontier->count);
    #pragma omp parallel for// shared(new_frontier_count)
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        int distance_new = distances[node] + 1;
        int cnt = 0;
        int tmp[end_edge - start_edge];
        int bb;
        int old_val;
        int index;
        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            //#pragma omp critical
            //{
            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                //distances[outgoing] = distances[node] + 1;
                //distances[outgoing] = distance_new;
                bb = __sync_bool_compare_and_swap(&(distances[outgoing]),NOT_VISITED_MARKER, distance_new);
                //printf("bb: %d\n", bb);
                //printf("i: %d, distances[outgoing]: %d\n", i, distances[outgoing]);
                
                //index = new_frontier->count++;
                
                //bb = __sync_bool_compare_and_swap(&(distances[outgoing]),NOT_VISITED_MARKER, distance_new);
                //cnt++;
                //new_frontier->vertices[index] = outgoing;
                tmp[cnt++] = outgoing;
            }
            //}
        }
        
        if(cnt){
/*            
           #pragma omp critical
            {
                old_val = new_frontier->count;
                new_frontier->count += cnt;
            }
*/
            do{
                old_val = new_frontier->count;
                //printf("old_val: %d", old_val);
            }while(!(__sync_bool_compare_and_swap(&(new_frontier->count), old_val, old_val+cnt)));


            for(int i=0;i<cnt;i++){
                new_frontier->vertices[old_val+i] = tmp[i];
            }
            
        }
        

    }
    
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
/*    for (int i = 0; i < graph->num_nodes; i++){
        printf("top-down i: %d, dis: %d\n", i, sol->distances[i]);
    }
    */
}

void bottom_up_step(
    Graph g,
    vertex_set *not_access_node,
    vertex_set *frontier_index,
    int *distances,
    int count, 
    int *vis)
{

    //int cnt = 0;
    int numNodes = g->num_nodes;
    //int tmp[g->num_nodes] = {};

    #pragma omp parallel for
    for(int i=0;i<not_access_node->count;i++) {
        int node = not_access_node->vertices[i];

        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[node + 1];


        int bb;//
        long int old_val = 0;
       // int cnt = 0;
        //int tmp[end_edge-start_edge];
        int flag = 0;
        for(int neighbor=start_edge;neighbor<end_edge;neighbor++){
            int incoming = g->incoming_edges[neighbor];
            if(vis[incoming]){
                //tmp[cnt++] = outgoing;
                distances[node] = count;
                //vis[node] = true;
                flag = 1;
                break;
            }
        }

        if(!flag){
            /*
            do{
                old_val = frontier_index->count;
            }while(!(__sync_bool_compare_and_swap(&(frontier_index->count), old_val, old_val+1)));
            */
            old_val = __sync_fetch_and_add(&(frontier_index->count), 1);
            //printf("old_val: %d\n", old_val);
            frontier_index->vertices[old_val] = node;

        }//else{
            /*
            do{
                old_val = cnt;
                printf("i: %d, cnt: %d\n", i, cnt);
            }while(!(__sync_bool_compare_and_swap(&(cnt), old_val, cnt+1)));
            tmp[cnt] = node;
            
            */
            //tmp[node] = count;
        //}
        

    }
/*
    for(int i=0;i<;i++) {
        distances[tmp[i]] = count;
    }    
    */
/*   
   if(not_access_node->count < 10){
        printf("leave first for loop\n");
   }
   #pragma omp parallel for
   for(int i=0;i<numNodes;i++){
       if(tmp[i]){
           distances[i] = tmp[i];
       }
   }
   if(not_access_node->count<10){
       printf("leave step\n");
   }
*/
    for(int i=0;i<numNodes;i++) {
        if(distances[i] != NOT_VISITED_MARKER){
            vis[i] = true;
        }
    }
}


void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *not_access_node = &list1;
    vertex_set *frontier_index = &list2;

    int *vis = (int*)malloc((graph->num_nodes)*sizeof(int));
    //memset(vis, 0, (graph->num_nodes)*sizeof(int));
    for(int i=0;i<graph->num_nodes;i++){
        vis[i] = false;
    }

    for (int i = 1; i < graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        not_access_node->vertices[i-1] = i;
    }
    sol->distances[ROOT_NODE_ID] = 0;
    vis[0] = true;
    not_access_node->count = graph->num_nodes-1;
    

    int count = 1;
    int cnt = 0;
    int i;


    while (not_access_node->count != 0)
    {
        //printf("not_access_node_count: %d\n", not_access_node->count);

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(frontier_index);

        bottom_up_step(graph, not_access_node, frontier_index, sol->distances, count, vis);
        count++;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = not_access_node;
        not_access_node = frontier_index;
        frontier_index = tmp;
        
       /*
        for(i=0;i<frontier_index->count-1;i++) {
            for(int j=frontier_index->vertices[i];j<frontier_index->vertices[i+1];j++){
                not_access_node->vertices[j] = not_access_node->vertices[j+1];
            }
        }

        i = frontier_index->vertices[i];
        while(i<not_access_node->count){
            not_access_node->vertices[i] = not_access_node->vertices[++i];
        }
*/

        //not_access_node->count = not_access_node->count - frontier_index->count;
    }

    //for (int i = 0; i < graph->num_nodes; i++){
    //    printf("bottom-up i: %d, dis: %d\n", i, sol->distances[i]);
    //}
    free(vis);
}


void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
