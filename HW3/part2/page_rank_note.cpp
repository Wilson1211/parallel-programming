int* node_counts = (int*)malloc(sizeof(int) * num_nodes); //number of incoming edges per node
int* node_scatter = (int*)malloc(sizeof(int) * num_nodes); //number of outcoming edges per node

graph->incoming_starts = (int*)malloc(sizeof(int) * num_nodes);
graph->incoming_edges = (int*)malloc(sizeof(int) * graph->num_edges);

graph->outgoing_starts = (int*)malloc(sizeof(int) * num_nodes);
graph->outgoing_edges = (int*)malloc(sizeof(int) * graph->num_edges);

int* scratch = (int*) malloc(sizeof(int) * (graph->num_nodes + graph->num_edges));


  for(int i = 0; i < num_nodes; i++)
  {
    graph->outgoing_starts[i] = scratch[i];
  }

// compute number of incoming edges per node
    for (int i=0; i<num_nodes; i++) {
        int start_edge = graph->outgoing_starts[i];
        int end_edge = (i == graph->num_nodes-1) ? graph->num_edges : graph->outgoing_starts[i+1];
        for (int j=start_edge; j<end_edge; j++) {
            int target_node = graph->outgoing_edges[j];
            node_counts[target_node]++;
            total_edges++;
        }
    }

// build the starts array
    graph->incoming_starts[0] = 0;
    for (int i=1; i<num_nodes; i++) {
        graph->incoming_starts[i] = graph->incoming_starts[i-1] + node_counts[i-1];
        //printf("%d: %d ", i, graph->incoming_starts[i]);
    }

    // now perform the scatter
    for (int i=0; i<num_nodes; i++) {
        int start_edge = graph->outgoing_starts[i];
        int end_edge = (i == graph->num_nodes-1) ? graph->num_edges : graph->outgoing_starts[i+1];
        for (int j=start_edge; j<end_edge; j++) {
            int target_node = graph->outgoing_edges[j];
            graph->incoming_edges[graph->incoming_starts[target_node] + node_scatter[target_node]] = i;
            node_scatter[target_node]++;
        }
    }

