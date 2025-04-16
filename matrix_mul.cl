__kernel void matrix_mul(
    __global const float* input_tile,         // Tile of the Input vector
    __global const float* weights_tile,       // Tile of the Weights matrix (flattened row-major)
    const int input_tile_size,                // Size of the input tile
    const int output_neurons_tile_size,       // Size of the output tile (number of neurons)
    __global float* output_tile               // Output vector tile
) {
    // Calculate the ID of the neuron this thread will compute
    int neuron_id = get_global_id(0);

    // Ensure we don't go beyond the size of the output tile
    if (neuron_id < output_neurons_tile_size) {
        float sum = 0.0f;

        for (int i = 0; i < input_tile_size; i++) {
            int weight_index = neuron_id * input_tile_size + i;
            sum += weights_tile[weight_index] * input_tile[i];
        }

        output_tile[neuron_id] = sum;
    }
}

