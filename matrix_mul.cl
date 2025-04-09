// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

 // ACL kernel for adding two input vectors
import pyopencl as cl
import numpy as np
__kernel void vector_add(__global const float *x, 
                         __global const float *y, 
                         __global float *restrict z)
{
    platforms = cl.get_platforms()
    cpu_devices = [device for device in platforms[0].get_devices(device_type=cl.device_type.GPU)]
    cpu_devices

    context = cl.Context(devices=cpu_devices)

    # Create a command queue for the target device
    queue = cl.CommandQueue(context)

    file_name = "./device/hidden.cl"  # Replace with the name of your uploaded .cl file
    with open(file_name, 'r') as file:
        kernel_code = file.read()

    program = cl.Program(context, kernel_code).build()

    input_tile_size = 16
    output_neurons_tile_size = 10

    # Initialize random data for the input tile and weights
    input_tile = np.random.rand(input_tile_size).astype(np.float32)
    weights_tile = np.random.rand(input_tile_size * output_neurons_tile_size).astype(np.float32)

    output_tile = np.zeros(output_neurons_tile_size).astype(np.float32)

    # Create memory buffers
    input_tile_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_tile)
    weights_tile_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=weights_tile)
    output_tile_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output_tile.nbytes)

    # Build the kernel
    program = cl.Program(context, kernel_code).build()

    # Execute the kernel
    global_size = (output_tile.size,)
    local_size = None
    program.hidden(queue, global_size, local_size,
                input_tile_buf, weights_tile_buf,
                np.int32(input_tile_size), np.int32(output_neurons_tile_size),
                output_tile_buf)

    # Read the output buffer back to the host
    cl.enqueue_copy(queue, output_tile, output_tile_buf)

    # Output the results
    print(output_tile)

    def matrix_vector_multiply(input_tile, weights_tile, input_tile_size, output_neurons_tile_size):
        # Reshape weights_tile to be a 2D array for matrix multiplication
        weights_matrix = weights_tile.reshape((output_neurons_tile_size, input_tile_size))

        # Perform matrix-vector multiplication
        output_tile = np.dot(weights_matrix, input_tile)

        return output_tile

    test_output = matrix_vector_multiply(input_tile, weights_tile, input_tile_size, output_neurons_tile_size)

    test_output
}

