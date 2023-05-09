N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

y .+= x             # increment each element of y with the corresponding element of x

using Test
@test all(y .== 3.0f0)


function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)


function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)

using BenchmarkTools
@btime sequential_add!($y, $x)
@btime parallel_add!($y, $x)


using CUDA

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

function bench_gpu4!(y, x)
    kernel = @cuda launch=false gpu_add3!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    display(threads)
    display(blocks)

    CUDA.@sync begin
        kernel(y, x; threads, blocks)
    end
end

bench_gpu4!(y_d, x_d)
@btime bench_gpu4!($y_d, $x_d)

### Add two random arrays ------------------------------------
x_d = CuArray{Float32}([5, 4, 3, 2, 1])  # a vector stored on the GPU with specified values
y_d = CuArray{Float32}([1, 2, 3, 4, 5])  # a vector stored on the GPU with specified values
y_d .+= x_d
@test all(Array(y_d) .== [6, 6, 6, 6, 6])

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

function bench_gpu4!(y, x)
    kernel = @cuda launch=false gpu_add3!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    display(threads)
    display(blocks)

    CUDA.@sync begin
        kernel(y, x; threads, blocks)
    end
end

bench_gpu4!(y_d, x_d)
display(y_d)
@btime bench_gpu4!($y_d, $x_d)
display(y_d) # Note: y_d is modified in bench function, it continuously increments
# This results in 
# 376.200 μs (275 allocations: 11.97 KiB)

x = [5, 4, 3, 2, 1]  
y = [1, 2, 3, 4, 5]  

function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

@btime sequential_add!($y, $x)
display(y)
display(x)

function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)

### Testing -------------------------------------------
all_particles = ones(STATE_VEC_SIZE, N_PARTICLES, N_STEPS)
particles = ones(STATE_VEC_SIZE, N_PARTICLES)
new_particles = ones(STATE_VEC_SIZE, N_PARTICLES)
weights = ones(N_PARTICLES)

dist_pos_init = Normal(0, INIT_STD_POS)
dist_rot_init = Normal(0, INIT_STD_ROT)

dist_pos_update = Normal(0, PF_STD_POS)
dist_rot_update = Normal(0, PF_STD_ROT)


for i in range(1, N_PARTICLES)
    # Generate sample
    x = START_X + rand(dist_pos_init)
    y = START_Y + rand(dist_pos_init)
    θ = START_θ + rand(dist_rot_init)
    
    particles[:,i] = particles[:,i].*[x, y, θ, 0, 0]
end

# Plot initial particles
fig, ax = PyPlot.subplots(figsize=[10,10])
ax.imshow(-bathy_map)
ax.plot(particles[1,:], particles[2,:], "r.")




# Passing functions into kernel ------------------------------------
using CUDAnative

# Define a function to be passed to the kernel
function myfunc(x::Float32)
    return x * 2
end

# Define the kernel that calls the function
function mykernel(y, x, f)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    while i <= length(y)
        @inbounds y[i] = f(x[i])
        i += stride
    end
    return
end

# Call the kernel with the function argument
y = CuArray{Float32}(undef, 10)
x = CuArray{Float32}(undef, 10)
f = CUDAnative.@device_code_warntype(myfunc)

@cuda threads=256 blocks=ceil(Int,length(y)/256) mykernel(y, x, f)



# Passing functions into kernel ------------------------
using CUDA

function myfunc(x)
    return x + 1
end

function mykernel(f, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds A[i] = f(A[i])
    end
    return nothing
end

function bench_gpu_kernel!(f, A)
    kernel = @cuda launch=false mykernel(f, A)
    config = launch_configuration(kernel.fun)
    threads = min(length(A), config.threads)
    blocks = cld(length(A), threads)

    display(threads)
    display(blocks)

    CUDA.@sync begin
        kernel(f, A; threads, blocks)
    end
end


# @cuda threads=16 mykernel(myfunc, A)
# synchronize()
# A
A = CUDA.fill(1.0f0, 1000)
bench_gpu_kernel!(myfunc, A)
A



# Passing functions into kernel, with global variables ------------------------
using CUDA

# Create a 2x2 matrix on the CPU
A = [1 2; 3 4]

# Allocate a 2x2 matrix on the GPU and copy A to it
A_d = CuArray(A)

function myfunc(x)
    A_d[1]
    return x + 1
end

function mykernel(f, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds A[i] = f(A[i])
    end
    return nothing
end

function bench_gpu_kernel!(f, A)
    kernel = @cuda launch=false mykernel(f, A)
    config = launch_configuration(kernel.fun)
    threads = min(length(A), config.threads)
    blocks = cld(length(A), threads)

    display(threads)
    display(blocks)

    CUDA.@sync begin
        kernel(f, A; threads, blocks)
    end
end


# @cuda threads=16 mykernel(myfunc, A)
# synchronize()
# A
A = CUDA.fill(1.0f0, 1000)
bench_gpu_kernel!(myfunc, A)
A





# Nearest pixel lookup ---------------------------------------

using CUDA

# Define a 2D array of pixel values
pixels = Float32[1.1 2 3; 4 5 6; 7 8 9]

# Define the kernel function
# function nearest_pixel_kernel(x::CuArray{Float32}, y::CuArray{Float32}, pixels::CuArray{Int32})
function nearest_pixel_kernel!(x, y, pixels, output)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        # Find the nearest pixel using floor and convert to 1D index
        ind = (floor(Int32, x[i] * (size(pixels, 1)-1)) + 1) + 
              (floor(Int32, y[i] * (size(pixels, 2)-1)) * size(pixels, 1))
        @inbounds output[i] = pixels[ind]
    end
    return nothing
end

# Define input data
n = 10_000
x = CUDA.rand(Float32, n)
y = CUDA.rand(Float32, n)

# Allocate memory on the GPU and copy input data
x_d = cu(x)
y_d = cu(y)
pixels_d = cu(pixels)
output_d = similar(x_d)

# Launch the kernel
block_size = 256
grid_size = cld(n, block_size)
@cuda threads=block_size blocks=grid_size nearest_pixel_kernel!(x_d, y_d, pixels_d, output_d)

# Copy results back to the CPU
output_d
# output = Array(output_d)



# 2d array example - like particles 
# function mykernel(A::CuArray{Float32, 2})
# function mykernel3(A)
#     i, j = (blockIdx().x - 1) * blockDim().x + threadIdx().x, threadIdx().y
#     if i <= size(A, 2)
#         @inbounds A[j, i] += 1
#     end
#     return nothing
# end

# function mykernel3(A)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1)#(blockIdx().x - 1) * blockDim().x + threadIdx().x#blockIdx().y #(blockIdx().y - 1) * blockDim().y + threadIdx().y

#     @inbounds A[j, i] = i
#     # if i <= size(A, 2)
#     #     # if j <= size(A, 1)
#     #     @inbounds A[j, i] = i
#     #     # end
#     # end
    
#     return nothing
# end
function mykernel3(A)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(A)
        @inbounds A[1, i] = i
        @inbounds A[2, i] = i
    end

    # i, j = (blockIdx().x - 1) * blockDim().x + threadIdx().x, (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # if i <= size(A, 1)
    #     A[i] = i

    # end
    # # if i <= size(A, 1) && j <= size(A, 2)
    # #     A[j] = j#1#3*i + 6*j
    # # end
    return nothing
end

function bench_gpu_kernel!(A)
    kernel = @cuda launch=false mykernel3(A)
    config = launch_configuration(kernel.fun)
    threads = min(length(A), config.threads)
    blocks = cld(length(A), threads)

    display(threads)
    display(blocks)

    CUDA.@sync begin
        kernel(A; threads, blocks)
    end
end


A = CUDA.rand(Float32, 5, 1000)
A[1,1]

bench_gpu_kernel!(A)
A
A[1]

A[2,1]
A[3,1]
A[10,1]
A[1,10]
A[3001]
A[1001]
A[2001]
A[1000]



# using CuArrays, CUDAnative

function add_offsets_kernel2(A)
    idx = linear_index(A)
    # for i in idx
    #     r, c = indices(A, i)
    #     A[i] += 5*r + 10*c
    # end
    return nothing
end

function bench_gpu_kernel!(A)
    kernel = @cuda launch=false add_offsets_kernel2(A)
    config = launch_configuration(kernel.fun)
    threads = min(length(A), config.threads)
    blocks = cld(length(A), threads)

    display(threads)
    display(blocks)

    CUDA.@sync begin
        kernel(A; threads, blocks)
    end
end

A = CUDA.rand(Float32, 5, 1000)
bench_gpu_kernel!(A)
A
