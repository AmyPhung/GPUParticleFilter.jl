module ParallelParticleFilters

export runfilterthreaded!
export runfiltergpu!

using CUDA
using Random
using Distributions
using StatsBase
using CUDA

function runfilterthreaded!(particles, inputs, measurements, 
    dynamics_model, measurement_model, output_particles)
    
    rng = MersenneTwister(1)
    
    num_steps = size(inputs)[1]
    num_particles = size(particles)[1]
  
    # Initialize weights vector
    weights = ones(num_particles)
    
    # Initialize space to copy particle data - needed for resampling step
    particle_buffer = similar(particles)
    copyto!(particle_buffer, particles)
    
    
    for idx in range(1, num_steps)
        u = inputs[idx,:]
        y = measurements[idx]
  
        # STEP 1: Compute weights
        Threads.@threads for p_idx = 1:num_particles
            p = particles[p_idx]
            weights[p_idx] = measurement_model(particle_buffer[p_idx], u, p, y)
        end
                
        # Compute CDF
        cdf = Weights(weights ./ sum(weights))

        # Copy particles to buffer 
        copyto!(particle_buffer, particles)
        copyto!(output_particles[idx], particles)
                
        # STEP 2: Resample particles
        samples = sample(1:num_particles, cdf, num_particles)

        # Extract the corresponding columns from the matrix
        particles = particle_buffer[samples]
        particle_buffer = particles

        # Step 3: Propogate particles
        Threads.@threads for p_idx = 1:num_particles
            p = particles[p_idx]
            particles[p_idx] = dynamics_model(p, u, y)
        end
    end
end

function runfiltergpu!(particles, inputs, measurements, 
        reweight_kernel!, propogate_kernel!, output_particles, pf_data)
    num_steps = length(inputs)
    num_particles = size(particles)[1]
    
    # Initialize weights vector
    weights_d = CUDA.fill(1.0f0, num_particles)
    
    # Initialize cumulative distribution vector
    cdf_d = similar(weights_d)
    
    # Initialize space to copy particle data - needed for resampling step
    particle_buffer = similar(particles)
    copyto!(particle_buffer, particles)
    
    # Initialize kernel configurations
    reweight_kernel = @cuda launch=false reweight_kernel!(particles, inputs[1], measurements[1], weights_d, pf_data)
    reweight_config = launch_configuration(reweight_kernel.fun)
    reweight_threads = min(num_particles, reweight_config.threads)
    reweight_blocks = cld(num_particles, reweight_threads)

#     display("Reweight Threads: $reweight_threads")
#     display("Reweight Blocks: $reweight_blocks")
    
    resample_kernel = @cuda launch=false resample_kernel!(particles, particle_buffer, cdf_d)
    resample_config = launch_configuration(resample_kernel.fun)
    resample_threads = min(num_particles, resample_config.threads)
    resample_blocks = cld(num_particles, resample_threads)

#     display("Resample Threads: $resample_threads")
#     display("Resample Blocks: $resample_blocks")
    
    propogate_kernel = @cuda launch=false propogate_kernel!(particles, inputs[1], pf_data)
    propogate_config = launch_configuration(propogate_kernel.fun)
    propogate_threads = min(num_particles, propogate_config.threads)
    propogate_blocks = cld(num_particles, propogate_threads)

#     display("Propogate Threads: $propogate_threads")
#     display("Propogate Blocks: $propogate_blocks")
    
    synchronize()
    
    for idx in range(1, num_steps)
        u = inputs[idx]
        y = measurements[idx]
        
        # STEP 1: Compute weights
        CUDA.@sync begin
            reweight_kernel(particles, u, y, weights_d, pf_data; threads=reweight_threads, blocks=reweight_blocks)
        end
        
        # Compute CDF
        cumsum!(cdf_d, weights_d)
        cdf_d ./= cdf_d[end]

        # Copy particles to buffer 
        copyto!(particle_buffer, particles)
        copyto!(output_particles[idx], particles)
        
        synchronize()
        
        # STEP 2: Resample particles
        CUDA.@sync begin
            resample_kernel(particles, particle_buffer, cdf_d; threads=resample_threads, blocks=resample_blocks)
        end
        
        # STEP 3: Propogate particles
        CUDA.@sync begin
            propogate_kernel(particles, u, pf_data; threads=propogate_threads, blocks=propogate_blocks)
        end
    end
end

function resample_kernel!(particles, particle_buffer, cdf)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    # Resample each particle
    # Approach - generate random number, find which CDF index
    # correlates with the random number. Values with higher likelihood
    # cover a wider range of values, and are consequently more likely
    # to be selected
    for i = index:stride:size(particles)[1]
        r = rand()
        
        # Use binary search to find corresponding index
        lo, hi = 1, size(particles)[1]
        while lo < hi
            mid = (lo + hi) ÷ 2
            if r < cdf[mid]
                hi = mid
            else
                lo = mid + 1
            end
        end
        idx = lo
        
        # Update particles
        particles[i, 1] = particle_buffer[idx, 1]
        particles[i, 2] = particle_buffer[idx, 2]
        particles[i, 3] = particle_buffer[idx, 3]
        particles[i, 4] = particle_buffer[idx, 4]
        particles[i, 5] = particle_buffer[idx, 5]
    end
    
    return nothing
end

end