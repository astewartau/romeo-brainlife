#!/usr/bin/env julia
import Pkg
try
    using MriResearchTools, ArgParse, JSON
catch
    Pkg.add(["MriResearchTools", "ArgParse", "JSON"])
    using MriResearchTools, ArgParse, JSON
end

# Load JSON data from file
function load_config(filename::String)
    open(filename, "r") do file
        return JSON.parse(file)
    end
end

# Extract paths and EchoTime values
function extract_info(json_data)
    magnitude_paths = json_data["magnitude"]
    phase_paths = json_data["phase"]

    TEs = Float64[]
    for entry in json_data["_inputs"]
        if entry["id"] == "magnitude"
            push!(TEs, entry["meta"]["EchoTime"] * 1000)
        end
    end

    return magnitude_paths, phase_paths, TEs
end

# Combine images into 4D array
function combine_images(loaded_images)
    num_images = length(loaded_images)
    combined_images = nothing

    if num_images > 0
        img_shape = size(loaded_images[1])
        combined_shape = tuple(img_shape..., num_images)

        # Initialize 4D array
        combined_images = Array{Float32}(undef, combined_shape...)

        # Fill 4D array with images
        for i in 1:num_images
            combined_images[:, :, :, i] = loaded_images[i]
        end
    end

    return combined_images
end

# Main function to handle script execution
function main()
    println("[INFO] Loading config.json...")
    config_data = load_config("config.json")

    println("[INFO] Extracting information...")
    magnitude_paths, phase_paths, TEs = extract_info(config_data)

    # Check all files exist before proceeding
    println("[INFO] Checking files exist...")
    all_files_exist = true
    for path in vcat(magnitude_paths, phase_paths)
        if !isfile(path)
            println("Error: File not found: $path")
            all_files_exist = false
        end
    end

    # Exit if any file is missing
    if !all_files_exist
        println("[ERROR] Missing files! Exiting...")
        exit(1)
    end

    # Load images using `readphase` and `readmag` functions
    println("[INFO] Reading NIfTI images...")
    magnitude_images = [Float64.(readmag(path)) for path in magnitude_paths]
    phase_images = [Float32.(readphase(path)) for path in phase_paths]
    
    # Load header from first phase image path
    phase_header = header(readphase(phase_paths[1]))

    # Combine images into 4D arrays
    println("[INFO] Combining NIfTI images...")
    magnitude_combined = combine_images(magnitude_images)
    phase_combined = combine_images(phase_images)

    # Create output folder
    println("[INFO] Creating output directory...")
    outputFolder = "outputFolder"
    mkpath(outputFolder)

    # Phase offset removal
    println("[INFO] Removing phase offsets...")
    combined = mcpc3ds(phase_combined, magnitude_combined; TEs)
    phase_combined = combined.phase
    magnitude_combined = combined.mag
    savenii(phase_combined, "phase_offsetremoved", outputFolder, phase_header)
    savenii(magnitude_combined, "mag_offsetremoved", outputFolder, phase_header)

    # Unwrapping
    println("[INFO] Phase unwrapping...")
    unwrapped = romeo(phase_combined; mag=magnitude_combined, TEs=TEs)
    savenii(unwrapped, "unwrapped", outputFolder, phase_header)

    # Quality map
    println("[INFO] Generating phase quality map...")
    qmap = romeovoxelquality(phase_combined; magnitude_combined, TEs=TEs)
    savenii(qmap, "quality_map", outputFolder, phase_header)

    # Phase+mag mask
    println("[INFO] Generating mask...")
    mask = robustmask(qmap)
    savenii(mask, "mask", outputFolder, phase_header)

    # T2* and R2* mapping
    println("[INFO] Generating T2* and R2* maps...")
    t2s = NumART2star(magnitude_combined, TEs)
    savenii(t2s, "t2s", outputFolder, phase_header)

    r2s = r2s_from_t2s(t2s)
    savenii(r2s, "r2s", outputFolder, phase_header)

    # B0
    println("[INFO] Generating B0 field map...")
    B0 = calculateB0_unwrapped(unwrapped, magnitude_combined, TEs)
    savenii(B0, "b0", outputFolder, phase_header)
end

# Run the main function
main()

