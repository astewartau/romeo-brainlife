#!/usr/bin/env julia

import Pkg
try
    using MriResearchTools, JSON
catch
    Pkg.add(Pkg.PackageSpec(name="JSON", version=v"0.21.4"))
    Pkg.add(Pkg.PackageSpec(name="MriResearchTools", version=v"3.2.0"))
    using MriResearchTools, JSON
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

    # Check for equal number of magnitude images, phase images, and TEs
    if length(magnitude_paths) != length(phase_paths) || length(magnitude_paths) != length(TEs)
        println("[ERROR] Number of magnitude images (" * string(length(magnitude_paths)) * 
                "), phase images (" * string(length(phase_paths)) * 
                "), and TEs (" * string(length(TEs)) * 
                ") do not match! Exiting...")
        exit(1)
    end

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

    # Calculate sum-of-squares for magnitude images along the 4th dimension
    println("[INFO] Calculating sum-of-squares for magnitude images...")
    magnitude_sos = sqrt.(sum(magnitude_combined .^ 2, dims=4))
    mkpath("fmap")
    savenii(magnitude_sos, "magnitude.nii.gz", "fmap", phase_header)

    # Phase offset removal
    println("[INFO] Removing phase offsets...")
    combined = mcpc3ds(phase_combined, magnitude_combined; TEs)
    phase_combined = combined.phase
    magnitude_combined = combined.mag

    # Unwrapping - t2starw with part-phase + unwrapped
    println("[INFO] Phase unwrapping...")
    mkpath("unwrapped")
    unwrapped = romeo(phase_combined; mag=magnitude_combined, TEs=TEs)
    savenii(unwrapped, "t2starw.nii.gz", "unwrapped", phase_header)

    # Quality map - new dtype? phasequality
    println("[INFO] Generating phase quality map...")
    mkpath("phase-quality")
    qmap = romeovoxelquality(phase_combined; magnitude_combined, TEs=TEs)
    savenii(qmap, "phase-quality.nii.gz", "phase-quality", phase_header)

    # Phase+mag mask - mask
    println("[INFO] Generating mask...")
    mkpath("mask")
    mask = robustmask(qmap)
    savenii(mask, "mask.nii.gz", "mask", phase_header)

    # T2* and R2* mapping - neuro/anat/qmri
    println("[INFO] Generating T2* and R2* maps...")
    mkpath("qmri")
    t2s = NumART2star(magnitude_combined, TEs)
    savenii(t2s, "T2star.nii.gz", "qmri", phase_header)
    r2s = r2s_from_t2s(t2s)
    savenii(r2s, "R2star.nii.gz", "qmri", phase_header)

    # B0 - neuro/fmap
    println("[INFO] Generating B0 field map...")
    mkpath("fmap")
    B0 = calculateB0_unwrapped(unwrapped, magnitude_combined, TEs)
    savenii(B0, "fieldmap.nii.gz", "fmap", phase_header)

    # Write fieldmap sidecar file with units
    println("[INFO] Writing fieldmap metadata...")
    open("fmap/fieldmap.json", "w") do file
        JSON.print(file, Dict("Units" => "Hz"))
    end

    # Copy or generate magnitude JSON sidecar
    magnitude_json_path = replace(magnitude_paths[1], r"\.nii(\.gz)?$" => ".json")
    if isfile(magnitude_json_path)
        println("[INFO] Copying magnitude JSON sidecar to fmap directory...")
        cp(magnitude_json_path, joinpath("fmap", "magnitude.json"), force=true)
    else
        # Attempt to find the first `meta` object in `_inputs` from config_data
        meta_data = nothing
        for entry in config_data["_inputs"]
            if entry["id"] == "magnitude" && "meta" in keys(entry)
                meta_data = entry["meta"]
                break
            end
        end

        # Check if meta data was found and write it to fmap/magnitude.json
        if meta_data != nothing
            open("fmap/magnitude.json", "w") do file
                JSON.print(file, meta_data)
            end
            println("[INFO] Written magnitude.json to fmap directory.")
        else
            @warn "No information found to generate magnitude.json - skipping..."
        end
    end
end

main()

