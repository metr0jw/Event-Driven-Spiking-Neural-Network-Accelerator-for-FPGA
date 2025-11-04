##-----------------------------------------------------------------------------
## Title         : HLS IP Export Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : export_ip.tcl
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Exports HLS designs as IP cores for Vivado
##-----------------------------------------------------------------------------

# Set IP repository path
set IP_REPO_PATH "[pwd]/../../ip/custom_ip"

# Create IP repository directory if it doesn't exist
file mkdir $IP_REPO_PATH

# Function to export IP
proc export_ip_core {proj_name ip_name version vendor} {
    global IP_REPO_PATH
    
    puts "\n=========================================="
    puts "Exporting IP: $ip_name"
    puts "=========================================="
    
    # Open project
    open_project $proj_name
    open_solution "solution1"
    
    # Configure export settings
    config_export -description "$ip_name - HLS generated IP core" \
                  -display_name $ip_name \
                  -format ip_catalog \
                  -library "hls" \
                  -rtl verilog \
                  -vendor $vendor \
                  -version $version \
                  -vivado_phys_opt place \
                  -vivado_report_level 0
    
    # Set IP repository path
    config_export -output $IP_REPO_PATH/$ip_name
    
    # Export design
    puts "Exporting to: $IP_REPO_PATH/$ip_name"
    export_design -rtl verilog -format ip_catalog
    
    close_project
    
    puts "IP core $ip_name exported successfully"
}

# Export all IP cores
# Format: project_name, ip_name, version, vendor

export_ip_core \
    "snn_learning_engine_prj" \
    "snn_learning_engine" \
    "1.0" \
    "PYNQ-Z2-SNN"

export_ip_core \
    "spike_encoder_prj" \
    "spike_encoder" \
    "1.0" \
    "PYNQ-Z2-SNN"

export_ip_core \
    "weight_updater_prj" \
    "weight_updater" \
    "1.0" \
    "PYNQ-Z2-SNN"

export_ip_core \
    "spike_decoder_prj" \
    "spike_decoder" \
    "1.0" \
    "PYNQ-Z2-SNN"

export_ip_core \
    "network_controller_prj" \
    "network_controller" \
    "1.0" \
    "PYNQ-Z2-SNN"

puts "\n=========================================="
puts "All IP cores exported to: $IP_REPO_PATH"
puts "=========================================="
puts ""
puts "To use these IP cores in Vivado:"
puts "1. Open your Vivado project"
puts "2. Go to Settings -> Repository"
puts "3. Add path: $IP_REPO_PATH"
puts "4. The IP cores will appear in the IP Catalog under 'User Repository'"
