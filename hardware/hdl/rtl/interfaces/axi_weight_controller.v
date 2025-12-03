//-----------------------------------------------------------------------------
// Title         : AXI Weight Controller
// Project       : PYNQ-Z2 SNN Accelerator
// File          : axi_weight_controller.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : AXI-Lite interface for runtime weight configuration
//                 Supports read/write access to weight memory
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module axi_weight_controller #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 16,    // 64KB address space
    parameter WEIGHT_WIDTH       = 16,
    parameter NUM_SYNAPSES       = 4096,  // 64x64 synaptic connections
    parameter ADDR_BITS          = 12     // log2(NUM_SYNAPSES)
)(
    // AXI4-Lite Slave Interface
    input  wire                             s_axi_aclk,
    input  wire                             s_axi_aresetn,
    // Write Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    s_axi_awaddr,
    input  wire [2:0]                       s_axi_awprot,
    input  wire                             s_axi_awvalid,
    output wire                             s_axi_awready,
    // Write Data Channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0]    s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                             s_axi_wvalid,
    output wire                             s_axi_wready,
    // Write Response Channel
    output wire [1:0]                       s_axi_bresp,
    output wire                             s_axi_bvalid,
    input  wire                             s_axi_bready,
    // Read Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]    s_axi_araddr,
    input  wire [2:0]                       s_axi_arprot,
    input  wire                             s_axi_arvalid,
    output wire                             s_axi_arready,
    // Read Data Channel
    output wire [C_S_AXI_DATA_WIDTH-1:0]    s_axi_rdata,
    output wire [1:0]                       s_axi_rresp,
    output wire                             s_axi_rvalid,
    input  wire                             s_axi_rready,
    
    // Weight Memory Interface (to synapse array)
    output wire                             weight_we,
    output wire [ADDR_BITS-1:0]             weight_addr,
    output wire [WEIGHT_WIDTH-1:0]          weight_wdata,
    input  wire [WEIGHT_WIDTH-1:0]          weight_rdata,
    output wire                             weight_rd_en,
    
    // Bulk transfer interface (for DMA-like operations)
    output wire                             bulk_start,
    output wire [ADDR_BITS-1:0]             bulk_start_addr,
    output wire [ADDR_BITS-1:0]             bulk_length,
    input  wire                             bulk_done,
    
    // Status outputs
    output wire [31:0]                      weight_status
);

    //-------------------------------------------------------------------------
    // Address Map (16-bit address space)
    // 0x0000-0x0FFF: Control registers
    // 0x1000-0x1FFF: Reserved
    // 0x2000-0x3FFF: Weight memory (direct access, 2 bytes per weight)
    // 0x4000-0x7FFF: Bulk transfer buffer
    //-------------------------------------------------------------------------
    localparam [15:0] ADDR_CTRL           = 16'h0000;  // Control register
    localparam [15:0] ADDR_STATUS         = 16'h0004;  // Status register
    localparam [15:0] ADDR_BULK_ADDR      = 16'h0008;  // Bulk transfer start address
    localparam [15:0] ADDR_BULK_LEN       = 16'h000C;  // Bulk transfer length
    localparam [15:0] ADDR_SINGLE_ADDR    = 16'h0010;  // Single weight address
    localparam [15:0] ADDR_SINGLE_DATA    = 16'h0014;  // Single weight data
    localparam [15:0] ADDR_WRITE_COUNT    = 16'h0018;  // Write operation counter
    localparam [15:0] ADDR_READ_COUNT     = 16'h001C;  // Read operation counter
    
    localparam [15:0] WEIGHT_MEM_BASE     = 16'h2000;
    localparam [15:0] WEIGHT_MEM_END      = 16'h3FFF;
    
    //-------------------------------------------------------------------------
    // Control register bits
    //-------------------------------------------------------------------------
    // [0]    : Enable weight updates
    // [1]    : Bulk transfer start
    // [2]    : Clear counters
    // [7:4]  : Reserved
    // [15:8] : Weight scaling factor
    // [31:16]: Reserved
    
    //-------------------------------------------------------------------------
    // Internal registers
    //-------------------------------------------------------------------------
    reg                             axi_awready;
    reg                             axi_wready;
    reg [1:0]                       axi_bresp;
    reg                             axi_bvalid;
    reg                             axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1:0]   axi_rdata;
    reg [1:0]                       axi_rresp;
    reg                             axi_rvalid;
    
    reg [C_S_AXI_ADDR_WIDTH-1:0]   axi_awaddr;
    reg [C_S_AXI_ADDR_WIDTH-1:0]   axi_araddr;
    
    // Control/status registers
    reg [31:0]                      ctrl_reg;
    reg [31:0]                      bulk_addr_reg;
    reg [31:0]                      bulk_len_reg;
    reg [31:0]                      single_addr_reg;
    reg [31:0]                      single_data_reg;
    reg [31:0]                      write_count;
    reg [31:0]                      read_count;
    
    // Weight memory access
    reg                             weight_we_reg;
    reg [ADDR_BITS-1:0]            weight_addr_reg;
    reg [WEIGHT_WIDTH-1:0]         weight_wdata_reg;
    reg                             weight_rd_en_reg;
    reg                             weight_rd_pending;
    
    // AXI handshake
    wire                            slv_reg_wren;
    wire                            slv_reg_rden;
    wire                            is_weight_mem_write;
    wire                            is_weight_mem_read;
    
    assign s_axi_awready = axi_awready;
    assign s_axi_wready  = axi_wready;
    assign s_axi_bresp   = axi_bresp;
    assign s_axi_bvalid  = axi_bvalid;
    assign s_axi_arready = axi_arready;
    assign s_axi_rdata   = axi_rdata;
    assign s_axi_rresp   = axi_rresp;
    assign s_axi_rvalid  = axi_rvalid;
    
    assign slv_reg_wren = axi_awready && s_axi_awvalid && axi_wready && s_axi_wvalid;
    assign slv_reg_rden = !axi_arready && s_axi_arvalid;
    
    // Check if access is to weight memory region
    assign is_weight_mem_write = (axi_awaddr >= WEIGHT_MEM_BASE) && (axi_awaddr <= WEIGHT_MEM_END);
    assign is_weight_mem_read  = (s_axi_araddr >= WEIGHT_MEM_BASE) && (s_axi_araddr <= WEIGHT_MEM_END);
    
    // Output assignments
    assign weight_we     = weight_we_reg;
    assign weight_addr   = weight_addr_reg;
    assign weight_wdata  = weight_wdata_reg;
    assign weight_rd_en  = weight_rd_en_reg;
    
    assign bulk_start      = ctrl_reg[1];
    assign bulk_start_addr = bulk_addr_reg[ADDR_BITS-1:0];
    assign bulk_length     = bulk_len_reg[ADDR_BITS-1:0];
    
    assign weight_status = {
        24'd0,
        bulk_done,          // [7]
        weight_rd_pending,  // [6]
        weight_we_reg,      // [5]
        ctrl_reg[2:0]       // [4:0] - first 3 bits of control
    };
    
    //-------------------------------------------------------------------------
    // Write Address Channel
    //-------------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_awready <= 1'b0;
            axi_awaddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (!axi_awready && s_axi_awvalid && s_axi_wvalid) begin
                axi_awready <= 1'b1;
                axi_awaddr  <= s_axi_awaddr;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end
    
    //-------------------------------------------------------------------------
    // Write Data Channel
    //-------------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_wready <= 1'b0;
        end else begin
            if (!axi_wready && s_axi_wvalid && s_axi_awvalid) begin
                axi_wready <= 1'b1;
            end else begin
                axi_wready <= 1'b0;
            end
        end
    end
    
    //-------------------------------------------------------------------------
    // Write Response Channel
    //-------------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_bvalid <= 1'b0;
            axi_bresp  <= 2'b00;
        end else begin
            if (slv_reg_wren && !axi_bvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b00;
            end else if (axi_bvalid && s_axi_bready) begin
                axi_bvalid <= 1'b0;
            end
        end
    end
    
    //-------------------------------------------------------------------------
    // Register Write Logic
    //-------------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            ctrl_reg        <= 32'd0;
            bulk_addr_reg   <= 32'd0;
            bulk_len_reg    <= 32'd0;
            single_addr_reg <= 32'd0;
            single_data_reg <= 32'd0;
            write_count     <= 32'd0;
            weight_we_reg   <= 1'b0;
            weight_addr_reg <= {ADDR_BITS{1'b0}};
            weight_wdata_reg <= {WEIGHT_WIDTH{1'b0}};
        end else begin
            // Default: deassert write enable after one cycle
            weight_we_reg <= 1'b0;
            
            // Clear bulk start after one cycle
            if (ctrl_reg[1]) begin
                ctrl_reg[1] <= 1'b0;
            end
            
            // Clear counter bit after processing
            if (ctrl_reg[2]) begin
                ctrl_reg[2] <= 1'b0;
                write_count <= 32'd0;
                read_count  <= 32'd0;
            end
            
            if (slv_reg_wren) begin
                if (is_weight_mem_write && ctrl_reg[0]) begin
                    // Direct weight memory write
                    weight_addr_reg  <= (axi_awaddr - WEIGHT_MEM_BASE) >> 1;  // Word address
                    weight_wdata_reg <= s_axi_wdata[WEIGHT_WIDTH-1:0];
                    weight_we_reg    <= 1'b1;
                    write_count      <= write_count + 1;
                end else begin
                    // Control register write
                    case (axi_awaddr[7:0])
                        ADDR_CTRL[7:0]: begin
                            ctrl_reg <= s_axi_wdata;
                        end
                        ADDR_BULK_ADDR[7:0]: begin
                            bulk_addr_reg <= s_axi_wdata;
                        end
                        ADDR_BULK_LEN[7:0]: begin
                            bulk_len_reg <= s_axi_wdata;
                        end
                        ADDR_SINGLE_ADDR[7:0]: begin
                            single_addr_reg <= s_axi_wdata;
                        end
                        ADDR_SINGLE_DATA[7:0]: begin
                            // Write to single weight via indirect access
                            single_data_reg <= s_axi_wdata;
                            if (ctrl_reg[0]) begin
                                weight_addr_reg  <= single_addr_reg[ADDR_BITS-1:0];
                                weight_wdata_reg <= s_axi_wdata[WEIGHT_WIDTH-1:0];
                                weight_we_reg    <= 1'b1;
                                write_count      <= write_count + 1;
                            end
                        end
                        default: begin
                            // Ignore writes to undefined addresses
                        end
                    endcase
                end
            end
        end
    end
    
    //-------------------------------------------------------------------------
    // Read Address Channel
    //-------------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_arready <= 1'b0;
            axi_araddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (!axi_arready && s_axi_arvalid) begin
                axi_arready <= 1'b1;
                axi_araddr  <= s_axi_araddr;
            end else begin
                axi_arready <= 1'b0;
            end
        end
    end
    
    //-------------------------------------------------------------------------
    // Read Data Channel
    //-------------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_rvalid <= 1'b0;
            axi_rresp  <= 2'b00;
            weight_rd_en_reg <= 1'b0;
            weight_rd_pending <= 1'b0;
        end else begin
            // Default
            weight_rd_en_reg <= 1'b0;
            
            if (slv_reg_rden) begin
                if (is_weight_mem_read) begin
                    // Initiate weight memory read
                    weight_addr_reg <= (s_axi_araddr - WEIGHT_MEM_BASE) >> 1;
                    weight_rd_en_reg <= 1'b1;
                    weight_rd_pending <= 1'b1;
                end
            end
            
            if (axi_arready && s_axi_arvalid && !axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b00;
                weight_rd_pending <= 1'b0;
            end else if (axi_rvalid && s_axi_rready) begin
                axi_rvalid <= 1'b0;
            end
        end
    end
    
    //-------------------------------------------------------------------------
    // Read Data Mux
    //-------------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}};
            read_count <= 32'd0;
        end else if (slv_reg_rden) begin
            if (is_weight_mem_read) begin
                // Weight memory read - data available next cycle
                axi_rdata <= {{(C_S_AXI_DATA_WIDTH-WEIGHT_WIDTH){1'b0}}, weight_rdata};
                read_count <= read_count + 1;
            end else begin
                // Control register read
                case (s_axi_araddr[7:0])
                    ADDR_CTRL[7:0]:        axi_rdata <= ctrl_reg;
                    ADDR_STATUS[7:0]:      axi_rdata <= weight_status;
                    ADDR_BULK_ADDR[7:0]:   axi_rdata <= bulk_addr_reg;
                    ADDR_BULK_LEN[7:0]:    axi_rdata <= bulk_len_reg;
                    ADDR_SINGLE_ADDR[7:0]: axi_rdata <= single_addr_reg;
                    ADDR_SINGLE_DATA[7:0]: axi_rdata <= {{(C_S_AXI_DATA_WIDTH-WEIGHT_WIDTH){1'b0}}, weight_rdata};
                    ADDR_WRITE_COUNT[7:0]: axi_rdata <= write_count;
                    ADDR_READ_COUNT[7:0]:  axi_rdata <= read_count;
                    default:               axi_rdata <= 32'hDEAD_BEEF;
                endcase
            end
        end
    end

endmodule
