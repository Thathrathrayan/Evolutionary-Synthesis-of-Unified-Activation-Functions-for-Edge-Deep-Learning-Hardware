`timescale 1ns / 1ps

// ============================================================================
// Full RI5CY unified-activation sweep test
//
// Tests the complete Option-A processor path for every sampled input:
//   custom instruction -> LW packed input -> accelerator -> SW substituted
//   result -> LW result back -> CSV
//
// Default sweep:
//   MODE = 0,1,2,3
//   X_IN raw = 0..65535 in increments of STEP (default STEP=64)
//
// Set STEP=1 for all 65,536 raw 16-bit values per mode.
// ============================================================================
module top_tb;

    parameter integer STEP = 64;
    parameter integer MAX_JOB_CYCLES = 200;

    reg         clk_i;
    reg         rstn_i;
    reg  [31:0] irq_i;

    reg         debug_req_i;
    wire        debug_gnt_o;
    wire        debug_rvalid_o;
    reg  [14:0] debug_addr_i;
    reg         debug_we_i;
    reg  [31:0] debug_wdata_i;
    wire [31:0] debug_rdata_o;
    wire        debug_halted_o;

    reg         fetch_enable_i;
    wire        core_busy_o;
    wire        activation_done;
    wire signed [15:0] activation_result;

    top #(
        .INSTR_RDATA_WIDTH (128),
        .ADDR_WIDTH        (14),
        .BOOT_ADDR         ('h80)
    ) uut (
        .clk_i              (clk_i),
        .rstn_i             (rstn_i),
        .irq_i              (irq_i),
        .debug_req_i        (debug_req_i),
        .debug_gnt_o        (debug_gnt_o),
        .debug_rvalid_o     (debug_rvalid_o),
        .debug_addr_i       (debug_addr_i),
        .debug_we_i         (debug_we_i),
        .debug_wdata_i      (debug_wdata_i),
        .debug_rdata_o      (debug_rdata_o),
        .debug_halted_o     (debug_halted_o),
        .load_i             (1'b0),
        .fetch_enable_i     (fetch_enable_i),
        .core_busy_o        (core_busy_o),
        .activation_done    (activation_done),
        .activation_result  (activation_result)
    );

    always #5 clk_i = ~clk_i;

    localparam integer INPUT_ADDR  = 32'h00000C00;
    localparam integer OUTPUT_ADDR = 32'h00000D00;
    localparam integer PROGRAM_BASE = 32'h00000080;

    localparam [31:0] I_LUI_X10_1       = 32'h00001537; // x10 = 0x1000
    localparam [31:0] I_ADDI_X10_NEG1K  = 32'hC0050513; // x10 = 0x0C00
    localparam [31:0] I_LUI_X12_1       = 32'h00001637; // x12 = 0x1000
    localparam [31:0] I_ADDI_X12_NEG768 = 32'hD0060613; // x12 = 0x0D00
    localparam [31:0] I_CUSTOM           = 32'h007302EB; // opcode[6:0] = 7'h6B
    localparam [31:0] I_LW_X11_0_X10    = 32'h00052583; // lw x11, 0(x10)
    localparam [31:0] I_SW_X0_0_X12     = 32'h00062023; // sw x0, 0(x12)
    localparam [31:0] I_LW_X13_0_X12    = 32'h00062683; // lw x13, 0(x12)
    localparam [31:0] I_NOP              = 32'h00000013;
    localparam [31:0] I_JSELF            = 32'h0000006F;

    integer csv_file;
    integer raw_x;
    integer mode_i;
    integer jobs;
    integer errors;
    integer cycles;
    integer pc;
    integer pad;

    reg [31:0] packed_input;
    reg [31:0] loaded_word;
    reg [31:0] ram_word;
    reg [15:0] y_raw;
    reg signed [15:0] x_signed;
    reg signed [15:0] y_signed;
    reg output_read_seen;
    reg job_done;

    task automatic write_word(input integer address, input [31:0] data);
        begin
            uut.ram_i.dp_ram_i.mem[address + 0] = data[7:0];
            uut.ram_i.dp_ram_i.mem[address + 1] = data[15:8];
            uut.ram_i.dp_ram_i.mem[address + 2] = data[23:16];
            uut.ram_i.dp_ram_i.mem[address + 3] = data[31:24];
        end
    endtask

    function automatic [31:0] read_word(input integer address);
        begin
            read_word = {
                uut.ram_i.dp_ram_i.mem[address + 3],
                uut.ram_i.dp_ram_i.mem[address + 2],
                uut.ram_i.dp_ram_i.mem[address + 1],
                uut.ram_i.dp_ram_i.mem[address + 0]
            };
        end
    endfunction

    task automatic load_fixed_program;
        begin
            pc = PROGRAM_BASE;

            write_word(pc, I_LUI_X10_1);       pc = pc + 4;
            write_word(pc, I_ADDI_X10_NEG1K);  pc = pc + 4;
            write_word(pc, I_LUI_X12_1);       pc = pc + 4;
            write_word(pc, I_ADDI_X12_NEG768); pc = pc + 4;
            write_word(pc, I_CUSTOM);           pc = pc + 4;
            write_word(pc, I_LW_X11_0_X10);     pc = pc + 4;

            // Margin for wrapper completion and pending-flag registration.
            for (pad = 0; pad < 4; pad = pad + 1) begin
                write_word(pc, I_NOP);
                pc = pc + 4;
            end

            write_word(pc, I_SW_X0_0_X12);  pc = pc + 4;
            write_word(pc, I_NOP);           pc = pc + 4;
            write_word(pc, I_LW_X13_0_X12); pc = pc + 4;
            write_word(pc, I_JSELF);
        end
    endtask

    task automatic run_one_job(
        input [15:0] x_bits,
        input [1:0] mode_bits
    );
        begin
            fetch_enable_i = 1'b0;
            rstn_i = 1'b0;

            // Hold reset for multiple clock edges.
            repeat (3) @(posedge clk_i);

            packed_input = {14'b0, mode_bits, x_bits};
            write_word(INPUT_ADDR, packed_input);
            write_word(OUTPUT_ADDR, 32'hA5A5A5A5);
            load_fixed_program();

            output_read_seen = 1'b0;
            job_done = 1'b0;
            loaded_word = 32'h00000000;

            @(negedge clk_i);
            rstn_i = 1'b1;
            fetch_enable_i = 1'b1;

            // Wait for the processor's LW from OUTPUT_ADDR and its response.
            for (cycles = 0;
                 (cycles < MAX_JOB_CYCLES) && !job_done;
                 cycles = cycles + 1) begin
                @(posedge clk_i);

                if (uut.data_req && uut.data_gnt && !uut.data_we &&
                    (uut.data_addr == OUTPUT_ADDR)) begin
                    output_read_seen = 1'b1;
                end

                if (output_read_seen && uut.data_rvalid) begin
                    // Sample after NBA updates so data_rdata is the returned word.
                    @(negedge clk_i);
                    loaded_word = uut.data_rdata;
                    job_done = 1'b1;
                end
            end

            fetch_enable_i = 1'b0;

            if (!job_done) begin
                $display("ERROR: timeout MODE=%0d X_RAW=%0d", mode_bits, x_bits);
                errors = errors + 1;
            end else begin
                ram_word = read_word(OUTPUT_ADDR);
                y_raw = loaded_word[15:0];
                x_signed = $signed(x_bits);
                y_signed = $signed(y_raw);

                // Validate processor load-back against actual stored RAM word.
                if (loaded_word !== ram_word) begin
                    $display(
                        "ERROR: load-back mismatch MODE=%0d X_RAW=%0d loaded=0x%08h RAM=0x%08h",
                        mode_bits, x_bits, loaded_word, ram_word
                    );
                    errors = errors + 1;
                end

                // Validate that RAM contains a sign-extended 16-bit result.
                if (ram_word !== {{16{ram_word[15]}}, ram_word[15:0]}) begin
                    $display(
                        "ERROR: sign-extension mismatch MODE=%0d X_RAW=%0d RAM=0x%08h",
                        mode_bits, x_bits, ram_word
                    );
                    errors = errors + 1;
                end

                // Raw columns preserve exact 16-bit Verilog encodings.
                // Signed columns make the CSV easier to inspect directly.
                $fwrite(
                    csv_file,
                    "%0d,%0d,%0d,%0d,%0d,0x%08h\n",
                    mode_bits,
                    x_bits,
                    y_raw,
                    x_signed,
                    y_signed,
                    ram_word
                );
                jobs = jobs + 1;
            end
        end
    endtask

    initial begin
        clk_i = 1'b0;
        rstn_i = 1'b0;
        irq_i = 32'd0;
        debug_req_i = 1'b0;
        debug_addr_i = 15'd0;
        debug_we_i = 1'b0;
        debug_wdata_i = 32'd0;
        fetch_enable_i = 1'b0;

        jobs = 0;
        errors = 0;

        if (STEP <= 0 || STEP > 65536) begin
            $fatal(1, "STEP must be in the range 1..65536");
        end

        csv_file = $fopen("ri5cy_unified_activation_results.csv", "w");
        if (csv_file == 0) begin
            $fatal(1, "Could not open ri5cy_unified_activation_results.csv");
        end

        $fwrite(
            csv_file,
            "MODE,X_RAW,Y_RAW,X_SIGNED,Y_SIGNED,RAM_WORD\n"
        );

        $display("=====================================================");
        $display("RI5CY unified activation full-path sweep");
        $display("STEP=%0d, expected jobs=%0d", STEP, 4 * ((65535 / STEP) + 1));
        $display("=====================================================");

        for (mode_i = 0; mode_i < 4; mode_i = mode_i + 1) begin
            $display("Starting MODE=%0d", mode_i);

            for (raw_x = 0; raw_x < 65536; raw_x = raw_x + STEP) begin
                run_one_job(raw_x[15:0], mode_i[1:0]);
            end

            $display("Completed MODE=%0d, cumulative jobs=%0d", mode_i, jobs);
        end

        $fclose(csv_file);

        $display("=====================================================");
        $display("Sweep complete: jobs=%0d errors=%0d", jobs, errors);
        $display("CSV: ri5cy_unified_activation_results.csv");
        if (errors == 0)
            $display("FULL-PATH CSV SWEEP PASS");
        else
            $display("FULL-PATH CSV SWEEP FAIL");
        $display("=====================================================");

        #20;
        $finish;
    end

endmodule
