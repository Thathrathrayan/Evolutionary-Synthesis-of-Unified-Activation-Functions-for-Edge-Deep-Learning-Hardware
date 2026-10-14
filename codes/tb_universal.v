
`timescale 1ns / 1ps
module tb_universal;
    reg [15:0] X_IN;
    reg [1:0] MODE;
    wire [15:0] Y_OUT;
    
    integer f, i, m;

    unified_input_quantized_activation dut (
        .X_IN(X_IN),
        .MODE(MODE),
        .Y_OUT(Y_OUT)
    );

    initial begin
        f = $fopen("sim_results.csv", "w");
        $fwrite(f, "MODE,X_IN,Y_OUT\n");
        for (m = 0; m < 4; m = m + 1) begin
            MODE = m;
            for (i = 0; i < 65536; i = i + 64) begin
                X_IN = i;
                #1;
                $fwrite(f, "%0d,%0d,%0d\n", MODE, X_IN, Y_OUT);
            end
        end
        $fclose(f);
        $finish;
    end
endmodule
