module test(input wire clk);
    logic [31:0] ra = 0;
    logic [31:0] rb;
    logic [31:0] rc = 0;
    always_ff @(posedge clk) begin
        ra <= ra + 1 + rc;
        rb <= ra ^ {ra[16:0],15'h0};
        rc <= (rc ^ (rb >> 4));
        if (rc == 32'h1) begin
            $finish;
        end
    end
endmodule