`timescale 1ns/1ns

module Boost();
reg l, s, z;

initial begin
  l = 0;
  s = 0;
end

always @(posedge l) z = 0;
always @(posedge s) z = 1;

endmodule
