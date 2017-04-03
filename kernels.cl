__kernel void dualGradientEnergy(__global const int *in_r, __global const int *in_g, __global const int *in_b, __global int *res)
{
  int height = get_global_size(0);
  int width = get_global_size(1);
  int hid = get_global_id(0);
  int wid = get_global_id(1);

  int left = wid - 1;
  if (left < 0) left = width - 1;
  int right = wid + 1;
  if (right >= width) right = 0;

  int top = hid - 1;
  if (top < 0) top = height - 1;
  int bottom = hid + 1;
  if (bottom >= height) bottom = 0;

  int rx = in_r[hid*width+left] - in_r[hid*width+right];
  int gx = in_g[hid*width+left] - in_g[hid*width+right];
  int bx = in_b[hid*width+left] - in_b[hid*width+right];

  int ry = in_r[top*width+wid] - in_r[bottom*width+wid]; 
  int gy = in_g[top*width+wid] - in_g[bottom*width+wid];
  int by = in_b[top*width+wid] - in_b[bottom*width+wid];

  res[hid*width+wid] = ((rx*rx) + (gx*gx) + (bx*bx)) + ((ry*ry) + (gy*gy) + (by*by));

}