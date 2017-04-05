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

   int rx = in_r[hid*width + left] - in_r[hid*width + right];
   int gx = in_g[hid*width + left] - in_g[hid*width + right];
   int bx = in_b[hid*width + left] - in_b[hid*width + right];

   int ry = in_r[top*width + wid] - in_r[bottom*width + wid];
   int gy = in_g[top*width + wid] - in_g[bottom*width + wid];
   int by = in_b[top*width + wid] - in_b[bottom*width + wid];

   res[hid*width + wid] = ((rx*rx) + (gx*gx) + (bx*bx)) + ((ry*ry) + (gy*gy) + (by*by));

}

__kernel void cumulateEnergyAndPath(const int height, __global const int * in_energy, __global int * res_cumulated, __global int * res_paths)
{
   int width = get_global_size(1);
   int hid = get_global_id(0);
   int wid = get_global_id(1);
   
   if (height == 0) {
      res_cumulated[hid*width + wid] = in_energy[hid*width + wid];
      res_paths[hid*width + wid] = 0;
   } else {
      int above = height - 1;
      if (wid == 0) {
         if (res_cumulated[above*width + wid] <= res_cumulated[above*width + wid + 1]) {
            res_cumulated[hid*width + wid] = res_cumulated[above*width + wid] + in_energy[hid*width + wid];
            res_paths[hid*width + wid] = wid;
         } else {
            res_cumulated[hid*width + wid] = res_cumulated[above*width + wid + 1] + in_energy[hid*width + wid];
            res_paths[hid*width + wid] = wid + 1;
         }
      }
      else if (wid == width - 1) {
         if (res_cumulated[above*width + wid - 1] <= res_cumulated[above*width + wid]) {
            res_cumulated[hid*width + wid] = res_cumulated[above*width + wid - 1] + in_energy[hid*width + wid];
            res_paths[hid*width + wid] = wid - 1;
         }
         else {
            res_cumulated[hid*width + wid] = res_cumulated[above*width + wid] + in_energy[hid*width + wid];
            res_paths[hid*width + wid] = wid;
         }
      } else {
         if (res_cumulated[above*width + wid - 1] <= res_cumulated[above*width + wid] && res_cumulated[above*width + wid - 1] <= res_cumulated[above*width + wid + 1]) {
            res_cumulated[hid*width + wid] = res_cumulated[above*width + wid - 1] + in_energy[hid*width + wid];
            res_paths[hid*width + wid] = wid - 1;
         }
         else if (res_cumulated[above*width + wid] <= res_cumulated[above*width + wid - 1] && res_cumulated[above*width + wid] <= res_cumulated[above*width + wid + 1]) {
            res_cumulated[hid*width + wid] = res_cumulated[above*width + wid] + in_energy[hid*width + wid];
            res_paths[hid*width + wid] = wid;
         } else {
            res_cumulated[hid*width + wid] = res_cumulated[above*width + wid + 1] + in_energy[hid*width + wid];
            res_paths[hid*width + wid] = wid + 1;
         }
      }
   }
}

__kernel void removeVSeam(__global const int * in_img, __global int * in_seams, __global int * res)
{
   int width = get_global_size(1);
   int numPixels = get_global_size(2);
   int hid = get_global_id(0);
   int wid = get_global_id(1);
   int rgb_idx = get_global_id(2);

   if (wid >= width - 1) return;
   if (wid >= in_seams[hid]) {
      int target = (hid * (width-1) * numPixels) + (wid * numPixels) + rgb_idx; 
      int source = (hid * width * numPixels) + ((wid+1) * numPixels) + rgb_idx; 
      res[target] = in_img[source];
   } else {
      int target = (hid * (width-1) * numPixels) + (wid * numPixels) + rgb_idx; 
      int source = (hid * width * numPixels) + (wid * numPixels) + rgb_idx; 
      res[target] = in_img[source];
   }
}
