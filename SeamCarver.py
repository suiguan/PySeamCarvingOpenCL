from PIL import Image
import numpy as np
import sys
import time

#USE_OPENCL = False #serial computation
USE_OPENCL = True #GPU parallel computation

if USE_OPENCL: import pyopencl as cl

#helper function to find the mimimum value & index from the input list
def get_min(ar):
   m = ar[0]
   i = 0
   idx = 1
   for a in ar[1:]:
      if a < m:
         m = a
	 i = idx
      idx += 1
   return (m, i)

class SeamCarver:
   def __init__(self, img): #assume the image is RGB mode
      self.reset(img)
      if USE_OPENCL: self.initOpenCL()

   def initOpenCL(self):
      platforms = cl.get_platforms()
      gpus = platforms[0].get_devices(device_type=cl.device_type.GPU)
      self.max_work_group_size = gpus[0].max_work_group_size
      self.cl_ctx = cl.Context(devices=[gpus[0],])
      self.cl_queue = cl.CommandQueue(self.cl_ctx)
      print("using GPU from platform %s, device %s, max work group size %d" % (platforms[0], gpus[0], self.max_work_group_size))
      self.cl_prog = self.loadCLProgramFromFile('kernels.cl') 
      print("successfully build OpenCL kernels")

   def loadCLProgramFromFile(self, fname):
      f = open(fname, 'r')
      cl_kernels = f.read()
      f.close()
      return cl.Program(self.cl_ctx, cl_kernels).build()

   def reset(self, img):
      self.img = img
      self.width, self.height = self.img.size
      self.img_ar = np.array(self.img, np.uint32)
      self.r_img, self.g_img, self.b_img = self.img.split()

      #numpy array used by OpenCL:
      self.r_ar = np.array(self.r_img, dtype=np.uint32)
      self.g_ar = np.array(self.g_img, dtype=np.uint32)
      self.b_ar = np.array(self.b_img, dtype=np.uint32)

      #image array used by regular access:
      self.r = self.r_img.load()
      self.g = self.g_img.load()
      self.b = self.b_img.load()

   #return the dual gradient energy at given pixel
   def getDualGradientEnergy(self, x, y):
      #first calcuate x-energy from r, g, b different between (x-1,y) and (x+1,y)
      x1 = x - 1 if x > 0 else self.width - 1
      x2 = x + 1 if x < self.width - 1 else 0
      rx = abs(self.r[x1, y] - self.r[x2, y])
      gx = abs(self.g[x1, y] - self.g[x2, y])
      bx = abs(self.b[x1, y] - self.b[x2, y])
      x_energy = (rx**2) + (gx**2) + (bx**2)

      #then calcuate y-energy from r, g, b different between (x,y-1) and (x,y+1)
      y1 = y - 1 if y > 0 else self.height - 1
      y2 = y + 1 if y < self.height - 1 else 0
      ry = abs(self.r[x, y1] - self.r[x, y2])
      gy = abs(self.g[x, y1] - self.g[x, y2])
      by = abs(self.b[x, y1] - self.b[x, y2])
      y_energy = (ry**2) + (gy**2) + (by**2)
      
      #return the total energy
      return x_energy + y_energy

   #compute the energy map using CPU
   def getEnergyMap(self):
      energy = np.zeros((self.height, self.width), dtype=np.uint32) 
      for y in range(0, self.height):
         for x in range(0, self.width):
            energy[y][x] = self.getDualGradientEnergy(x,y)
      return energy

   #compute the energy map using OpenCL 
   def getEnergyMapWithCL(self):
      energy = np.zeros((self.height, self.width), dtype=np.uint32) 
      mf = cl.mem_flags
      in_r = cl.Buffer(self.cl_ctx, mf.READ_ONLY|mf.USE_HOST_PTR, hostbuf=self.r_ar)
      in_g = cl.Buffer(self.cl_ctx, mf.READ_ONLY|mf.USE_HOST_PTR, hostbuf=self.g_ar)
      in_b = cl.Buffer(self.cl_ctx, mf.READ_ONLY|mf.USE_HOST_PTR, hostbuf=self.b_ar)
      res = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, energy.nbytes)
      self.cl_prog.dualGradientEnergy(self.cl_queue, energy.shape, None, in_r, in_g, in_b, res)
      cl.enqueue_copy(self.cl_queue, energy, res)
      return energy

   #return a length of height array of the pixel x-index of the vertical seam
   def findVerticalSeam(self):
      before = time.time()
      energy_map_ar = self.getEnergyMap()
      after = time.time()
      #print("findVerticalSeam: Take %.6f seconds to build energy map" % (after-before,))

      cumulated_energy = np.zeros(energy_map_ar.shape, dtype=np.uint32)
      paths = np.zeros(energy_map_ar.shape, dtype=np.uint32)
      for y in range(0, self.height):
         for x in range(0, self.width):
            if y == 0:
               cumulated_energy[y][x] = energy_map_ar[y][x]
               paths[y][x] = 0 #unused
            else:
               if x == 0:
		  (m, i) = get_min(cumulated_energy[y-1][x:x+2])
                  cumulated_energy[y][x] = energy_map_ar[y][x] + m 
                  paths[y][x] = x + i 
               elif x == self.width - 1:
		  (m, i) = get_min(cumulated_energy[y-1][x-1:x+1])
                  cumulated_energy[y][x] = energy_map_ar[y][x] + m 
                  paths[y][x] = x - 1 + i 
               else:
		  (m, i) = get_min(cumulated_energy[y-1][x-1:x+2])
                  cumulated_energy[y][x] = energy_map_ar[y][x] + m 
                  paths[y][x] = x - 1 + i 
      (m, bottom_x) = get_min(cumulated_energy[self.height-1])
      return self.findVerticalSeamFromBottomX(paths, bottom_x)

   #find the vertical seam using OpenCL
   def findVerticalSeamWithOpenCL(self):
      before = time.time()
      energy_map_ar = self.getEnergyMapWithCL()
      after = time.time()
      #print("findVerticalSeamWithOpenCL: Take %.6f seconds to build energy map" % (after-before,))

      cumulated_energy = np.zeros(energy_map_ar.shape, dtype=np.uint32)
      paths = np.zeros(energy_map_ar.shape, dtype=np.uint32)

      mf = cl.mem_flags
      in_energy = cl.Buffer(self.cl_ctx, mf.READ_ONLY|mf.USE_HOST_PTR, hostbuf=energy_map_ar)
      res_cumulated = cl.Buffer(self.cl_ctx, mf.READ_WRITE, energy_map_ar.nbytes)
      res_paths = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, energy_map_ar.nbytes)
      for h in range(0, self.height):
         self.cl_prog.cumulateEnergyAndPath(self.cl_queue, (1, self.width), None, np.uint32(h), in_energy, res_cumulated, res_paths, global_offset=(h,0))
      cl.enqueue_copy(self.cl_queue, cumulated_energy, res_cumulated)
      cl.enqueue_copy(self.cl_queue, paths, res_paths)
      (m, bottom_x) = get_min(cumulated_energy[self.height-1])
      return self.findVerticalSeamFromBottomX(paths, bottom_x)

   #from the paths array, return the vertical seam based on the given bottom x
   def findVerticalSeamFromBottomX(self, paths, x):
      seam = [x,]
      seam_x = x
      for y in reversed(range(1, self.height)):
         seam_x = paths[y][seam_x]
         seam.append(seam_x);
      seam.reverse()
      return seam

   #remove the given vertical seam index
   def removeVerticalSeam(self, vseam):
      if len(vseam) != self.height:
         print("invalid seam length %d, height %d" % (len(vseam), self.height))
         return
      new_img_ar = np.array([np.delete(self.img_ar[y], vseam[y], axis=0) for y in range(0, self.height)])
      self.reset(Image.fromarray(new_img_ar.astype(dtype=np.uint8, copy=False)))

   #remove the given vertical seam index using OpenCL
   def removeVerticalSeamWithOpenCL(self, vseam):
      (h, w, n) = self.img_ar.shape
      new_img_ar = np.zeros((h, w-1, n), dtype=np.uint32)
      mf = cl.mem_flags
      in_img = cl.Buffer(self.cl_ctx, mf.READ_ONLY|mf.USE_HOST_PTR, hostbuf=self.img_ar)
      in_seams = cl.Buffer(self.cl_ctx, mf.READ_ONLY|mf.USE_HOST_PTR, hostbuf=np.array(vseam, np.uint32))
      res = cl.Buffer(self.cl_ctx, mf.WRITE_ONLY, new_img_ar.nbytes)
      self.cl_prog.removeVSeam(self.cl_queue, self.img_ar.shape, None, in_img, in_seams, res)
      cl.enqueue_copy(self.cl_queue, new_img_ar, res)
      self.reset(Image.fromarray(new_img_ar.astype(dtype=np.uint8, copy=False)))

   #save the img to a JPEG file
   def dumpImg(self, name):
      self.img.save(name, "JPEG")




#---------------- Main ---------------
def usage(prog):
   print("Usage: python %s <jpeg>" % prog)
   sys.exit(-1);

def main(argv):
   if len(argv) != 2: usage(argv[0])
   img = Image.open(argv[1])
   w,h = img.size
   new_w = w/2

   #regular resizing
   img.resize((new_w, h), Image.BILINEAR).save("bilinear_resize_%dx%d.jpg" % (new_w, h), "JPEG") 

   #seam carving resizing
   carver = SeamCarver(img)
   columns_to_remove = w - new_w

   begin = time.time()
   for i in range(0, columns_to_remove): #for each width that needs to be removed 
      #1) Find the seam
      #print("iteration = %d, new_w = %d" % (i, new_w))
      before = time.time()
      if USE_OPENCL: vseam = carver.findVerticalSeamWithOpenCL()
      else: vseam = carver.findVerticalSeam()
      afterFindSeam = time.time()
      #print("Take %.6f seconds to find seam" % (afterFindSeam - before))

      #2) Remove the seam
      if USE_OPENCL: carver.removeVerticalSeamWithOpenCL(vseam)
      else: carver.removeVerticalSeam(vseam)
      afterRemoveSeam = time.time()
      #print("Take %.6f seconds to remove seam" % (afterRemoveSeam - afterFindSeam))

   end = time.time()
   device = "GPU" if USE_OPENCL else "CPU"
   print("Using %s : Take %.6f seconds to complete." % (device, end - begin))

   #save the final result
   carver.dumpImg("out_%s.jpg" % device)

if __name__ == "__main__":
   main(sys.argv)
