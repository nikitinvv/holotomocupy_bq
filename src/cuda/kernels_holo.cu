#define PI 3.1415926535897932384626433

void __global__ pad_sym(float2* g, float2 *f, int pad_width, int n0, int n1, int ntheta, bool direction)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  int txx,tyy;
  if (tx >= n0+2*pad_width || ty >= n1+2*pad_width || tz >= ntheta)
    return;           
  if (ty < pad_width)
      tyy = pad_width-ty-1;
  else 
      if (ty >= n1 + pad_width)
        tyy = 2*n1-ty+pad_width-1;           
      else                
        tyy = ty-pad_width;
  if (tx < pad_width)
      txx = pad_width-tx-1;
  else 
  if (tx >= n0 + pad_width)
    txx = 2*n0-tx+pad_width-1;
  else                
    txx = tx-pad_width;
  int id1 = tz*(n0+2*pad_width)*(n1+2*pad_width)+ty*(n0+2*pad_width)+tx;
  int id2 = tz*n0*n1+tyy*n0+txx;
  if (direction == 0) 
  {
    g[id1].x = f[id2].x;
    g[id1].y = f[id2].y;
  } else {
    atomicAdd(&f[id2].x, g[id1].x);
    atomicAdd(&f[id2].y, g[id1].y);
  }
}

// Divide by phi
void __global__ divker2d(float2 *g, float2 *f, int n0, int n1, int ntheta, int m0,
                         int m1, float mu0, float mu1, bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= ntheta)
    return;
  float ker = __expf(-mu0 * (tx - n0 / 2) * (tx - n0 / 2) -
                     mu1 * (ty - n1 / 2) * (ty - n1 / 2));
  int f_ind = tx + ty * n0 + tz * n0 * n1;
  int g_ind = tx + n0 / 2 + m0 + (ty + n1 / 2 + m1) * (2 * n0 + 2 * m0) +
              tz * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1);
  
  if (direction == 0){
    g[g_ind].x = f[f_ind].x / ker / (4 * n0 * n1);
    g[g_ind].y = f[f_ind].y / ker / (4 * n0 * n1);
  } else {
    f[f_ind].x = g[g_ind].x / ker / (4 * n0 * n1);
    f[f_ind].y = g[g_ind].y / ker / (4 * n0 * n1);
  }
}

void __global__ fftshiftc2d(float2 *f, int n0, int n1, int ntheta) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= ntheta)
    return;
  int g = (1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2));
  f[tx + ty * n0 + tz * n0 * n1].x *= g;
  f[tx + ty * n0 + tz * n0 * n1].y *= g;
}

void __global__ wrap2d(float2 *f, int n0, int n1, int ntheta, int m0, int m1, bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * n0 + 2 * m0 || ty >= 2 * n1 + 2 * m1 || tz >= ntheta)
    return;
  if (tx < m0 || tx >= 2 * n0 + m0 || ty < m1 || ty >= 2 * n1 + m1) {
    int tx0 = (tx - m0 + 2 * n0) % (2 * n0);
    int ty0 = (ty - m1 + 2 * n1) % (2 * n1);
    int id1 = (+tx + ty * (2 * n0 + 2 * m0) +
               tz * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1));
    int id2 = (+tx0 + m0 + (ty0 + m1) * (2 * n0 + 2 * m0) +
               tz * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1));
    if (direction == 0) {
      f[id1].x = f[id2].x;
      f[id1].y = f[id2].y;
    } else {
      atomicAdd(&f[id2].x, f[id1].x);
      atomicAdd(&f[id2].y, f[id1].y);
    }
  }
}
void __global__ gather2d(float2 *g, float2 *f, float *x, float *y, int m0,
                         int m1, float mu0, float mu1, int n0e, int n1e, int ntheta, int n0, int n1,  bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n0 || ty >= n1 || tz >= ntheta)
    return;

  int g_ind = tx + ty * n0 + tz* n0 * n1;
  int x_ind = tx + ty * n0;

  float x0 = x[x_ind];
  float y0 = y[x_ind];

  float2 g0;
  if (direction == 0) {
    g0.x = 0.0f;
    g0.y = 0.0f;
  } else {
    g0.x = g[g_ind].x;
    g0.y = g[g_ind].y;
  }
  for (int i1 = 0; i1 < 2 * m1 + 1; i1++) {
    int ell1 = floorf(2 * n1e * y0) - m1 + i1;
    for (int i0 = 0; i0 < 2 * m0 + 1; i0++) {
      int ell0 = floorf(2 * n0e * x0) - m0 + i0;
      float w0 = ell0 / (float)(2 * n0e) - x0;
      float w1 = ell1 / (float)(2 * n1e) - y0;
      float w = PI / sqrtf(mu0 * mu1) *
                __expf(-PI * PI / mu0 * (w0 * w0) - PI * PI / mu1 * (w1 * w1));
      int f_ind = n0e + m0 + ell0 + (2 * n0e + 2 * m0) * (n1e + m1 + ell1) +
                  (2 * n0e + 2 * m0) * (2 * n1e + 2 * m1) * tz;
      if (direction == 0) {
        g0.x += w * f[f_ind].x;
        g0.y += w * f[f_ind].y;
      } else {
        float *fx = &(f[f_ind].x);
        float *fy = &(f[f_ind].y);
        atomicAdd(fx, w * g0.x);
        atomicAdd(fy, w * g0.y);
      }
    }
  }
  if (direction == 0){
    g[g_ind].x = g0.x;
    g[g_ind].y = g0.y;
  }
}
