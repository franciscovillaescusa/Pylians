cdef extern from "MAS_c.h":
       ctypedef float FLOAT
       void CIC3D(FLOAT *pos, FLOAT *number, long particles, int dims,
                  FLOAT BoxSize, int threads)
       void CIC2D(FLOAT *pos, FLOAT *number, long particles, int dims,
                  FLOAT BoxSize, int threads)
       void CICW3D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
                   FLOAT BoxSize, int threads)
       void CICW2D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
                   FLOAT BoxSize, int threads)
       void NGP3D(FLOAT *pos, FLOAT *number, long particles, int dims,
                  FLOAT BoxSize, int threads)
       void NGP2D(FLOAT *pos, FLOAT *number, long particles, int dims,
                  FLOAT BoxSize, int threads)
       void NGPW3D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
                   FLOAT BoxSize, int threads)
       void NGPW2D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
                   FLOAT BoxSize, int threads)
       void TSC3D(FLOAT *pos, FLOAT *number, long particles, int dims,
                  FLOAT BoxSize, int threads)
       void TSC2D(FLOAT *pos, FLOAT *number, long particles, int dims,
                  FLOAT BoxSize, int threads)
       void TSCW3D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
                   FLOAT BoxSize, int threads)
       void TSCW2D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
                   FLOAT BoxSize, int threads)
       
