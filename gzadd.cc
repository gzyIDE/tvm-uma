#ifdef __cplusplus
extern "C"
#endif

int vanilla_accelerator_addvec(float *in1, float *in2, float *result, int w) {
  for ( int i = 0; i < w; i++ ) {
    result[i] = in1[i] + in2[i];
  }

  return 0;
}

int vanilla_accelerator_addvec_bc(float *in1, float *in2, float *result, int w, int wb) {
  for ( int i = 0; i < w/wb; i++ ) {
    for ( int j = 0; j < wb; j++ ) {
      int idx = i * wb + j;
      result[idx] = in1[idx] + in2[i];
    }
  }

  return 0;
}
