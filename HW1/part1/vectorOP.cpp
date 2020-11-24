#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x_value;
  __pp_vec_float x_value_sqr;
  __pp_vec_float x_result;
  __pp_vec_int x_exp;
  __pp_vec_float result;
  //__pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;
  
  __pp_vec_float threshold = _pp_vset_float(9.999999f);
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_vec_float one = _pp_vset_float(1.f);
  //__pp_vec_float two = _pp_vset_float(2.f);
  __pp_vec_int int_zero = _pp_vset_int(0);
  __pp_vec_int int_one = _pp_vset_int(1);
  __pp_vec_int int_two = _pp_vset_int(2);
  maskAll = _pp_init_ones();
  

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_int(x_exp, exponents + i, maskAll);
    maskIsNotNegative = _pp_mask_not(maskAll); // initialize maskIsNotNegative to all zero
    _pp_vgt_int(maskIsNotNegative, x_exp, int_zero, maskAll);
    maskIsNegative = _pp_mask_not(maskIsNotNegative); 
    
    _pp_vload_float(x_value, values + i, maskIsNotNegative); // x_value = value[i]
    _pp_vmove_float(result, x_value, maskIsNotNegative);
    // deal with those that has zero exponents
    _pp_vset_float(result, 1.f, maskIsNegative);

    _pp_vsub_int(x_exp, x_exp, int_one, maskIsNotNegative);
    
    // compute x_square
    _pp_vmult_float(x_value_sqr, x_value, x_value, maskAll);

    maskIsNotNegative = _pp_mask_not(maskAll);
    _pp_vgt_int(maskIsNotNegative, x_exp, int_one, maskAll);
    while(_pp_cntbits(maskIsNotNegative)) {
      
      _pp_vmult_float(result, result, x_value_sqr, maskIsNotNegative);
      _pp_vsub_int(x_exp, x_exp, int_two, maskIsNotNegative);

      // check threshold
      maskIsNegative = _pp_mask_not(maskAll);
      _pp_vgt_float(maskIsNegative, result, threshold, maskIsNotNegative);
      _pp_vset_float(result, 9.999999f, maskIsNegative);
      _pp_vset_int(x_exp, 0, maskIsNegative);
      
      _pp_vgt_int(maskIsNotNegative, x_exp, int_one, maskIsNotNegative);
    }

    // the remains exponents that are gt zero are all ones
    _pp_vgt_int(maskIsNotNegative, x_exp, int_zero, maskAll);
    _pp_vmult_float(result, result, x_value, maskIsNotNegative);
    
    maskIsNegative = _pp_mask_not(maskAll);
    _pp_vgt_float(maskIsNegative, result, threshold, maskIsNotNegative);
    _pp_vset_float(result, 9.999999f, maskIsNegative);
    _pp_vsub_int(x_exp, x_exp, int_one, maskIsNotNegative);

    // check N i value
    maskIsNotNegative = _pp_init_ones(N-i);
    _pp_vstore_float(output + i, result, maskIsNotNegative); // store result back to output
  }
}


// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float x_value, tmp_ans_v, tmp_x_value;
  __pp_mask maskAll, maskfirstvalue;
  maskAll = _pp_init_ones();
  maskfirstvalue = _pp_init_ones(1);
  tmp_ans_v = _pp_vset_float(0);

  float ans;

  int N_tmp;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(x_value, values+i, maskAll);
    N_tmp = VECTOR_WIDTH/2;
    while(N_tmp) {
      _pp_hadd_float(x_value, x_value);
      tmp_x_value = _pp_vset_float(0);
      _pp_interleave_float(tmp_x_value, x_value);
      _pp_vmove_float(x_value, tmp_x_value, maskAll);
      N_tmp = N_tmp/2;
    }
    _pp_vadd_float(tmp_ans_v, tmp_ans_v, x_value, maskfirstvalue);
  }
  _pp_vstore_float(&ans, tmp_ans_v, maskfirstvalue);

  return ans;
}