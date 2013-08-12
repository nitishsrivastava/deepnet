# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>

using namespace std;

# include "ziggurat.h"

//****************************************************************************80

float r4_exp ( unsigned long int *jsr, int ke[256], float fe[256], 
  float we[256] )

//****************************************************************************80
//
//  Purpose:
//
//    R4_EXP returns an exponentially distributed single precision real value.
//
//  Discussion:
//
//    The underlying algorithm is the ziggurat method.
//
//    Before the first call to this function, the user must call R4_EXP_SETUP
//    to determine the values of KE, FE and WE.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 December 20080
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    George Marsaglia, Wai Wan Tsang,
//    The Ziggurat Method for Generating Random Variables,
//    Journal of Statistical Software,
//    Volume 5, Number 8, October 2000, seven pages.
//
//  Parameters:
//
//    Input/output, unsigned long int *JSR, the seed.
//
//    Input, int KE[256], data computed by R4_EXP_SETUP.
//
//    Input, float FE[256], WE[256], data computed by R4_EXP_SETUP.
//
//    Output, float R4_EXP, an exponentially distributed random value.
//
{
  int iz;
  int jz;
  float value;
  float x;

  jz = shr3 ( jsr );
  iz = ( jz & 255 );

  if ( abs ( jz  ) < ke[iz] )
  {
    value = ( float ) ( abs ( jz ) ) * we[iz];
  }
  else
  {
    for ( ; ; )
    {
      if ( iz == 0 )
      {
        value = 7.69711 - log ( r4_uni ( jsr ) );
        break;
      }

      x = ( float ) ( abs ( jz ) ) * we[iz];

      if ( fe[iz] + r4_uni ( jsr ) * ( fe[iz-1] - fe[iz] ) < exp ( - x ) )
      {
        value = x;
        break;
      }

      jz = shr3 ( jsr );
      iz = ( jz & 255 );

      if ( abs ( jz ) < ke[iz] )
      {
        value = ( float ) ( abs ( jz ) ) * we[iz];
        break;
      }
    }
  }
  return value;
}
//****************************************************************************80

void r4_exp_setup ( int ke[256], float fe[256], float we[256] )

//****************************************************************************80
//
//  Purpose:
//
//    R4_EXP_SETUP sets data needed by R4_EXP.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 December 2008
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    George Marsaglia, Wai Wan Tsang,
//    The Ziggurat Method for Generating Random Variables,
//    Journal of Statistical Software,
//    Volume 5, Number 8, October 2000, seven pages.
//
//  Parameters:
//
//    Output, int KE[256], data needed by R4_EXP.
//
//    Output, float FE[256], WE[256], data needed by R4_EXP.
//
{
  double de = 7.697117470131487;
  int i;
  const double m2 = 2147483648.0;
  double q;
  double te = 7.697117470131487;
  const double ve = 3.949659822581572E-03;

  q = ve / exp ( - de );

  ke[0] = ( int ) ( ( de / q ) * m2 );
  ke[1] = 0;

  we[0] = ( float ) ( q / m2 );
  we[255] = ( float ) ( de / m2 );

  fe[0] = 1.0;
  fe[255] = ( float ) ( exp ( - de ) );

  for ( i = 254; 1 <= i; i-- )
  {
    de = - log ( ve / de + exp ( - de ) );
    ke[i+1] = ( int ) ( ( de / te ) * m2 );
    te = de;
    fe[i] = ( float ) ( exp ( - de ) );
    we[i] = ( float ) ( de / m2 );
  }
  return;
}
//****************************************************************************80

float r4_nor ( unsigned long int *jsr, int kn[128], float fn[128], 
  float wn[128] )

//****************************************************************************80
//
//  Purpose:
//
//    R4_NOR returns a normally distributed single precision real value.
//
//  Discussion:
//
//    The value returned is generated from a distribution with mean 0 and 
//    variance 1.
//
//    The underlying algorithm is the ziggurat method.
//
//    Before the first call to this function, the user must call R4_NOR_SETUP
//    to determine the values of KN, FN and WN.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 December 2008
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    George Marsaglia, Wai Wan Tsang,
//    The Ziggurat Method for Generating Random Variables,
//    Journal of Statistical Software,
//    Volume 5, Number 8, October 2000, seven pages.
//
//  Parameters:
//
//    Input/output, unsigned long int *JSR, the seed.
//
//    Input, int KN[128], data computed by R4_NOR_SETUP.
//
//    Input, float FN[128], WN[128], data computed by R4_NOR_SETUP.
//
//    Output, float R4_NOR, a normally distributed random value.
//
{
  int hz;
  int iz;
  const float r = 3.442620;
  float value;
  float x;
  float y;

  hz = shr3 ( jsr );
  iz = ( hz & 127 );

  if ( abs ( hz ) < kn[iz] )
  {
    value = ( float ) ( hz ) * wn[iz];
  }
  else
  {
    for ( ; ; )
    {
      if ( iz == 0 )
      {
        for ( ; ; )
        {
          x = - 0.2904764 * log ( r4_uni ( jsr ) );
          y = - log ( r4_uni ( jsr ) );
          if ( x * x <= y + y );
          {
            break;
          }
        }

        if ( hz <= 0 )
        {
          value = - r - x;
        }
        else
        {
          value = + r + x;
        }
        break;
      }

      x = ( float ) ( hz ) * wn[iz];

      if ( fn[iz] + r4_uni ( jsr ) * ( fn[iz-1] - fn[iz] ) < exp ( - 0.5 * x * x ) )
      {
        value = x;
        break;
      }

      hz = shr3 ( jsr );
      iz = ( hz & 127 );

      if ( abs ( hz ) < kn[iz] )
      {
        value = ( float ) ( hz ) * wn[iz];
        break;
      }
    }
  }

  return value;
}
//****************************************************************************80

void r4_nor_setup ( int kn[128], float fn[128], float wn[128] )

//****************************************************************************80
//
//  Purpose:
//
//    R4_NOR_SETUP sets data needed by R4_NOR.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    04 May 2008
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    George Marsaglia, Wai Wan Tsang,
//    The Ziggurat Method for Generating Random Variables,
//    Journal of Statistical Software,
//    Volume 5, Number 8, October 2000, seven pages.
//
//  Parameters:
//
//    Output, int KN[128], data needed by R4_NOR.
//
//    Output, float FN[128], WN[128], data needed by R4_NOR.
//
{
  double dn = 3.442619855899;
  int i;
  const double m1 = 2147483648.0;
  double q;
  double tn = 3.442619855899;
  const double vn = 9.91256303526217E-03;

  q = vn / exp ( - 0.5 * dn * dn );

  kn[0] = ( int ) ( ( dn / q ) * m1 );
  kn[1] = 0;

  wn[0] = ( float ) ( q / m1 );
  wn[127] = ( float ) ( dn / m1 );

  fn[0] = 1.0;
  fn[127] = ( float ) ( exp ( - 0.5 * dn * dn ) );

  for ( i = 126; 1 <= i; i-- )
  {
    dn = sqrt ( - 2.0 * log ( vn / dn + exp ( - 0.5 * dn * dn ) ) );
    kn[i+1] = ( int ) ( ( dn / tn ) * m1 );
    tn = dn;
    fn[i] = ( float ) ( exp ( - 0.5 * dn * dn ) );
    wn[i] = ( float ) ( dn / m1 );
  }
  return;
}
//****************************************************************************80

float r4_uni ( unsigned long int *jsr )

//****************************************************************************80
//
//  Purpose:
//
//    R4_UNI returns a uniformly distributed real value.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    20 May 2008
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    George Marsaglia, Wai Wan Tsang,
//    The Ziggurat Method for Generating Random Variables,
//    Journal of Statistical Software,
//    Volume 5, Number 8, October 2000, seven pages.
//
//  Parameters:
//
//    Input/output, unsigned long int *JSR, the seed.
//
//    Output, float R4_UNI, a uniformly distributed random value in
//    the range [0,1].
//
{
  unsigned long int jsr_input;
  float value;

  jsr_input = *jsr;

  *jsr = ( *jsr ^ ( *jsr <<   13 ) );
  *jsr = ( *jsr ^ ( *jsr >>   17 ) );
  *jsr = ( *jsr ^ ( *jsr <<    5 ) );

  value = fmod ( 0.5 + ( float ) ( jsr_input + *jsr ) / 65536.0 / 65536.0, 1.0 );

  return value;
}
//****************************************************************************80

unsigned long int shr3 ( unsigned long int *jsr )

//****************************************************************************80
//
//  Purpose:
//
//    SHR3 evaluates the SHR3 generator for integers.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 December 2008
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    George Marsaglia, Wai Wan Tsang,
//    The Ziggurat Method for Generating Random Variables,
//    Journal of Statistical Software,
//    Volume 5, Number 8, October 2000, seven pages.
//
//  Parameters:
//
//    Input/output, unsigned long int *JSR, the seed, which is updated 
//    on each call.
//
//    Output, unsigned long int SHR3, the new value.
//
{
  unsigned long int value;

  value = *jsr;

  *jsr = ( *jsr ^ ( *jsr <<   13 ) );
  *jsr = ( *jsr ^ ( *jsr >>   17 ) );
  *jsr = ( *jsr ^ ( *jsr <<    5 ) );

  value = value + *jsr;

  return value;
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    24 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}

int main(int argc, char** argv) {
  std::cout << "Testing..\n";
  unsigned long int seed = 1;
  int kn[128];
  float fn[128], wn[128];
  r4_nor_setup (kn, fn, wn);
  for (int i = 0; i < 10; i++) {
    std::cout << r4_nor (&seed, kn, fn, wn) << "\n";
  }
}
