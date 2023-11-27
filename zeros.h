/* Written by Charles Harris charles.harris@sdl.usu.edu */
/* Modified to this project by Kasimir Tanner kasimir.tanner@gmail.com */

/* Modified to not depend on Python everywhere by Travis Oliphant.
 */

#ifndef ZEROS_H
#define ZEROS_H

extern double brentq(double xa, double xb, double xtol,
                     double rtol, int iter, double y, double V, double w_prime);

#endif
