#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double alpha_update(arma::vec v,
                    double alpha_a,
                    double alpha_b){

int m_max = v.size();

double alpha_a_update = alpha_a +
                        m_max;

double alpha_b_update = alpha_b -
                        sum(log(1 - v));

double alpha = R::rgamma(alpha_a_update,
                         (1/alpha_b_update));

return(alpha);

} 
                         