#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double sigma2_theta_update(arma::vec theta,
                           double sigma2_theta_a,
                           double sigma2_theta_b){

int m_max = theta.size();

double sigma2_theta_a_update = 0.50*m_max +  
                               sigma2_theta_a;
double sigma2_theta_b_update = 0.50*sum(theta%theta) + 
                               sigma2_theta_b;

double sigma2_theta = 1.00/R::rgamma(sigma2_theta_a_update,
                                     (1.00/sigma2_theta_b_update));

return sigma2_theta;

}





























































