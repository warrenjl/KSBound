#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double sigma2_epsilon_update(arma::vec y,
                             arma::mat x,
                             int n,
                             arma::vec off_set,
                             arma::vec beta_old,
                             arma::vec theta_old,
                             arma::vec g_old,
                             double sigma2_epsilon_a,
                             double sigma2_epsilon_b){

arma::uvec g_subset_old = as<arma::uvec>(wrap(g_old)) - 1;
arma::vec theta_g_old = theta_old(g_subset_old);
  
double sigma2_epsilon_a_update = 0.50*n + 
                                 sigma2_epsilon_a;

double sigma2_epsilon_b_update = 0.50*dot((y - off_set - x*beta_old - theta_g_old), (y - off_set - x*beta_old - theta_g_old)) + 
                                 sigma2_epsilon_b;

double sigma2_epsilon = 1.00/R::rgamma(sigma2_epsilon_a_update,
                                       (1.00/sigma2_epsilon_b_update));

return(sigma2_epsilon);

}



