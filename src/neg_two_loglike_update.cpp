#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double neg_two_loglike_update(arma::vec y,
                              arma::vec offset, 
                              arma::mat x, 
                              arma::vec beta,  
                              arma::vec theta,
                              arma::vec g){

int n = y.size();
arma::vec dens(n); dens.fill(0);
arma::uvec g_subset = as<arma::uvec>(wrap(g)) - 1;
arma::vec theta_g = theta(g_subset);
arma::vec lambda = exp(offset + 
                       x*beta + 
                       theta_g);

for(int j = 0; j < n; ++j){
   dens(j) = R::dpois(y(j),
                      lambda(j),
                      TRUE);
   }
double neg_two_loglike = -2.0*sum(dens);

return neg_two_loglike;

}
















































