#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double neg_two_loglike_update(arma::vec y,
                              arma::mat x,
                              int n,
                              arma::vec off_set, 
                              arma::vec tri_als,
                              int likelihood_indicator,
                              int r,
                              double sigma2_epsilon,
                              arma::vec beta,  
                              arma::vec theta,
                              arma::vec g){

arma::vec dens(n); dens.fill(0.00);
arma::uvec g_subset = as<arma::uvec>(wrap(g)) - 1;
arma::vec theta_g = theta(g_subset);
arma::vec mu = off_set + 
               x*beta + 
               theta_g;

if(likelihood_indicator == 0){
  
  arma::vec lambda = exp(mu);
  for(int j = 0; j < n; ++j){
     dens(j) = R::dpois(y(j),
                        lambda(j),
                        TRUE);
     }
  
  }

if(likelihood_indicator == 1){
  
  arma::vec probs = exp(mu)/(1.00 + exp(mu));
  for(int j = 0; j < n; ++j){
     dens(j) = R::dbinom(y(j),
                         tri_als(j),
                         probs(j),
                         TRUE);
     }
  
  }

if(likelihood_indicator == 2){
  for(int j = 0; j < n; ++j){
     dens(j) = R::dnorm(y(j),
                        mu(j),
                        sqrt(sigma2_epsilon),
                        TRUE);
     }
  }

if(likelihood_indicator == 3){
  
  arma::vec probs = exp(mu)/(1.00 + exp(mu));
  for(int j = 0; j < n; ++j){
     dens(j) = R::dnbinom(y(j), 
                          r, 
                          (1.00 - probs(j)),        
                          TRUE);
     }
  
  }

double neg_two_loglike = -2.00*sum(dens);

return neg_two_loglike;

}
















































