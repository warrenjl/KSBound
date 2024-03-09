#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

int r_update(arma::vec y,
             arma::mat x,
             int n,
             arma::vec off_set,
             arma::vec beta,
             arma::vec theta,
             arma::vec g,
             int r_a,
             int r_b){

arma::uvec g_subset = as<arma::uvec>(wrap(g)) - 1;
arma::vec theta_g = theta(g_subset);
arma::vec mu = off_set + 
               x*beta + 
               theta_g;
  
arma::vec prob = 1.00/(1.00 + exp(-mu));
  
arma::vec r_log_val(r_b - r_a + 1); r_log_val.fill(0.00);  
int counter = 0;
for(int j = (r_a - 1); j < r_b; ++j){

   for(int k = 0; k < n; ++k){
      r_log_val(counter) = r_log_val(counter) +
                           R::dnbinom(y(k),
                                      (j + 1),
                                      (1.00 - prob(k)),
                                      TRUE);
      }
   counter = counter +
             1;
  
   }
  
arma::vec r_prob(r_b - r_a + 1); r_prob.fill(0.00);
for(int j = 0; j < (r_b - r_a + 1); ++j){
   r_prob(j) = 1.00/sum(exp(r_log_val - r_log_val(j)));
   }

IntegerVector sample_set = seq(r_a, r_b);
int r = sampleRcpp(wrap(sample_set), 
                   1, 
                   TRUE, 
                   wrap(r_prob))(0);
    
return(r);

}





