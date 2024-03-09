#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec g_other_update(arma::vec y,
                         arma::mat x, 
                         int n,
                         arma::vec off_set, 
                         arma::vec w,
                         arma::vec gamma,
                         arma::vec beta, 
                         arma::vec theta, 
                         arma::vec c, 
                         arma::vec u, 
                         arma::mat p){
 
arma::vec mu = off_set + 
               x*beta;
arma::vec g(n); g.fill(0.00);

for(int j = 0; j < n; ++j){

   arma::vec log_val(c(j)); log_val.fill(0.00);
   arma::rowvec p_row = p.row(j);

   for(int k = 0; k < c(j); ++k){
     
      log_val(k) = -0.50*pow((mu(j) + theta(k) - gamma(j)), 2)*w(j) +
                   log(u(j) <= p_row(k));
      
      }

   arma::vec probs(c(j)); probs.fill(0.00);

   for(int k = 0; k < c(j); ++k){
     
      probs(k) = 1.00/(sum(exp(log_val - log_val(k))));
      if(arma::is_finite(probs(k)) == 0){
        probs(k) = 0.00;  /*Computational Correction*/
        }
      
      }

   IntegerVector sample_set = seq(1, c(j));
   g(j) = sampleRcpp(wrap(sample_set), 
                     1, 
                     TRUE, 
                     wrap(probs))(0);
  
   }

return g;

}



