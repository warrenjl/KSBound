#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec g_update(arma::vec y,
                   arma::vec offset, 
                   arma::mat x, 
                   arma::vec beta, 
                   arma::vec theta, 
                   arma::vec c, 
                   arma::vec u, 
                   arma::mat p){
 
int n = y.size();
arma::vec mean_piece = offset + x*beta;
arma::vec lambda(1); lambda.fill(0.00);
arma::vec g(n); g.fill(0.00);

for(int j = 0; j < n; ++j){

   arma::vec log_val(c(j)); log_val.fill(0.00);
   arma::rowvec p_row = p.row(j);

   for(int k = 0; k < c(j); ++k){
      lambda = exp(mean_piece(j) + 
                   theta(k));

      log_val(k) = R::dpois(y(j), 
                            lambda(0), 
                            TRUE) +
                   log(u(j) <= p_row(k));
      }

   arma::vec probs(c(j)); probs.fill(0.00);

   for(int k = 0; k < c(j); ++k){
      probs(k) = 1.00/(sum(exp(log_val - log_val(k))));
      if(arma::is_finite(probs(k)) == 0.00){
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



