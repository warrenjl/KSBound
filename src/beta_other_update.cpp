#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec beta_other_update(arma::mat x, 
                            int n,
                            int p_x,
                            arma::vec off_set,
                            arma::vec w,
                            arma::vec gamma,
                            arma::vec theta_old,
                            arma::vec g_old,
                            double sigma2_beta){

arma::mat w_mat(n, p_x);
for(int j = 0; j < p_x; ++j){
   w_mat.col(j) = w;
   }

arma::uvec g_subset_old = as<arma::uvec>(wrap(g_old)) - 1;
arma::vec theta_g_old = theta_old(g_subset_old);

arma::mat x_trans = trans(x);

arma::mat cov_beta = inv_sympd(x_trans*(w_mat%x) + 
                               (1.00/sigma2_beta)*eye(p_x, p_x));

arma::vec mean_beta = cov_beta*(x_trans*(w%(gamma - off_set - theta_g_old)));

arma::mat ind_norms = arma::randn(1, 
                                  p_x);
arma::vec beta = mean_beta + 
                 trans(ind_norms*arma::chol(cov_beta));

return(beta);

}



