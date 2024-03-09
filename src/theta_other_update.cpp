#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec theta_other_update(arma::vec y,
                             arma::mat x,
                             int m_max,
                             arma::vec off_set, 
                             arma::vec w,
                             arma::vec gamma,
                             arma::vec beta,
                             double sigma2_theta_old,
                             arma::vec g_old){
  
arma::vec theta(m_max); 

for(int j = 0; j < max(g_old); ++j){
    
   if(sum((g_old - 1) == j) == 0){
     theta(j) = R::rnorm(0.00,
                         sqrt(sigma2_theta_old));
     }
    
   if(sum((g_old - 1) == j) > 0){
     
     arma::vec mu_beta = x*beta;
     arma::mat ones_vec(sum((g_old - 1) == j), 1); ones_vec.fill(1.00);
     arma::uvec temp_set = find((g_old - 1) == j);
     
     arma::mat ones_vec_trans = trans(ones_vec);
     arma::mat cov_theta = inv_sympd(ones_vec_trans*(w.elem(temp_set)%ones_vec) + (1.00/sigma2_theta_old));
     arma::vec mean_theta = cov_theta*(ones_vec_trans*(w.elem(temp_set)%(gamma.elem(temp_set) - off_set.elem(temp_set) - mu_beta.elem(temp_set))));
     
     arma::mat ind_norm = arma::randn(1, 
                                      1);
     arma::vec theta_temp = mean_theta + 
                            trans(ind_norm*arma::chol(cov_theta));
     theta(j) = theta_temp(0);
     
     }
    
   }
  
for(int j = max(g_old); j < m_max; ++j){
   theta(j) = R::rnorm(0.00,
                       sqrt(sigma2_theta_old));
   }
  
return theta;
  
}

