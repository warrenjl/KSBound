#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List beta_update(arma::vec y,
                       arma::mat x,
                       int n,
                       int p_x,
                       arma::vec off_set, 
                       arma::vec beta_old,  
                       arma::vec theta_old,
                       arma::vec g_old,
                       double sigma2_beta,
                       arma::vec mhvar_beta,
                       arma::vec acctot_beta){
  
arma::vec lambda(n); lambda.fill(0.00);
arma::vec dens(n); dens.fill(0.00);
double second = 0.00;
double first = 0.00;
double ratio = 0.00;
int acc = 0;
arma::vec beta = beta_old;
arma::uvec g_subset = as<arma::uvec>(wrap(g_old)) - 1;
arma::vec theta_g = theta_old(g_subset);
  
for(int j = 0; j < p_x; ++j){
    
   /*Second*/
   lambda = exp(off_set + 
                x*beta + 
                theta_g);
    
   for(int k = 0; k < n; ++k){
      dens(k) = R::dpois(y(k), 
                         lambda(k), 
                         TRUE);
      }
    
   second = sum(dens) +
            R::dnorm(beta(j),
                     0.00,
                     sqrt(sigma2_beta),
                     TRUE); 
    
   /*First*/
   beta(j) = R::rnorm(beta_old(j), 
                      sqrt(mhvar_beta(j)));
    
   lambda = exp(off_set + 
                x*beta + 
                theta_g);
    
   for(int k = 0; k < n; ++k){
      dens(k) = R::dpois(y(k), 
                         lambda(k), 
                         TRUE);
      }
    
   first = sum(dens) +
           R::dnorm(beta(j),
                    0.00,
                    sqrt(sigma2_beta),
                    TRUE);
    
   /*Decision*/
   ratio = exp(first - second);  
   acc = 1;
   if(ratio < R::runif(0.00, 1.00)){
      
     beta(j) = beta_old(j);
     acc = 0;
     
     }
   acctot_beta(j) = acctot_beta(j) + acc;
    
   }
  
return Rcpp::List::create(Rcpp::Named("beta")=beta,
                          Rcpp::Named("acctot_beta")=acctot_beta);

}
