#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List beta_update(arma::vec y,
                       arma::vec offset, 
                       arma::mat x, 
                       arma::vec beta_old,  
                       arma::vec theta_old,
                       arma::vec g_old,
                       arma::vec beta_mu,
                       arma::vec beta_sd,
                       arma::vec mhvar_beta,
                       arma::vec acctot_beta){
  
int n = y.size();
int p_x = x.n_cols;
arma::vec lambda(n); lambda.fill(0);
arma::vec dens(n); dens.fill(0);
double second = 0;
double first = 0;
double ratio = 0;
int acc = 0;
arma::vec beta = beta_old;
arma::uvec g_subset = as<arma::uvec>(wrap(g_old)) - 1;
arma::vec theta_g = theta_old(g_subset);
  
for(int j = 0; j < p_x; ++j){
    
   /*Second*/
   lambda = exp(offset + 
                x*beta + 
                theta_g);
    
   for(int k = 0; k < n; ++k){
      dens(k) = R::dpois(y(k), 
                         lambda(k), 
                         TRUE);
      }
    
   second = sum(dens) +
            R::dnorm(beta(j),
                     beta_mu(j),
                     beta_sd(j),
                     TRUE); 
    
   /*First*/
   beta(j) = R::rnorm(beta_old(j), 
                      sqrt(mhvar_beta(j)));
    
   lambda = exp(offset + 
                x*beta + 
                theta_g);
    
   for(int k = 0; k < n; ++k){
      dens(k) = R::dpois(y(k), 
                         lambda(k), 
                         TRUE);
      }
    
   first = sum(dens) +
           R::dnorm(beta(j),
                    beta_mu(j),
                    beta_sd(j),
                    TRUE);
    
   /*Decision*/
   ratio = exp(first - second);  
   acc = 1;
   if(ratio < R::runif(0,1)){
     beta(j) = beta_old(j);
     acc = 0;
     }
   acctot_beta(j) = acctot_beta(j) + acc;
    
   }
  
return Rcpp::List::create(Rcpp::Named("beta")=beta,
                          Rcpp::Named("acctot_beta")=acctot_beta);

}
