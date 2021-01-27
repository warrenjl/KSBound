#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List theta_update(arma::vec y,
                        arma::vec offset, 
                        arma::mat x, 
                        arma::vec beta,
                        arma::vec theta_old,
                        double sigma2_theta_old,
                        arma::vec g_old,
                        arma::vec mhvar_theta,
                        arma::vec acctot_theta){
  
int n = y.size();
int m_max = theta_old.size();
double second = 0.00;
double first = 0.00;
double ratio = 0.00;
int acc = 0;
arma::vec theta = theta_old;
  
for(int j = 0; j < max(g_old); ++j){
    
   if(sum((g_old - 1) == j) == 0){
     theta(j) = R::rnorm(0,
                         sqrt(sigma2_theta_old));
     }
    
   if(sum((g_old - 1) == j) > 0){
     
     arma::vec dens(n); dens.fill(0.00);
     arma::vec lambda(n); lambda.fill(0.00);
      
     /*Second*/
     lambda = exp(offset + 
                  x*beta + 
                  theta(j));
      
     for(int k = 0; k < n; ++k){ 
        if((g_old(k) - 1) == j){
          dens(k) = R::dpois(y(k), 
                             lambda(k),
                             TRUE);
          }
        }
      
     second = sum(dens) +
              R::dnorm(theta(j),
                       0.00,
                       sqrt(sigma2_theta_old),
                       TRUE);
      
     /*First*/
     theta(j) = R::rnorm(theta_old(j), 
                         sqrt(mhvar_theta(j)));
      
     lambda = exp(offset + 
                  x*beta + 
                  theta(j));
      
     for(int k = 0; k < n; ++k){
        if((g_old(k) - 1) == j){
          dens(k) = R::dpois(y(k), 
                             lambda(k),
                             TRUE);
          }
        }
      
     first = sum(dens) +
             R::dnorm(theta(j),
                      0.00,
                      sqrt(sigma2_theta_old),
                      TRUE);
      
     /*Decision*/
     ratio = exp(first - second);   
     acc = 1;
     if(ratio < R::runif(0.00, 1.00)){
       theta(j) = theta_old(j);
       acc = 0;
       }
     acctot_theta(j) = acctot_theta(j) + acc;   
      
     }
    
   }
  
for(int j = max(g_old); j < m_max; ++j){
   theta(j) = R::rnorm(0.00,
                       sqrt(sigma2_theta_old));
   }
  
return Rcpp::List::create(Rcpp::Named("theta")=theta,
                          Rcpp::Named("acctot_theta")=acctot_theta);
  
}

