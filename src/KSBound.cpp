#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List KSBound(int mcmc_samples,
                   int m_max,
                   arma::mat spatial_neighbors,
                   arma::vec y,
                   arma::vec offset, 
                   arma::mat x, 
                   arma::vec mhvar_beta,
                   arma::vec mhvar_theta,
                   Rcpp::Nullable<Rcpp::NumericVector> beta_mu_prior = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> beta_sd_prior = R_NilValue,
                   Rcpp::Nullable<double> sigma2_theta_a_prior = R_NilValue,
                   Rcpp::Nullable<double> sigma2_theta_b_prior = R_NilValue,
                   Rcpp::Nullable<double> alpha_a_prior = R_NilValue,
                   Rcpp::Nullable<double> alpha_b_prior = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> beta_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> theta_init = R_NilValue,
                   Rcpp::Nullable<double> sigma2_theta_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> g_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> v_init = R_NilValue,
                   Rcpp::Nullable<double> alpha_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> psi_init = R_NilValue){

//Defining Parameters and Quantities of Interest
int n = y.size();
int p_x = x.n_cols;
arma::mat beta(p_x, mcmc_samples); beta.fill(0.00);
arma::mat theta(m_max, mcmc_samples); theta.fill(0.00);
arma::vec sigma2_theta(mcmc_samples); sigma2_theta.fill(0.00);
arma::mat g(n, mcmc_samples); g.fill(0.00);
arma::mat v(m_max, mcmc_samples); v.fill(0.00);
arma::vec alpha(mcmc_samples); alpha.fill(0.00);
arma::mat psi(m_max, mcmc_samples); psi.fill(0.00);
arma::vec u(n); u.fill(0);
arma::vec c(n); c.fill(0);
arma::mat p(n, m_max); p.fill(0.00);
arma::vec neg_two_loglike(mcmc_samples); neg_two_loglike.fill(0.00);

//Prior Information
arma::vec beta_mu(p_x); beta_mu.fill(0.00);
if(beta_mu_prior.isNotNull()){
  beta_mu = Rcpp::as<arma::vec>(beta_mu_prior);
  }

arma::vec beta_sd(p_x); beta_sd.fill(100.00);
if(beta_sd_prior.isNotNull()){
  beta_sd = Rcpp::as<arma::vec>(beta_sd_prior);
  }

double sigma2_theta_a = 0.01;
if(sigma2_theta_a_prior.isNotNull()){
  sigma2_theta_a = Rcpp::as<double>(sigma2_theta_a_prior);
  }

double sigma2_theta_b = 0.01;
if(sigma2_theta_b_prior.isNotNull()){
  sigma2_theta_b = Rcpp::as<double>(sigma2_theta_b_prior);
  }

double alpha_a = 0.01;
if(alpha_a_prior.isNotNull()){
  alpha_a = Rcpp::as<double>(alpha_a_prior);
  }

double alpha_b = 0.01;
if(alpha_b_prior.isNotNull()){
  alpha_b = Rcpp::as<double>(alpha_b_prior);
  }

//Initial Values
beta.col(0).fill(0.00);
if(beta_init.isNotNull()){
  beta.col(0) = Rcpp::as<arma::vec>(beta_init);
  }

theta.col(0).fill(0.00);
if(theta_init.isNotNull()){
  theta.col(0) = Rcpp::as<arma::vec>(theta_init);
  }

sigma2_theta(0) = 1.00;
if(sigma2_theta_init.isNotNull()){
  sigma2_theta(0) = Rcpp::as<double>(sigma2_theta_init);
  }

alpha(0) = 1.00;
if(alpha_init.isNotNull()){
  alpha(0) = Rcpp::as<double>(alpha_init);
  }

v.col(0).fill(0.999);
if(v_init.isNotNull()){
  v.col(0) = Rcpp::as<arma::vec>(v_init);
  }

psi.col(0).fill(1.00);
for(int j = 0; j < n; ++j){
   psi(j,0) = (j + 1.00);
   }
if(psi_init.isNotNull()){
  psi.col(0) = Rcpp::as<arma::vec>(psi_init);
  }

for(int j = 0; j < n; ++j){
  
   double stop = 0.00;
   for(int k = 0; k < n; ++k){
     
      if(stop == 0.00){
        if(spatial_neighbors(j,k) == 1){
             
          g(j,0) = (k + 1);
          stop = 1.00;
             
          }
        }
      
      }
   }
if(g_init.isNotNull()){
  g.col(0) = Rcpp::as<arma::vec>(g_init);
  }

//Metropolis Settings
arma::vec acctot_beta(p_x); acctot_beta.fill(0.00);
arma::vec acctot_theta(m_max); acctot_theta.fill(0.00);

//Main Sampling Loop
for(int j = 1; j < mcmc_samples; ++j){
  
  //u, p, c Update
  Rcpp::List u_p_c_output = u_p_c_update(spatial_neighbors,
                                         v.col(j-1),
                                         psi.col(j-1),
                                         g.col(j-1));
  
  u = Rcpp::as<arma::vec>(u_p_c_output[0]);
  p = Rcpp::as<arma::mat>(u_p_c_output[1]);
  c = Rcpp::as<arma::vec>(u_p_c_output[2]);
  
  //beta Update
  Rcpp::List beta_output = beta_update(y,
                                       offset, 
                                       x,
                                       beta.col(j-1),  
                                       theta.col(j-1),
                                       g.col(j-1),
                                       beta_mu,
                                       beta_sd,
                                       mhvar_beta,
                                       acctot_beta);
  beta.col(j) = Rcpp::as<arma::colvec>(beta_output[0]);
  acctot_beta = Rcpp::as<arma::colvec>(beta_output[1]);
  
  //theta Update
  Rcpp::List theta_output = theta_update(y,
                                         offset, 
                                         x, 
                                         beta.col(j),
                                         theta.col(j-1),
                                         sigma2_theta(j-1),
                                         g.col(j-1),
                                         mhvar_theta,
                                         acctot_theta);
  
  theta.col(j) = Rcpp::as<arma::vec>(theta_output[0]);
  acctot_theta = Rcpp::as<arma::vec>(theta_output[1]);

  //sigma2_theta Update
  sigma2_theta(j) = sigma2_theta_update(theta.col(j),
                                        sigma2_theta_a,
                                        sigma2_theta_b);
  
  //g Update
  g.col(j) = g_update(y,
                      offset, 
                      x, 
                      beta.col(j), 
                      theta.col(j), 
                      c, 
                      u, 
                      p);
  
  //v, psi Update
  Rcpp::List v_psi_output = v_psi_update(m_max,
                                         spatial_neighbors,
                                         y,
                                         g.col(j),
                                         alpha(j-1));
  
  v.col(j) = Rcpp::as<arma::vec>(v_psi_output[0]);
  psi.col(j) = Rcpp::as<arma::vec>(v_psi_output[1]);
  
  //alpha Update
  alpha(j) = alpha_update(v.col(j),
                          alpha_a,
                          alpha_b);
  
  //neg_two_loglike Update
  neg_two_loglike(j) = neg_two_loglike_update(y,
                                              offset, 
                                              x, 
                                              beta.col(j),  
                                              theta.col(j),
                                              g.col(j));
  
  //Progress
  if((j + 1) % 10 == 0){ 
    Rcpp::checkUserInterrupt();
    }
  
  if(((j + 1) % int(round(mcmc_samples*0.10)) == 0)){
    
    double completion = round(100*((j + 1)/(double)mcmc_samples));
    Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
    
    Rcpp::Rcout << "**************" << std::endl;
    
    }
  
  }
                                  
return Rcpp::List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("acctot_beta") = acctot_beta,
                          Rcpp::Named("theta") = theta,
                          Rcpp::Named("acctot_theta") = acctot_theta,
                          Rcpp::Named("sigma2_theta") = sigma2_theta,
                          Rcpp::Named("g") = g,
                          Rcpp::Named("v") = v,
                          Rcpp::Named("psi") = psi,
                          Rcpp::Named("alpha") = alpha,
                          Rcpp::Named("neg_two_loglike") = neg_two_loglike);

}

