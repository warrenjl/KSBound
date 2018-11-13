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
                   arma::vec beta_mu,
                   arma::vec beta_sd,
                   double sigma2_theta_a,
                   double sigma2_theta_b,
                   double alpha_a,
                   double alpha_b,
                   arma::vec mhvar_beta,
                   arma::vec mhvar_theta,
                   arma::vec beta_init,
                   arma::vec theta_init,
                   double sigma2_theta_init,
                   arma::vec g_init,
                   arma::vec v_init,
                   double alpha_init,
                   arma::vec psi_init,
                   arma::vec u_init,
                   arma::vec c_init,
                   arma::mat p_init,
                   double neg_two_loglike_init){

//Defining Parameters and Quantities of Interest
int n = y.size();
arma::mat beta(x.n_cols, mcmc_samples); beta.fill(0);
arma::mat theta(m_max, mcmc_samples); theta.fill(0);
arma::vec sigma2_theta(mcmc_samples); sigma2_theta.fill(0);
arma::mat g(n, mcmc_samples); g.fill(0);
arma::mat v(m_max, mcmc_samples); v.fill(0);
arma::vec alpha(mcmc_samples); alpha.fill(0);
arma::mat psi(m_max, mcmc_samples); psi.fill(0);
arma::vec u(n); u.fill(0);
arma::vec c(n); c.fill(0);
arma::mat p(n, m_max); p.fill(0);
arma::vec neg_two_loglike(mcmc_samples); neg_two_loglike.fill(0);

//Initial Values
beta.col(0) = beta_init;
theta.col(0) = theta_init;
sigma2_theta(0) = sigma2_theta_init;
g.col(0) = g_init;
v.col(0) = v_init;
alpha(0) = alpha_init;
psi.col(0) = psi_init;
u = u_init;
c = c_init;
p = p_init;
neg_two_loglike(0) = neg_two_loglike_init;

//Metropolis Settings
arma::vec acctot_beta(x.n_cols); acctot_beta.fill(0);
arma::vec acctot_theta(m_max); acctot_theta.fill(0);

for(int j = 1; j < mcmc_samples; ++j){
  
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
  
  //u, p, c Update
  Rcpp::List u_p_c_output = u_p_c_update(spatial_neighbors,
                                         v.col(j),
                                         psi.col(j),
                                         g.col(j));
  
  u = Rcpp::as<arma::vec>(u_p_c_output[0]);
  p = Rcpp::as<arma::mat>(u_p_c_output[1]);
  c = Rcpp::as<arma::vec>(u_p_c_output[2]);
  
  //neg_two_loglike Update
  neg_two_loglike(j) = neg_two_loglike_update(y,
                                              offset, 
                                              x, 
                                              beta.col(j),  
                                              theta.col(j),
                                              g.col(j));
  
  //Progress
  if(j % 10 == 0){ 
    Rcpp::checkUserInterrupt();
    }
  
  if((j % 10) == 0){
    Rcpp::Rcout << j << std::endl;
    Rcpp::Rcout << neg_two_loglike(j) << std::endl;
    }
  
  }
                                  
return Rcpp::List::create(Rcpp::Named("beta")=beta,
                          Rcpp::Named("acctot_beta")=acctot_beta,
                          Rcpp::Named("theta")=theta,
                          Rcpp::Named("acctot_theta")=acctot_theta,
                          Rcpp::Named("sigma2_theta")=sigma2_theta,
                          Rcpp::Named("g")=g,
                          Rcpp::Named("v")=v,
                          Rcpp::Named("psi")=psi,
                          Rcpp::Named("alpha")=alpha,
                          Rcpp::Named("neg_two_loglike")=neg_two_loglike);

}

