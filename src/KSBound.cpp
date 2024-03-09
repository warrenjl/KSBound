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
                   arma::mat x, 
                   int likelihood_indicator,
                   Rcpp::Nullable<Rcpp::NumericVector> offset = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> trials = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> mhvar_beta = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> mhvar_theta = R_NilValue,
                   Rcpp::Nullable<double> r_a_prior = R_NilValue,
                   Rcpp::Nullable<double> r_b_prior = R_NilValue,
                   Rcpp::Nullable<double> sigma2_epsilon_a_prior = R_NilValue,
                   Rcpp::Nullable<double> sigma2_epsilon_b_prior = R_NilValue,
                   Rcpp::Nullable<double> sigma2_beta_prior = R_NilValue,
                   Rcpp::Nullable<double> sigma2_theta_a_prior = R_NilValue,
                   Rcpp::Nullable<double> sigma2_theta_b_prior = R_NilValue,
                   Rcpp::Nullable<double> alpha_a_prior = R_NilValue,
                   Rcpp::Nullable<double> alpha_b_prior = R_NilValue,
                   Rcpp::Nullable<double> r_init = R_NilValue,
                   Rcpp::Nullable<double> sigma2_epsilon_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> beta_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> theta_init = R_NilValue,
                   Rcpp::Nullable<double> sigma2_theta_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> g_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> v_init = R_NilValue,
                   Rcpp::Nullable<double> alpha_init = R_NilValue,
                   Rcpp::Nullable<Rcpp::NumericVector> psi_init = R_NilValue,
                   Rcpp::Nullable<double> alpha_fix = R_NilValue,
                   Rcpp::Nullable<double> keep_all_ind = R_NilValue){

//Defining Parameters and Quantities of Interest
int n = y.size();
int p_x = x.n_cols;
arma::vec r(mcmc_samples); r.fill(0.00);
arma::vec sigma2_epsilon(mcmc_samples); sigma2_epsilon.fill(0.00);
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
arma::mat theta_g(n, mcmc_samples); theta_g.fill(0.00);

arma::vec off_set(n); off_set.fill(0.00);
if(offset.isNotNull()){
  off_set = Rcpp::as<arma::vec>(offset);
  }

arma::vec tri_als(n); tri_als.fill(1);
if(trials.isNotNull()){
  tri_als = Rcpp::as<arma::vec>(trials);
  }

//Prior Information
int r_a = 1;
if(r_a_prior.isNotNull()){
  r_a = Rcpp::as<int>(r_a_prior);
  }

int r_b = 100;
if(r_b_prior.isNotNull()){
  r_b = Rcpp::as<int>(r_b_prior);
  }

double sigma2_epsilon_a = 0.01;
if(sigma2_epsilon_a_prior.isNotNull()){
  sigma2_epsilon_a = Rcpp::as<double>(sigma2_epsilon_a_prior);
  }

double sigma2_epsilon_b = 0.01;
if(sigma2_epsilon_b_prior.isNotNull()){
  sigma2_epsilon_b = Rcpp::as<double>(sigma2_epsilon_b_prior);
  }

double sigma2_beta = 10000.00;
if(sigma2_beta_prior.isNotNull()){
  sigma2_beta = Rcpp::as<double>(sigma2_beta_prior);
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

if(alpha_fix.isNotNull()){
  alpha.fill(Rcpp::as<double>(alpha_fix));
  }

//Initial Values
r(0) = r_b;
if(r_init.isNotNull()){
  r(0) = Rcpp::as<int>(r_init);
  }

sigma2_epsilon(0) = var(y);
if(sigma2_epsilon_init.isNotNull()){
  sigma2_epsilon(0) = Rcpp::as<double>(sigma2_epsilon_init);
  }

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

neg_two_loglike(0) = neg_two_loglike_update(y,
                                            x,
                                            n,
                                            off_set, 
                                            likelihood_indicator,
                                            r(0),
                                            sigma2_epsilon(0),
                                            beta.col(0),  
                                            theta.col(0),
                                            g.col(0));

//Metropolis Settings
arma::vec acctot_beta(p_x); acctot_beta.fill(0.00);
arma::vec acctot_theta(m_max); acctot_theta.fill(0.00);

arma::vec mhvar_beta_vec(p_x); mhvar_beta_vec.fill(-arma::math::inf());
if(mhvar_beta.isNotNull()){
  mhvar_beta_vec = Rcpp::as<arma::vec>(mhvar_beta);
  }

arma::vec mhvar_theta_vec(p_x); mhvar_theta_vec.fill(-arma::math::inf());
if(mhvar_theta.isNotNull()){
  mhvar_theta_vec = Rcpp::as<arma::vec>(mhvar_theta);
  }

//Main Sampling Loop
arma::vec w(n); w.fill(0.00);
arma::vec gamma = y;
if(likelihood_indicator == 3){
  
  Rcpp::List w_output = w_update(y,
                                 x,
                                 n,
                                 off_set,
                                 tri_als,
                                 likelihood_indicator,
                                 r(0),
                                 beta.col(0),
                                 theta.col(0),
                                 g.col(0));
  w = Rcpp::as<arma::vec>(w_output[0]);
  gamma = Rcpp::as<arma::vec>(w_output[1]);
  
  }

for(int j = 1; j < mcmc_samples; ++j){
  
  if(likelihood_indicator == 2){
    
    //sigma2_epsilon Update
    sigma2_epsilon(j) = sigma2_epsilon_update(y,
                                              x,
                                              n,
                                              off_set,
                                              beta.col(j-1),
                                              theta.col(j-1),
                                              g.col(j-1),
                                              sigma2_epsilon_a,
                                              sigma2_epsilon_b);
    w.fill(1.00/sigma2_epsilon(j));
    
    }
  
  if(likelihood_indicator == 1){
    
    //w Update
    Rcpp::List w_output = w_update(y,
                                   x,
                                   n,
                                   off_set,
                                   tri_als,
                                   likelihood_indicator,
                                   r(j-1),
                                   beta.col(j-1),
                                   theta.col(j-1),
                                   g.col(j-1));
    w = Rcpp::as<arma::vec>(w_output[0]);
    gamma = Rcpp::as<arma::vec>(w_output[1]);
    
    }
  
  //u, p, c Update
  Rcpp::List u_p_c_output = u_p_c_update(n,
                                         m_max,
                                         spatial_neighbors,
                                         v.col(j-1),
                                         psi.col(j-1),
                                         g.col(j-1));
  
  u = Rcpp::as<arma::vec>(u_p_c_output[0]);
  p = Rcpp::as<arma::mat>(u_p_c_output[1]);
  c = Rcpp::as<arma::vec>(u_p_c_output[2]);
  
  //beta, theta, g Updates (Poisson)
  if(likelihood_indicator == 0){
    
    //beta Update
    Rcpp::List beta_output = beta_update(y,
                                         x,
                                         n,
                                         p_x,
                                         off_set, 
                                         beta.col(j-1),  
                                         theta.col(j-1),
                                         g.col(j-1),
                                         sigma2_beta,
                                         mhvar_beta_vec,
                                         acctot_beta);
    beta.col(j) = Rcpp::as<arma::vec>(beta_output[0]);
    acctot_beta = Rcpp::as<arma::vec>(beta_output[1]);
    
    //theta Update
    Rcpp::List theta_output = theta_update(y,
                                           x,
                                           n,
                                           m_max,
                                           off_set,
                                           beta.col(j),
                                           theta.col(j-1),
                                           sigma2_theta(j-1),
                                           g.col(j-1),
                                           mhvar_theta_vec,
                                           acctot_theta);
    
    theta.col(j) = Rcpp::as<arma::vec>(theta_output[0]);
    acctot_theta = Rcpp::as<arma::vec>(theta_output[1]);
    
    //g Update
    g.col(j) = g_update(y,
                        x, 
                        n,
                        off_set, 
                        beta.col(j), 
                        theta.col(j), 
                        c, 
                        u, 
                        p);
    
    }
  
  //beta, theta, g Updates (Other)
  if(likelihood_indicator != 0){
    
    //beta Update
    beta.col(j) = beta_other_update(x,
                                    n,
                                    p_x,
                                    off_set, 
                                    w,
                                    gamma,
                                    theta.col(j-1),
                                    g.col(j-1),
                                    sigma2_beta);
    
    //theta Update
    theta.col(j) = theta_other_update(y,
                                      x,
                                      m_max,
                                      off_set,
                                      w,
                                      gamma,
                                      beta.col(j),
                                      sigma2_theta(j-1),
                                      g.col(j-1));
    
    //g Update
    g.col(j) = g_other_update(y,
                              x,
                              n,
                              off_set, 
                              w,
                              gamma,
                              beta.col(j), 
                              theta.col(j), 
                              c, 
                              u, 
                              p);

    }

  //sigma2_theta Update
  sigma2_theta(j) = sigma2_theta_update(m_max,
                                        theta.col(j),
                                        sigma2_theta_a,
                                        sigma2_theta_b);
  
  //v, psi Update
  Rcpp::List v_psi_output = v_psi_update(n,
                                         m_max,
                                         spatial_neighbors,
                                         g.col(j),
                                         alpha(j-1));
  
  v.col(j) = Rcpp::as<arma::vec>(v_psi_output[0]);
  psi.col(j) = Rcpp::as<arma::vec>(v_psi_output[1]);
  
  //alpha Update
  if(alpha_fix.isNotNull() == 0){
    alpha(j) = alpha_update(m_max,
                            v.col(j),
                            alpha_a,
                            alpha_b);
    }
  
  if(likelihood_indicator == 3){
    
    //r Update
    r(j) = r_update(y,
                    x,
                    n,
                    off_set,
                    beta.col(j),
                    theta.col(j),
                    g.col(j),
                    r_a,
                    r_b);
    
    //w Update
    Rcpp::List w_output = w_update(y,
                                   x,
                                   n,
                                   off_set,
                                   tri_als,
                                   likelihood_indicator,
                                   r(j),
                                   beta.col(j),
                                   theta.col(j),
                                   g.col(j));
    w = Rcpp::as<arma::vec>(w_output[0]);
    gamma = Rcpp::as<arma::vec>(w_output[1]);
    
    }
  
  //neg_two_loglike Update
  neg_two_loglike(j) = neg_two_loglike_update(y,
                                              x,
                                              n,
                                              off_set, 
                                              likelihood_indicator,
                                              r(j),
                                              sigma2_epsilon(j),
                                              beta.col(j),  
                                              theta.col(j),
                                              g.col(j));
  
  //Combined Random Effect Update
  arma::uvec g_temp = as<arma::uvec>(wrap(g.col(j))) - 1;
  arma::vec theta_temp = theta.col(j);
  theta_g.col(j) = theta_temp(g_temp);
  
  //Progress
  if((j + 1) % 10 == 0){ 
    Rcpp::checkUserInterrupt();
    }
  
  if(((j + 1) % int(round(mcmc_samples*0.10)) == 0)){
    
    double completion = round(100*((j + 1)/(double)mcmc_samples));
    Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
    
    if(likelihood_indicator == 0){

      if(p_x == 1){
      
        double accrate_beta = round(100*(min(acctot_beta)/(double)j));
        Rcpp::Rcout << "beta Acceptance: " << accrate_beta << "%" << std::endl;
       
        }
    
      if(p_x > 1){
      
        double accrate_beta_min = round(100*(min(acctot_beta)/(double)j));
        Rcpp::Rcout << "beta Acceptance (min): " << accrate_beta_min << "%" << std::endl;
    
        double accrate_beta_max = round(100*(max(acctot_beta)/(double)j));
        Rcpp::Rcout << "beta Acceptance (max): " << accrate_beta_max << "%" << std::endl;
      
        }
    
      double accrate_theta_min = round(100*(min(acctot_theta.subvec(0, (max(g.col(j)) - 1)))/(double)j));
      Rcpp::Rcout << "theta Acceptance (min): " << accrate_theta_min << "%" << std::endl;
    
      double accrate_theta_max = round(100*(max(acctot_theta.subvec(0, (max(g.col(j)) - 1)))/(double)j));
      Rcpp::Rcout << "theta Acceptance (max): " << accrate_theta_max << "%" << std::endl;
    
      Rcpp::Rcout << "****************************" << std::endl;
      
      }
    
    }
  
  }
  
double keep_all = 0.00;
if(keep_all_ind.isNotNull()){
  keep_all = Rcpp::as<double>(keep_all_ind);
  }

if(keep_all == 0.00){
  return Rcpp::List::create(Rcpp::Named("r") = r,
                            Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                            Rcpp::Named("beta") = beta,
                            Rcpp::Named("theta_g") = theta_g,
                            Rcpp::Named("sigma2_theta") = sigma2_theta,
                            Rcpp::Named("alpha") = alpha,
                            Rcpp::Named("neg_two_loglike") = neg_two_loglike);
  }

if(keep_all != 0.00){
  return Rcpp::List::create(Rcpp::Named("r") = r,
                            Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                            Rcpp::Named("beta") = beta,
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

}

