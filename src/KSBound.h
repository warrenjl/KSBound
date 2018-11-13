#ifndef __KSBound__
#define __KSBound__

Rcpp::NumericVector sampleRcpp(Rcpp::NumericVector x,
                               int size,
                               bool replace,
                               Rcpp::NumericVector prob = Rcpp::NumericVector::create());

Rcpp::List beta_update(arma::vec y, 
                       arma::vec offset,
                       arma::mat x, 
                       arma::vec beta_old,  
                       arma::vec theta_old,
                       arma::vec g_old,
                       arma::vec beta_mu,
                       arma::vec beta_sd,
                       arma::vec mhvar_beta,
                       arma::vec acctot_beta);

Rcpp::List theta_update(arma::vec y,
                        arma::vec offset, 
                        arma::mat x, 
                        arma::vec beta,
                        arma::vec theta_old,
                        double sigma2_theta_old,
                        arma::vec g_old,
                        arma::vec mhvar_theta,
                        arma::vec acctot_theta);

double sigma2_theta_update(arma::vec theta,
                           double sigma2_theta_a,
                           double sigma2_theta_b);

arma::vec g_update(arma::vec y,
                   arma::vec offset, 
                   arma::mat x, 
                   arma::vec beta, 
                   arma::vec theta, 
                   arma::vec c, 
                   arma::vec u, 
                   arma::mat p);

Rcpp::List v_psi_update(int m_max,
                        arma::mat spatial_neighbors,
                        arma::vec y,
                        arma::vec g,
                        double alpha_old);

double alpha_update(arma::vec v,
                    double alpha_a,
                    double alpha_b);

Rcpp::List u_p_c_update(arma::mat spatial_neighbors,
                        arma::vec v,
                        arma::vec psi,
                        arma::vec g);

double neg_two_loglike_update(arma::vec y,
                              arma::vec offset, 
                              arma::mat x, 
                              arma::vec beta,  
                              arma::vec theta,
                              arma::vec g);

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
                   double neg_two_loglike_init); 

#endif // __KSBound__
