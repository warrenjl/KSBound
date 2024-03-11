#ifndef __KSBound__
#define __KSBound__

arma::vec rcpp_pgdraw(arma::vec b, 
                      arma::vec c);

Rcpp::NumericVector sampleRcpp(Rcpp::NumericVector x,
                               int size,
                               bool replace,
                               Rcpp::NumericVector prob = Rcpp::NumericVector::create());

int r_update(arma::vec y,
             arma::mat x,
             int n,
             arma::vec off_set,
             arma::vec beta,
             arma::vec theta,
             arma::vec g,
             int r_a,
             int r_b);
  
double sigma2_epsilon_update(arma::vec y,
                             arma::mat x,
                             int n,
                             arma::vec off_set,
                             arma::vec beta_old,
                             arma::vec theta_old,
                             arma::vec g_old,
                             double sigma2_epsilon_a,
                             double sigma2_epsilon_b);
  
Rcpp::List w_update(arma::vec y,
                    arma::mat x,
                    int n,
                    arma::vec off_set,
                    arma::vec tri_als,
                    int likelihood_indicator,
                    int r,
                    arma::vec beta_old,
                    arma::vec theta_old,
                    arma::vec g_old);

Rcpp::List beta_update(arma::vec y, 
                       arma::mat x, 
                       int n,
                       int p_x,
                       arma::vec off_set,
                       arma::vec beta_old,  
                       arma::vec theta_old,
                       arma::vec g_old,
                       double sigma2_beta,
                       arma::vec mhvar_beta_vec,
                       arma::vec acctot_beta);

arma::vec beta_other_update(arma::mat x, 
                            int n,
                            int p_x,
                            arma::vec off_set,
                            arma::vec w,
                            arma::vec gamma,
                            arma::vec theta_old,
                            arma::vec g_old,
                            double sigma2_beta);

Rcpp::List theta_update(arma::vec y,
                        arma::mat x,
                        int n,
                        int m_max,
                        arma::vec off_set, 
                        arma::vec beta,
                        arma::vec theta_old,
                        double sigma2_theta_old,
                        arma::vec g_old,
                        arma::vec mhvar_theta_vec,
                        arma::vec acctot_theta);

arma::vec theta_other_update(arma::vec y,
                             arma::mat x,
                             int m_max,
                             arma::vec off_set, 
                             arma::vec w,
                             arma::vec gamma,
                             arma::vec beta,
                             double sigma2_theta_old,
                             arma::vec g_old);

arma::vec g_update(arma::vec y,
                   arma::mat x,
                   int n,
                   arma::vec off_set, 
                   arma::vec beta, 
                   arma::vec theta, 
                   arma::vec c, 
                   arma::vec u, 
                   arma::mat p);

arma::vec g_other_update(arma::vec y,
                         arma::mat x, 
                         int n,
                         arma::vec off_set, 
                         arma::vec w,
                         arma::vec gamma,
                         arma::vec beta, 
                         arma::vec theta, 
                         arma::vec c, 
                         arma::vec u, 
                         arma::mat p);

double sigma2_theta_update(int m_max,
                           arma::vec theta,
                           double sigma2_theta_a,
                           double sigma2_theta_b);

Rcpp::List v_psi_update(int n,
                        int m_max,
                        arma::mat spatial_neighbors,
                        arma::vec g,
                        double alpha_old);

double alpha_update(int m_max,
                    arma::vec v,
                    double alpha_a,
                    double alpha_b);

Rcpp::List u_p_c_update(int n,
                        int m_max,
                        arma::mat spatial_neighbors,
                        arma::vec v,
                        arma::vec psi,
                        arma::vec g);

double neg_two_loglike_update(arma::vec y,
                              arma::mat x, 
                              int n,
                              arma::vec off_set,
                              arma::vec tri_als,
                              int likelihood_indicator,
                              int r,
                              double sigma2_epsilon,
                              arma::vec beta,  
                              arma::vec theta,
                              arma::vec g);

Rcpp::List KSBound(int mcmc_samples,
                   int m_max,
                   arma::mat spatial_neighbors,
                   arma::vec y,
                   arma::mat x, 
                   arma::vec mhvar_beta,
                   arma::vec mhvar_theta,
                   int likelihood_indicator,
                   Rcpp::Nullable<Rcpp::NumericVector> offset,
                   Rcpp::Nullable<Rcpp::NumericVector> trials,
                   Rcpp::Nullable<double> r_a_prior,
                   Rcpp::Nullable<double> r_b_prior,
                   Rcpp::Nullable<double> sigma2_epsilon_a_prior,
                   Rcpp::Nullable<double> sigma2_epsilon_b_prior,
                   Rcpp::Nullable<double> sigma2_beta_prior,
                   Rcpp::Nullable<double> sigma2_theta_a_prior,
                   Rcpp::Nullable<double> sigma2_theta_b_prior,
                   Rcpp::Nullable<double> alpha_a_prior,
                   Rcpp::Nullable<double> alpha_b_prior,
                   Rcpp::Nullable<double> r_init,
                   Rcpp::Nullable<double> sigma2_epsilon_init,
                   Rcpp::Nullable<Rcpp::NumericVector> beta_init,
                   Rcpp::Nullable<Rcpp::NumericVector> theta_init,
                   Rcpp::Nullable<double> sigma2_theta_init,
                   Rcpp::Nullable<Rcpp::NumericVector> g_init,
                   Rcpp::Nullable<Rcpp::NumericVector> v_init,
                   Rcpp::Nullable<double> alpha_init,
                   Rcpp::Nullable<Rcpp::NumericVector> psi_init,
                   Rcpp::Nullable<double> alpha_fix,
                   Rcpp::Nullable<double> keep_all_ind); 

#endif // __KSBound__
