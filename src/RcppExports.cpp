// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// KSBound
Rcpp::List KSBound(int mcmc_samples, int m_max, arma::mat spatial_neighbors, arma::vec y, arma::mat x, int likelihood_indicator, Rcpp::Nullable<Rcpp::NumericVector> offset, Rcpp::Nullable<Rcpp::NumericVector> trials, Rcpp::Nullable<Rcpp::NumericVector> mhvar_beta, Rcpp::Nullable<Rcpp::NumericVector> mhvar_theta, Rcpp::Nullable<double> r_a_prior, Rcpp::Nullable<double> r_b_prior, Rcpp::Nullable<double> sigma2_epsilon_a_prior, Rcpp::Nullable<double> sigma2_epsilon_b_prior, Rcpp::Nullable<double> sigma2_beta_prior, Rcpp::Nullable<double> sigma2_theta_a_prior, Rcpp::Nullable<double> sigma2_theta_b_prior, Rcpp::Nullable<double> alpha_a_prior, Rcpp::Nullable<double> alpha_b_prior, Rcpp::Nullable<double> r_init, Rcpp::Nullable<double> sigma2_epsilon_init, Rcpp::Nullable<Rcpp::NumericVector> beta_init, Rcpp::Nullable<Rcpp::NumericVector> theta_init, Rcpp::Nullable<double> sigma2_theta_init, Rcpp::Nullable<Rcpp::NumericVector> g_init, Rcpp::Nullable<Rcpp::NumericVector> v_init, Rcpp::Nullable<double> alpha_init, Rcpp::Nullable<Rcpp::NumericVector> psi_init, Rcpp::Nullable<double> alpha_fix, Rcpp::Nullable<double> keep_all_ind);
RcppExport SEXP _KSBound_KSBound(SEXP mcmc_samplesSEXP, SEXP m_maxSEXP, SEXP spatial_neighborsSEXP, SEXP ySEXP, SEXP xSEXP, SEXP likelihood_indicatorSEXP, SEXP offsetSEXP, SEXP trialsSEXP, SEXP mhvar_betaSEXP, SEXP mhvar_thetaSEXP, SEXP r_a_priorSEXP, SEXP r_b_priorSEXP, SEXP sigma2_epsilon_a_priorSEXP, SEXP sigma2_epsilon_b_priorSEXP, SEXP sigma2_beta_priorSEXP, SEXP sigma2_theta_a_priorSEXP, SEXP sigma2_theta_b_priorSEXP, SEXP alpha_a_priorSEXP, SEXP alpha_b_priorSEXP, SEXP r_initSEXP, SEXP sigma2_epsilon_initSEXP, SEXP beta_initSEXP, SEXP theta_initSEXP, SEXP sigma2_theta_initSEXP, SEXP g_initSEXP, SEXP v_initSEXP, SEXP alpha_initSEXP, SEXP psi_initSEXP, SEXP alpha_fixSEXP, SEXP keep_all_indSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type mcmc_samples(mcmc_samplesSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type spatial_neighbors(spatial_neighborsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type likelihood_indicator(likelihood_indicatorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type trials(trialsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type mhvar_beta(mhvar_betaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type mhvar_theta(mhvar_thetaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type r_a_prior(r_a_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type r_b_prior(r_b_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_epsilon_a_prior(sigma2_epsilon_a_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_epsilon_b_prior(sigma2_epsilon_b_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_beta_prior(sigma2_beta_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_theta_a_prior(sigma2_theta_a_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_theta_b_prior(sigma2_theta_b_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type alpha_a_prior(alpha_a_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type alpha_b_prior(alpha_b_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type r_init(r_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_epsilon_init(sigma2_epsilon_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type beta_init(beta_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_theta_init(sigma2_theta_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type g_init(g_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type v_init(v_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type alpha_init(alpha_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type psi_init(psi_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type alpha_fix(alpha_fixSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type keep_all_ind(keep_all_indSEXP);
    rcpp_result_gen = Rcpp::wrap(KSBound(mcmc_samples, m_max, spatial_neighbors, y, x, likelihood_indicator, offset, trials, mhvar_beta, mhvar_theta, r_a_prior, r_b_prior, sigma2_epsilon_a_prior, sigma2_epsilon_b_prior, sigma2_beta_prior, sigma2_theta_a_prior, sigma2_theta_b_prior, alpha_a_prior, alpha_b_prior, r_init, sigma2_epsilon_init, beta_init, theta_init, sigma2_theta_init, g_init, v_init, alpha_init, psi_init, alpha_fix, keep_all_ind));
    return rcpp_result_gen;
END_RCPP
}
// alpha_update
double alpha_update(int m_max, arma::vec v, double alpha_a, double alpha_b);
RcppExport SEXP _KSBound_alpha_update(SEXP m_maxSEXP, SEXP vSEXP, SEXP alpha_aSEXP, SEXP alpha_bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type v(vSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_a(alpha_aSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_b(alpha_bSEXP);
    rcpp_result_gen = Rcpp::wrap(alpha_update(m_max, v, alpha_a, alpha_b));
    return rcpp_result_gen;
END_RCPP
}
// beta_other_update
arma::vec beta_other_update(arma::mat x, int n, int p_x, arma::vec off_set, arma::vec w, arma::vec gamma, arma::vec theta_old, arma::vec g_old, double sigma2_beta);
RcppExport SEXP _KSBound_beta_other_update(SEXP xSEXP, SEXP nSEXP, SEXP p_xSEXP, SEXP off_setSEXP, SEXP wSEXP, SEXP gammaSEXP, SEXP theta_oldSEXP, SEXP g_oldSEXP, SEXP sigma2_betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type p_x(p_xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_old(theta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g_old(g_oldSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_beta(sigma2_betaSEXP);
    rcpp_result_gen = Rcpp::wrap(beta_other_update(x, n, p_x, off_set, w, gamma, theta_old, g_old, sigma2_beta));
    return rcpp_result_gen;
END_RCPP
}
// beta_update
Rcpp::List beta_update(arma::vec y, arma::mat x, int n, int p_x, arma::vec off_set, arma::vec beta_old, arma::vec theta_old, arma::vec g_old, double sigma2_beta, arma::vec mhvar_beta, arma::vec acctot_beta);
RcppExport SEXP _KSBound_beta_update(SEXP ySEXP, SEXP xSEXP, SEXP nSEXP, SEXP p_xSEXP, SEXP off_setSEXP, SEXP beta_oldSEXP, SEXP theta_oldSEXP, SEXP g_oldSEXP, SEXP sigma2_betaSEXP, SEXP mhvar_betaSEXP, SEXP acctot_betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type p_x(p_xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta_old(beta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_old(theta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g_old(g_oldSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_beta(sigma2_betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mhvar_beta(mhvar_betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type acctot_beta(acctot_betaSEXP);
    rcpp_result_gen = Rcpp::wrap(beta_update(y, x, n, p_x, off_set, beta_old, theta_old, g_old, sigma2_beta, mhvar_beta, acctot_beta));
    return rcpp_result_gen;
END_RCPP
}
// g_other_update
arma::vec g_other_update(arma::vec y, arma::mat x, int n, arma::vec off_set, arma::vec w, arma::vec gamma, arma::vec beta, arma::vec theta, arma::vec c, arma::vec u, arma::mat p);
RcppExport SEXP _KSBound_g_other_update(SEXP ySEXP, SEXP xSEXP, SEXP nSEXP, SEXP off_setSEXP, SEXP wSEXP, SEXP gammaSEXP, SEXP betaSEXP, SEXP thetaSEXP, SEXP cSEXP, SEXP uSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c(cSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type u(uSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(g_other_update(y, x, n, off_set, w, gamma, beta, theta, c, u, p));
    return rcpp_result_gen;
END_RCPP
}
// g_update
arma::vec g_update(arma::vec y, arma::mat x, int n, arma::vec off_set, arma::vec beta, arma::vec theta, arma::vec c, arma::vec u, arma::mat p);
RcppExport SEXP _KSBound_g_update(SEXP ySEXP, SEXP xSEXP, SEXP nSEXP, SEXP off_setSEXP, SEXP betaSEXP, SEXP thetaSEXP, SEXP cSEXP, SEXP uSEXP, SEXP pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c(cSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type u(uSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type p(pSEXP);
    rcpp_result_gen = Rcpp::wrap(g_update(y, x, n, off_set, beta, theta, c, u, p));
    return rcpp_result_gen;
END_RCPP
}
// neg_two_loglike_update
double neg_two_loglike_update(arma::vec y, arma::mat x, int n, arma::vec off_set, arma::vec tri_als, int likelihood_indicator, int r, double sigma2_epsilon, arma::vec beta, arma::vec theta, arma::vec g);
RcppExport SEXP _KSBound_neg_two_loglike_update(SEXP ySEXP, SEXP xSEXP, SEXP nSEXP, SEXP off_setSEXP, SEXP tri_alsSEXP, SEXP likelihood_indicatorSEXP, SEXP rSEXP, SEXP sigma2_epsilonSEXP, SEXP betaSEXP, SEXP thetaSEXP, SEXP gSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type tri_als(tri_alsSEXP);
    Rcpp::traits::input_parameter< int >::type likelihood_indicator(likelihood_indicatorSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_epsilon(sigma2_epsilonSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g(gSEXP);
    rcpp_result_gen = Rcpp::wrap(neg_two_loglike_update(y, x, n, off_set, tri_als, likelihood_indicator, r, sigma2_epsilon, beta, theta, g));
    return rcpp_result_gen;
END_RCPP
}
// r_update
int r_update(arma::vec y, arma::mat x, int n, arma::vec off_set, arma::vec beta, arma::vec theta, arma::vec g, int r_a, int r_b);
RcppExport SEXP _KSBound_r_update(SEXP ySEXP, SEXP xSEXP, SEXP nSEXP, SEXP off_setSEXP, SEXP betaSEXP, SEXP thetaSEXP, SEXP gSEXP, SEXP r_aSEXP, SEXP r_bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g(gSEXP);
    Rcpp::traits::input_parameter< int >::type r_a(r_aSEXP);
    Rcpp::traits::input_parameter< int >::type r_b(r_bSEXP);
    rcpp_result_gen = Rcpp::wrap(r_update(y, x, n, off_set, beta, theta, g, r_a, r_b));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_pgdraw
arma::vec rcpp_pgdraw(arma::vec b, arma::vec c);
RcppExport SEXP _KSBound_rcpp_pgdraw(SEXP bSEXP, SEXP cSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c(cSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_pgdraw(b, c));
    return rcpp_result_gen;
END_RCPP
}
// sigma2_epsilon_update
double sigma2_epsilon_update(arma::vec y, arma::mat x, int n, arma::vec off_set, arma::vec beta_old, arma::vec theta_old, arma::vec g_old, double sigma2_epsilon_a, double sigma2_epsilon_b);
RcppExport SEXP _KSBound_sigma2_epsilon_update(SEXP ySEXP, SEXP xSEXP, SEXP nSEXP, SEXP off_setSEXP, SEXP beta_oldSEXP, SEXP theta_oldSEXP, SEXP g_oldSEXP, SEXP sigma2_epsilon_aSEXP, SEXP sigma2_epsilon_bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta_old(beta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_old(theta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g_old(g_oldSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_epsilon_a(sigma2_epsilon_aSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_epsilon_b(sigma2_epsilon_bSEXP);
    rcpp_result_gen = Rcpp::wrap(sigma2_epsilon_update(y, x, n, off_set, beta_old, theta_old, g_old, sigma2_epsilon_a, sigma2_epsilon_b));
    return rcpp_result_gen;
END_RCPP
}
// sigma2_theta_update
double sigma2_theta_update(int m_max, arma::vec theta, double sigma2_theta_a, double sigma2_theta_b);
RcppExport SEXP _KSBound_sigma2_theta_update(SEXP m_maxSEXP, SEXP thetaSEXP, SEXP sigma2_theta_aSEXP, SEXP sigma2_theta_bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_theta_a(sigma2_theta_aSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_theta_b(sigma2_theta_bSEXP);
    rcpp_result_gen = Rcpp::wrap(sigma2_theta_update(m_max, theta, sigma2_theta_a, sigma2_theta_b));
    return rcpp_result_gen;
END_RCPP
}
// theta_other_update
arma::vec theta_other_update(arma::vec y, arma::mat x, int m_max, arma::vec off_set, arma::vec w, arma::vec gamma, arma::vec beta, double sigma2_theta_old, arma::vec g_old);
RcppExport SEXP _KSBound_theta_other_update(SEXP ySEXP, SEXP xSEXP, SEXP m_maxSEXP, SEXP off_setSEXP, SEXP wSEXP, SEXP gammaSEXP, SEXP betaSEXP, SEXP sigma2_theta_oldSEXP, SEXP g_oldSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_theta_old(sigma2_theta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g_old(g_oldSEXP);
    rcpp_result_gen = Rcpp::wrap(theta_other_update(y, x, m_max, off_set, w, gamma, beta, sigma2_theta_old, g_old));
    return rcpp_result_gen;
END_RCPP
}
// theta_update
Rcpp::List theta_update(arma::vec y, arma::mat x, int n, int m_max, arma::vec off_set, arma::vec beta, arma::vec theta_old, double sigma2_theta_old, arma::vec g_old, arma::vec mhvar_theta, arma::vec acctot_theta);
RcppExport SEXP _KSBound_theta_update(SEXP ySEXP, SEXP xSEXP, SEXP nSEXP, SEXP m_maxSEXP, SEXP off_setSEXP, SEXP betaSEXP, SEXP theta_oldSEXP, SEXP sigma2_theta_oldSEXP, SEXP g_oldSEXP, SEXP mhvar_thetaSEXP, SEXP acctot_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_old(theta_oldSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_theta_old(sigma2_theta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g_old(g_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mhvar_theta(mhvar_thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type acctot_theta(acctot_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(theta_update(y, x, n, m_max, off_set, beta, theta_old, sigma2_theta_old, g_old, mhvar_theta, acctot_theta));
    return rcpp_result_gen;
END_RCPP
}
// u_p_c_update
Rcpp::List u_p_c_update(int n, int m_max, arma::mat spatial_neighbors, arma::vec v, arma::vec psi, arma::vec g);
RcppExport SEXP _KSBound_u_p_c_update(SEXP nSEXP, SEXP m_maxSEXP, SEXP spatial_neighborsSEXP, SEXP vSEXP, SEXP psiSEXP, SEXP gSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type spatial_neighbors(spatial_neighborsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type v(vSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type psi(psiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g(gSEXP);
    rcpp_result_gen = Rcpp::wrap(u_p_c_update(n, m_max, spatial_neighbors, v, psi, g));
    return rcpp_result_gen;
END_RCPP
}
// v_psi_update
Rcpp::List v_psi_update(int n, int m_max, arma::mat spatial_neighbors, arma::vec g, double alpha_old);
RcppExport SEXP _KSBound_v_psi_update(SEXP nSEXP, SEXP m_maxSEXP, SEXP spatial_neighborsSEXP, SEXP gSEXP, SEXP alpha_oldSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type spatial_neighbors(spatial_neighborsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g(gSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_old(alpha_oldSEXP);
    rcpp_result_gen = Rcpp::wrap(v_psi_update(n, m_max, spatial_neighbors, g, alpha_old));
    return rcpp_result_gen;
END_RCPP
}
// w_update
Rcpp::List w_update(arma::vec y, arma::mat x, int n, arma::vec off_set, arma::vec tri_als, int likelihood_indicator, int r, arma::vec beta_old, arma::vec theta_old, arma::vec g_old);
RcppExport SEXP _KSBound_w_update(SEXP ySEXP, SEXP xSEXP, SEXP nSEXP, SEXP off_setSEXP, SEXP tri_alsSEXP, SEXP likelihood_indicatorSEXP, SEXP rSEXP, SEXP beta_oldSEXP, SEXP theta_oldSEXP, SEXP g_oldSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type tri_als(tri_alsSEXP);
    Rcpp::traits::input_parameter< int >::type likelihood_indicator(likelihood_indicatorSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta_old(beta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta_old(theta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g_old(g_oldSEXP);
    rcpp_result_gen = Rcpp::wrap(w_update(y, x, n, off_set, tri_als, likelihood_indicator, r, beta_old, theta_old, g_old));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_KSBound_KSBound", (DL_FUNC) &_KSBound_KSBound, 30},
    {"_KSBound_alpha_update", (DL_FUNC) &_KSBound_alpha_update, 4},
    {"_KSBound_beta_other_update", (DL_FUNC) &_KSBound_beta_other_update, 9},
    {"_KSBound_beta_update", (DL_FUNC) &_KSBound_beta_update, 11},
    {"_KSBound_g_other_update", (DL_FUNC) &_KSBound_g_other_update, 11},
    {"_KSBound_g_update", (DL_FUNC) &_KSBound_g_update, 9},
    {"_KSBound_neg_two_loglike_update", (DL_FUNC) &_KSBound_neg_two_loglike_update, 11},
    {"_KSBound_r_update", (DL_FUNC) &_KSBound_r_update, 9},
    {"_KSBound_rcpp_pgdraw", (DL_FUNC) &_KSBound_rcpp_pgdraw, 2},
    {"_KSBound_sigma2_epsilon_update", (DL_FUNC) &_KSBound_sigma2_epsilon_update, 9},
    {"_KSBound_sigma2_theta_update", (DL_FUNC) &_KSBound_sigma2_theta_update, 4},
    {"_KSBound_theta_other_update", (DL_FUNC) &_KSBound_theta_other_update, 9},
    {"_KSBound_theta_update", (DL_FUNC) &_KSBound_theta_update, 11},
    {"_KSBound_u_p_c_update", (DL_FUNC) &_KSBound_u_p_c_update, 6},
    {"_KSBound_v_psi_update", (DL_FUNC) &_KSBound_v_psi_update, 5},
    {"_KSBound_w_update", (DL_FUNC) &_KSBound_w_update, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_KSBound(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
