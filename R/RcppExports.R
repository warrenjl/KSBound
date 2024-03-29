# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

KSBound <- function(mcmc_samples, m_max, spatial_neighbors, y, x, likelihood_indicator, offset = NULL, trials = NULL, mhvar_beta = NULL, mhvar_theta = NULL, r_a_prior = NULL, r_b_prior = NULL, sigma2_epsilon_a_prior = NULL, sigma2_epsilon_b_prior = NULL, sigma2_beta_prior = NULL, sigma2_theta_a_prior = NULL, sigma2_theta_b_prior = NULL, alpha_a_prior = NULL, alpha_b_prior = NULL, r_init = NULL, sigma2_epsilon_init = NULL, beta_init = NULL, theta_init = NULL, sigma2_theta_init = NULL, g_init = NULL, v_init = NULL, alpha_init = NULL, psi_init = NULL, alpha_fix = NULL, keep_all_ind = NULL) {
    .Call(`_KSBound_KSBound`, mcmc_samples, m_max, spatial_neighbors, y, x, likelihood_indicator, offset, trials, mhvar_beta, mhvar_theta, r_a_prior, r_b_prior, sigma2_epsilon_a_prior, sigma2_epsilon_b_prior, sigma2_beta_prior, sigma2_theta_a_prior, sigma2_theta_b_prior, alpha_a_prior, alpha_b_prior, r_init, sigma2_epsilon_init, beta_init, theta_init, sigma2_theta_init, g_init, v_init, alpha_init, psi_init, alpha_fix, keep_all_ind)
}

alpha_update <- function(m_max, v, alpha_a, alpha_b) {
    .Call(`_KSBound_alpha_update`, m_max, v, alpha_a, alpha_b)
}

beta_other_update <- function(x, n, p_x, off_set, w, gamma, theta_old, g_old, sigma2_beta) {
    .Call(`_KSBound_beta_other_update`, x, n, p_x, off_set, w, gamma, theta_old, g_old, sigma2_beta)
}

beta_update <- function(y, x, n, p_x, off_set, beta_old, theta_old, g_old, sigma2_beta, mhvar_beta, acctot_beta) {
    .Call(`_KSBound_beta_update`, y, x, n, p_x, off_set, beta_old, theta_old, g_old, sigma2_beta, mhvar_beta, acctot_beta)
}

g_other_update <- function(y, x, n, off_set, w, gamma, beta, theta, c, u, p) {
    .Call(`_KSBound_g_other_update`, y, x, n, off_set, w, gamma, beta, theta, c, u, p)
}

g_update <- function(y, x, n, off_set, beta, theta, c, u, p) {
    .Call(`_KSBound_g_update`, y, x, n, off_set, beta, theta, c, u, p)
}

neg_two_loglike_update <- function(y, x, n, off_set, tri_als, likelihood_indicator, r, sigma2_epsilon, beta, theta, g) {
    .Call(`_KSBound_neg_two_loglike_update`, y, x, n, off_set, tri_als, likelihood_indicator, r, sigma2_epsilon, beta, theta, g)
}

r_update <- function(y, x, n, off_set, beta, theta, g, r_a, r_b) {
    .Call(`_KSBound_r_update`, y, x, n, off_set, beta, theta, g, r_a, r_b)
}

rcpp_pgdraw <- function(b, c) {
    .Call(`_KSBound_rcpp_pgdraw`, b, c)
}

sigma2_epsilon_update <- function(y, x, n, off_set, beta_old, theta_old, g_old, sigma2_epsilon_a, sigma2_epsilon_b) {
    .Call(`_KSBound_sigma2_epsilon_update`, y, x, n, off_set, beta_old, theta_old, g_old, sigma2_epsilon_a, sigma2_epsilon_b)
}

sigma2_theta_update <- function(m_max, theta, sigma2_theta_a, sigma2_theta_b) {
    .Call(`_KSBound_sigma2_theta_update`, m_max, theta, sigma2_theta_a, sigma2_theta_b)
}

theta_other_update <- function(y, x, m_max, off_set, w, gamma, beta, sigma2_theta_old, g_old) {
    .Call(`_KSBound_theta_other_update`, y, x, m_max, off_set, w, gamma, beta, sigma2_theta_old, g_old)
}

theta_update <- function(y, x, n, m_max, off_set, beta, theta_old, sigma2_theta_old, g_old, mhvar_theta, acctot_theta) {
    .Call(`_KSBound_theta_update`, y, x, n, m_max, off_set, beta, theta_old, sigma2_theta_old, g_old, mhvar_theta, acctot_theta)
}

u_p_c_update <- function(n, m_max, spatial_neighbors, v, psi, g) {
    .Call(`_KSBound_u_p_c_update`, n, m_max, spatial_neighbors, v, psi, g)
}

v_psi_update <- function(n, m_max, spatial_neighbors, g, alpha_old) {
    .Call(`_KSBound_v_psi_update`, n, m_max, spatial_neighbors, g, alpha_old)
}

w_update <- function(y, x, n, off_set, tri_als, likelihood_indicator, r, beta_old, theta_old, g_old) {
    .Call(`_KSBound_w_update`, y, x, n, off_set, tri_als, likelihood_indicator, r, beta_old, theta_old, g_old)
}

