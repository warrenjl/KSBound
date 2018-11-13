#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List v_psi_update(int m_max,
                        arma::mat spatial_neighbors,
                        arma::vec y,
                        arma::vec g,
                        double alpha_old){
 
int n = y.size();
IntegerVector full_set_temp = seq(0, (n-1));
arma::vec full_set = as<arma::vec>(full_set_temp);
IntegerVector psi_sample_set = seq(1, n);
arma::vec g_set = (g - 1);
arma::vec weights(n); weights.fill(0);
arma::vec v_temp(n); v_temp.fill(0);
double v_beta_update = 0;
arma::vec v(m_max); v.fill(0);
arma::vec psi(m_max); psi.fill(0);
   
for(int j = 0; j < max(g); ++j){

   arma::uvec A = find(g_set == j);  
   int v_alpha_update = A.size() + 1;

   for(int k = 0; k < n; ++k){
      arma::mat spatial_neighbors_rows = spatial_neighbors.rows(A);
      arma::colvec temp_col = spatial_neighbors_rows.col(k);
      weights(k) = arma::prod(temp_col);
      arma::uvec temp_set = find(spatial_neighbors.col(k) == 1);
      v_beta_update = alpha_old + 
                      sum((g.elem(temp_set) - 1) > j);


      v_temp(k) = R::rbeta(v_alpha_update,
                           v_beta_update);
      }
   arma::vec final_weights = weights/sum(weights);

   v(j) = sampleRcpp(wrap(v_temp), 
                     1, 
                     TRUE, 
                     wrap(final_weights))(0);

   arma::vec log_val(n); log_val.fill(0);
   for(int k = 0; k < n; ++k){
      arma::mat spatial_neighbors_rows = spatial_neighbors.rows(A);
      arma::colvec temp_col = spatial_neighbors_rows.col(k);
      arma::uvec temp_set = find(spatial_neighbors.col(k) == 1);

      log_val(k) = sum(log(temp_col)) + 
                   sum((g.elem(temp_set) - 1) > j)*log(1 - v(j));
      }

   arma::vec probs(n); probs.fill(0);
   for(int k = 0; k < n; ++k){
      probs(k) = 1/(sum(exp(log_val - log_val(k))));
      if(arma::is_finite(probs(k)) == 0){
        probs(k) = 0;  /*Computational Correction*/
        }
      }

   psi(j) = sampleRcpp(wrap(psi_sample_set), 
                       1, 
                       TRUE, 
                       wrap(probs))(0);

   }

arma::vec probs(n); probs.fill(1); probs = probs/n;
for(int j = max(g); j < m_max; ++j){
   v(j) = R::rbeta(1.0,
                   alpha_old);
  
   psi(j) = sampleRcpp(wrap(psi_sample_set), 
                       1, 
                       TRUE, 
                       wrap(probs))(0);
   }

return Rcpp::List::create(Rcpp::Named("v")=v,
                          Rcpp::Named("psi")=psi);

}

