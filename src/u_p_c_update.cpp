#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List u_p_c_update(arma::mat spatial_neighbors,
                        arma::vec v,
                        arma::vec psi,
                        arma::vec g){

int n = spatial_neighbors.n_rows;
int m_max = v.size();
arma::rowvec one(1); one.fill(1);
arma::mat p(n, m_max); p.fill(0);
arma::vec u(n); u.fill(0);
arma::vec c(n); c.fill(0);

arma::uvec psi_set = as<arma::uvec>(wrap(psi)) - 1;
arma::mat spatial_neighbors_psi = spatial_neighbors.cols(psi_set);
arma::rowvec v_use = as<arma::rowvec>(wrap(v));

for(int j = 0; j < n; ++j){

   arma::rowvec vw = v_use%spatial_neighbors_psi.row(j);
   arma::rowvec stick_to_right_temp = arma::cumprod(1 - vw);
   arma::rowvec subset = stick_to_right_temp.subvec(0, (m_max - 2));
   arma::rowvec stick_to_right = join_rows(one, subset);
   arma::rowvec weights = stick_to_right%vw;
   p.row(j) = weights;
   
   u(j) = R::runif(0,
                   p(j, (g(j) - 1)));
   int k = 0;
   c(j) = k;
   while((1 - u(j)) >= sum(weights.subvec(0, c(j)))){
        ++k;
        c(j) = k;
        }
   ++c(j);

   }

return Rcpp::List::create(Rcpp::Named("u")=u,
                          Rcpp::Named("p")=p,
                          Rcpp::Named("c")=c);

}



















































