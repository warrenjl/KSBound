#include "RcppArmadillo.h"
#include "KSBound.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double alpha_update(arma::vec v,
                    double alpha_a,
                    double alpha_b){

int m_max = v.size();

double alpha_a_update = alpha_a +
                        m_max;

double alpha_b_update = alpha_b -
                        sum(log(1.00 - v));

double alpha = R::rgamma(alpha_a_update,
                         (1.00/alpha_b_update));

Rcpp::Rcout << "alpha: " << alpha << std::endl;
Rcpp::Rcout << "a: " << alpha_a_update << std::endl;
Rcpp::Rcout << "b: " << alpha_b_update << std::endl;



return(alpha);

} 
                         