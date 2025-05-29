#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <numeric>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Utility functions
double mean_squared_error(const VectorXd& y_true, const VectorXd& y_pred) {
    return (y_true - y_pred).squaredNorm() / y_true.size();
}

double r2_score(const VectorXd& y_true, const VectorXd& y_pred) {
    double mean_y = y_true.mean();
    double total = (y_true.array() - mean_y).square().sum();
    double residual = (y_true - y_pred).squaredNorm();
    return 1.0 - residual / total;
}

void save_csv(const string& filename, const MatrixXd& X, const VectorXd& y_true, const VectorXd& y_pred) {
    ofstream file(filename);
    file << "X,True Y,Predicted Y\n";
    for (int i = 0; i < X.rows(); ++i) {
        file << X(i, 0) << "," << y_true(i) << "," << y_pred(i) << "\n";
    }
    file.close();
}

// Linear Regression
class LinearRegression {
public:
    VectorXd weights;

    void fit(const MatrixXd& X, const VectorXd& y) {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        weights = (X_bias.transpose() * X_bias).ldlt().solve(X_bias.transpose() * y);
    }

    VectorXd predict(const MatrixXd& X) const {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        return X_bias * weights;
    }
};

// Ridge Regression
class RidgeRegression {
public:
    VectorXd weights;
    double alpha;

    RidgeRegression(double alpha = 1.0) : alpha(alpha) {}

    void fit(const MatrixXd& X, const VectorXd& y) {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        MatrixXd I = MatrixXd::Identity(X_bias.cols(), X_bias.cols());
        I(0, 0) = 0; // Don't regularize bias
        weights = (X_bias.transpose() * X_bias + alpha * I).ldlt().solve(X_bias.transpose() * y);
    }

    VectorXd predict(const MatrixXd& X) const {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        return X_bias * weights;
    }
};

// Kernel Ridge Regression (RBF kernel)
class KernelRidgeRegression {
public:
    double alpha, gamma;
    MatrixXd X_train;
    VectorXd alpha_vec;

    KernelRidgeRegression(double alpha = 1.0, double gamma = 1.0) : alpha(alpha), gamma(gamma) {}

    MatrixXd rbf_kernel(const MatrixXd& A, const MatrixXd& B) const {
        MatrixXd K(A.rows(), B.rows());
        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < B.rows(); ++j) {
                K(i, j) = exp(-gamma * (A.row(i) - B.row(j)).squaredNorm());
            }
        }
        return K;
    }

    void fit(const MatrixXd& X, const VectorXd& y) {
        X_train = X;
        MatrixXd K = rbf_kernel(X, X);
        alpha_vec = (K + alpha * MatrixXd::Identity(K.rows(), K.cols())).ldlt().solve(y);
    }

    VectorXd predict(const MatrixXd& X) const {
        MatrixXd K = rbf_kernel(X, X_train);
        return K * alpha_vec;
    }
};

int main() {
    // Generate synthetic data
    int n_samples = 100;
    MatrixXd X(n_samples, 1);
    VectorXd y(n_samples);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(0, 2);
    std::normal_distribution<> noise(0, 0.5);

    for (int i = 0; i < n_samples; ++i) {
        X(i, 0) = dist(gen);
        y(i) = 4.0 + 3.0 * X(i, 0) + noise(gen);
    }

    // Train and evaluate models
    LinearRegression linear;
    RidgeRegression ridge(1.0);
    KernelRidgeRegression kernel_ridge(1.0, 5.0);

    linear.fit(X, y);
    ridge.fit(X, y);
    kernel_ridge.fit(X, y);

    VectorXd y_pred_linear = linear.predict(X);
    VectorXd y_pred_ridge = ridge.predict(X);
    VectorXd y_pred_kernel = kernel_ridge.predict(X);

    cout << "Linear -> MSE: " << mean_squared_error(y, y_pred_linear) << ", R2: " << r2_score(y, y_pred_linear) << endl;
    cout << "Ridge  -> MSE: " << mean_squared_error(y, y_pred_ridge) << ", R2: " << r2_score(y, y_pred_ridge) << endl;
    cout << "Kernel -> MSE: " << mean_squared_error(y, y_pred_kernel) << ", R2: " << r2_score(y, y_pred_kernel) << endl;

    save_csv("predictions_linear.csv", X, y, y_pred_linear);
    save_csv("predictions_ridge.csv", X, y, y_pred_ridge);
    save_csv("predictions_kernel_ridge.csv", X, y, y_pred_kernel);

    return 0;
}


// Lasso Regression (Coordinate Descent)
class LassoRegression {
public:
    VectorXd weights;
    double alpha;
    int max_iter;
    double tol;

    LassoRegression(double alpha = 0.1, int max_iter = 1000, double tol = 1e-4)
        : alpha(alpha), max_iter(max_iter), tol(tol) {}

    void fit(const MatrixXd& X, const VectorXd& y) {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        int n_samples = X_bias.rows();
        int n_features = X_bias.cols();
        weights = VectorXd::Zero(n_features);

        for (int iter = 0; iter < max_iter; ++iter) {
            VectorXd weights_old = weights;
            for (int j = 0; j < n_features; ++j) {
                double tmp = 0.0;
                for (int i = 0; i < n_samples; ++i) {
                    double dot = 0.0;
                    for (int k = 0; k < n_features; ++k) {
                        if (k != j)
                            dot += X_bias(i, k) * weights(k);
                    }
                    tmp += X_bias(i, j) * (y(i) - dot);
                }
                double rho = tmp;
                double norm_sq = X_bias.col(j).squaredNorm();

                if (j == 0) {
                    weights(j) = rho / norm_sq;
                } else {
                    if (rho < -alpha / 2)
                        weights(j) = (rho + alpha / 2) / norm_sq;
                    else if (rho > alpha / 2)
                        weights(j) = (rho - alpha / 2) / norm_sq;
                    else
                        weights(j) = 0.0;
                }
            }
            if ((weights - weights_old).lpNorm<1>() < tol)
                break;
        }
    }

    VectorXd predict(const MatrixXd& X) const {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        return X_bias * weights;
    }
};

    LassoRegression lasso(0.1);
    lasso.fit(X, y);
    VectorXd y_pred_lasso = lasso.predict(X);
    cout << "Lasso  -> MSE: " << mean_squared_error(y, y_pred_lasso) << ", R2: " << r2_score(y, y_pred_lasso) << endl;
    save_csv("predictions_lasso.csv", X, y, y_pred_lasso);
