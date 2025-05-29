#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <numeric>
#include <Eigen/Dense>
#include <algorithm>

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
    file << "X1,X2,...,True Y,Predicted Y\n";
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j)
            file << X(i, j) << (j == X.cols() - 1 ? "," : ",");
        file << y_true(i) << "," << y_pred(i) << "\n";
    }
    file.close();
}

// k-fold cross-validation
void cross_validate(const MatrixXd& X, const VectorXd& y, int k,
                    function<void(const MatrixXd&, const VectorXd&)> fit_func,
                    function<VectorXd(const MatrixXd&)> predict_func,
                    double& avg_mse, double& avg_r2) {
    int n = X.rows();
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());

    avg_mse = 0.0;
    avg_r2 = 0.0;
    for (int i = 0; i < k; ++i) {
        int start = i * n / k;
        int end = (i + 1) * n / k;

        vector<int> test_idx(indices.begin() + start, indices.begin() + end);
        vector<int> train_idx;
        for (int j = 0; j < n; ++j) {
            if (j < start || j >= end)
                train_idx.push_back(indices[j]);
        }

        MatrixXd X_train(train_idx.size(), X.cols());
        VectorXd y_train(train_idx.size());
        for (int j = 0; j < train_idx.size(); ++j) {
            X_train.row(j) = X.row(train_idx[j]);
            y_train(j) = y(train_idx[j]);
        }

        MatrixXd X_test(test_idx.size(), X.cols());
        VectorXd y_test(test_idx.size());
        for (int j = 0; j < test_idx.size(); ++j) {
            X_test.row(j) = X.row(test_idx[j]);
            y_test(j) = y(test_idx[j]);
        }

        fit_func(X_train, y_train);
        VectorXd y_pred = predict_func(X_test);
        avg_mse += mean_squared_error(y_test, y_pred);
        avg_r2 += r2_score(y_test, y_pred);
    }

    avg_mse /= k;
    avg_r2 /= k;
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
        I(0, 0) = 0;
        weights = (X_bias.transpose() * X_bias + alpha * I).ldlt().solve(X_bias.transpose() * y);
    }

    VectorXd predict(const MatrixXd& X) const {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << MatrixXd::Ones(X.rows(), 1), X;
        return X_bias * weights;
    }
};

// Kernel Ridge Regression
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

// Lasso Regression
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

int main() {
    // Generate multivariate synthetic data
    int n_samples = 150;
    int n_features = 3;
    MatrixXd X = MatrixXd::Random(n_samples, n_features);
    VectorXd y = 2.0 + X * VectorXd::LinSpaced(n_features, 1.0, 3.0) + VectorXd::Random(n_samples) * 0.5;

    LinearRegression linear;
    RidgeRegression ridge(1.0);
    LassoRegression lasso(0.1);
    KernelRidgeRegression kernel(1.0, 2.0);

    linear.fit(X, y);
    ridge.fit(X, y);
    lasso.fit(X, y);
    kernel.fit(X, y);

    VectorXd y_pred_linear = linear.predict(X);
    VectorXd y_pred_ridge = ridge.predict(X);
    VectorXd y_pred_lasso = lasso.predict(X);
    VectorXd y_pred_kernel = kernel.predict(X);

    save_csv("predictions_linear.csv", X, y, y_pred_linear);
    save_csv("predictions_ridge.csv", X, y, y_pred_ridge);
    save_csv("predictions_lasso.csv", X, y, y_pred_lasso);
    save_csv("predictions_kernel.csv", X, y, y_pred_kernel);

    cout << "Cross-validation results (5-fold):\n";
    double mse_avg, r2_avg;

    cross_validate(X, y, 5,
        [&](const MatrixXd& Xtr, const VectorXd& ytr){ linear.fit(Xtr, ytr); },
        [&](const MatrixXd& Xte){ return linear.predict(Xte); },
        mse_avg, r2_avg);
    cout << "Linear     -> MSE: " << mse_avg << ", R2: " << r2_avg << endl;

    cross_validate(X, y, 5,
        [&](const MatrixXd& Xtr, const VectorXd& ytr){ ridge.fit(Xtr, ytr); },
        [&](const MatrixXd& Xte){ return ridge.predict(Xte); },
        mse_avg, r2_avg);
    cout << "Ridge      -> MSE: " << mse_avg << ", R2: " << r2_avg << endl;

    cross_validate(X, y, 5,
        [&](const MatrixXd& Xtr, const VectorXd& ytr){ lasso.fit(Xtr, ytr); },
        [&](const MatrixXd& Xte){ return lasso.predict(Xte); },
        mse_avg, r2_avg);
    cout << "Lasso      -> MSE: " << mse_avg << ", R2: " << r2_avg << endl;

    cross_validate(X, y, 5,
        [&](const MatrixXd& Xtr, const VectorXd& ytr){ kernel.fit(Xtr, ytr); },
        [&](const MatrixXd& Xte){ return kernel.predict(Xte); },
        mse_avg, r2_avg);
    cout << "Kernel     -> MSE: " << mse_avg << ", R2: " << r2_avg << endl;

    return 0;
}
